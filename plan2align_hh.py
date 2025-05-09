import os
import glob
import json
import argparse
import pandas as pd
import spacy
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import AutoModelForCausalLMWithValueHead
from safetensors.torch import load_file
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_reward_model_and_tokenizer(rm_path, device_map, device):
    # Load reward model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        rm_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    # Load value head weights
    v_weights = load_file(os.path.join(rm_path, "value_head.safetensors"))
    new_state_dict = {}
    for key, value in v_weights.items():
        if key.startswith("v_head."):
            new_key = key.replace("v_head.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.v_head.load_state_dict(new_state_dict)
    model.eval()
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(rm_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def reward_fn(prompt, response, model, tokenizer, device):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    # move to device
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs.to(device)}
    else:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_value=True)
        reward = outputs[2][:, -1].item()
    return reward

def score_response(prompt, response, rm_model, rm_tokenizer, device):
    return reward_fn(prompt, response, rm_model, rm_tokenizer, device)

def ensure_conv(prompt):
    base = prompt.strip()
    if not base.startswith("Human:"):
        base = "Human: " + base
    if not base.endswith("Assistant:"):
        base += "\nAssistant:"
    return base

def generate_local(gen_model, gen_tokenizer, device, sys_prompt, user_prompt,
                   max_new_tokens=1024, temperature=0.7, top_p=0.9):
    chat = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    inputs = gen_tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs.to(device)}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature>0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=gen_tokenizer.eos_token_id
        )
    seq = out[0]
    inp_len = inputs["input_ids"].shape[1]
    text = gen_tokenizer.decode(seq[inp_len:], skip_special_tokens=True).strip()
    if text.lower().startswith("assistant"):
        text = text[len("assistant"):].lstrip()
    return text

def run_mpc_generation(args):
    """Iterative MPC-style without buffer: refine best response each iter."""
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"
    # Reward model
    rm_path = args.rm_path
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(rm_path, {"": device}, device)

    gen_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"":device},
        trust_remote_code=True
    ).to(device);
    gen_model.eval()
    df = pd.read_csv(args.input_file)
    os.makedirs(args.output_folder, exist_ok=True)
    def generate_refined_mpc(prompt, last_resp):
        base = ensure_conv(prompt)
        base += f"\n\n# Previous best:\n{last_resp}"
        sys_ps = [
            "You are an Accuracy Checker. Given the previous best response, revise it to correct any factual or logical errors and improve its accuracy. Output only the improved response.",
            "You are a Safety Auditor. Given the previous best response, revise it to remove biased or unsafe content, and ensure it is neutral and respectful. Output only the improved response.",
            "You are a Helpfulness Enhancer. Given the previous best response, revise it to add helpful examples, clarify explanations, and increase its usefulness. Output only the improved response."
        ]

        return [generate_local(gen_model, gen_tokenizer, device, sp, base) for sp in sys_ps]
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        # initial raw
        sys_p = "You are a helpful assistant. Output only the response."
        base = ensure_conv(prompt)
        best_resp = generate_local(gen_model, gen_tokenizer, device, sys_p, base)
        best_score = score_response(prompt, best_resp, rm_model, rm_tokenizer, device)
        history = {0: {'prompt': prompt, 'response': best_resp, 'score': best_score}}
        for it in range(1, args.max_iterations + 1):
            candidates = generate_refined_mpc(prompt, best_resp)
            scored_candidates = [
                (cand, score_response(prompt, cand, rm_model, rm_tokenizer, device))
                for cand in candidates
            ]
            best_resp, best_score = max(scored_candidates, key=lambda x: x[1])
            history[it] = {'prompt': prompt, 'response': best_resp, 'score': best_score}
        with open(f"{args.output_folder}/prompt_{idx}.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    print("MPC-style generation done.")

def run_p2a_generation(args):
    """
    P2A generation with a fixed-size buffer of top responses:
    - No planning stage.
    - Maintain a buffer (size=3) of the highest-scoring responses seen so far.
    - On each iteration, let the LLM see the buffer and their scores (with threshold guidance).
    - For each persona, ask for a full revised response based on the buffer and threshold guidance.
    - Update the buffer with new candidates, keeping only the top 3.
    - Always update to the best current candidate (avoid local optima).
    """
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # Load reward model
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
        args.rm_path, {"": device}, device
    )

    # Load generation model
    gen_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_name)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    ).to(device)
    gen_model.eval()

    # Score threshold guidance
    threshold = 3.5
    threshold_note = (
        f"Score guidance: responses scoring >= {threshold} are considered good; "
        f"those below {threshold} are considered poor. Avoid using directions from "
        f"low-scoring responses (< {threshold}) as primary improvement targets."
    )

    # Persona definitions with reward-maximization emphasis
    personas = [
        (
            "Accuracy Checker",
            "You are an Accuracy Checker whose goal is to maximize the reward score. "
            + threshold_note
            + " Given top responses and their scores, rewrite to fix factual or logical errors and improve accuracy, focusing on changes that will increase the response's reward score. Output only the revised response."
        ),
        (
            "Safety Auditor",
            "You are a Safety Auditor whose goal is to maximize the reward score. "
            + threshold_note
            + " Given top responses and their scores, rewrite to remove biased or unsafe content, ensure neutrality, and increase the response's reward score. Output only the revised response."
        ),
        (
            "Helpfulness Enhancer",
            "You are a Helpfulness Enhancer whose goal is to maximize the reward score. "
            + threshold_note
            + " Given top responses and their scores, rewrite to add examples, clarify explanations, increase usefulness, and maximize the response's reward score. Output only the revised response."
        ),
    ]

    # Prepare I/O
    df = pd.read_csv(args.input_file)
    os.makedirs(args.output_folder, exist_ok=True)

    for idx, row in df.iterrows():
        prompt = row["prompt"]
        conv = ensure_conv(prompt)

        # 1) Initial generation
        init_sys = "You are a helpful assistant. Output only the response."
        response = generate_local(gen_model, gen_tokenizer, device, init_sys, conv)
        best_score = score_response(prompt, response, rm_model, rm_tokenizer, device)

        # Initialize buffer with the first response
        buffer = [{"response": response, "score": best_score}]

        # History: iteration -> { ... }
        history = {0: {"prompt": prompt, "response": response, "score": best_score}}

        # 2) Iterative refinement with buffer
        for it in range(1, args.max_iterations + 1):
            # Build buffer context string
            buffer_lines = []
            for i, entry in enumerate(buffer):
                resp_clean = entry['response'].replace("\n", " ")
                buffer_lines.append(f"{i}: (score={entry['score']:.4f}) {resp_clean}")
            buffer_context = "\n".join(buffer_lines)

            candidates = []
            # Generate candidate revisions per persona
            for persona_name, persona_sys in personas:
                full_context = (
                    conv
                    + "\n\n" + threshold_note
                    + "\n\nTop responses and scores:\n"
                    + buffer_context
                    + f"\n\nPlease provide a fully revised response (persona: {persona_name}):"
                )
                revised = generate_local(
                    gen_model,
                    gen_tokenizer,
                    device,
                    persona_sys,
                    full_context
                )
                score = score_response(prompt, revised, rm_model, rm_tokenizer, device)
                candidates.append({
                    "persona": persona_name,
                    "response": revised,
                    "score": score
                })

            # Update buffer with new candidates
            for cand in candidates:
                buffer.append({"response": cand['response'], "score": cand['score']})
            # Keep top 3 by score
            buffer = sorted(buffer, key=lambda x: x['score'], reverse=True)[:3]

            # Select best candidate to update
            best_cand = buffer[0]
            response = best_cand['response']
            best_score = best_cand['score']

            # Record iteration
            history[it] = {
                "prompt": prompt,
                "response": response,
                "score": best_score,
                "buffer": buffer.copy(),
                "candidates": candidates
            }

        # 3) Save full history
        out_path = os.path.join(args.output_folder, f"prompt_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print("P2A-style generation done.")

def eval_reward_fn(prompt, response, model, tokenizer, device):
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        reward = outputs.logits.squeeze(-1).item()
    return reward

def evaluate_results(folder_path: str, it: int, output_file: str, max_index: int, cuda_num: int):
    # Evaluation of best responses up to iteration it
    device = f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'
    device = f'cuda:{cuda_num}'
    path = "nicolinho/QRM-Llama3.1-8B-v2" # "argsearch/llama-7b-rm-float32", "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", "nicolinho/QRM-Llama3.1-8B-v2" "OpenAssistant/reward-model-deberta-v3-large-v2"
    rm_model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    rm_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    records = []
    for i in trange(max_index):
        file_name = f'prompt_{i}.json'
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid = {int(k): v for k, v in data.items() if int(k) == it}
        
        if not valid:
            continue
        best_it = max(valid, key=lambda x: valid[x]['score'])
        best = valid[best_it]
        prompt = best['prompt']
        response = best['response']
        rm_score = best['score']
        eval_score = eval_reward_fn(prompt, response, rm_model, rm_tokenizer, device)
        records.append({
            'file': file_name,
            'prompt': prompt,
            'best_iteration': best_it,
            'rm_score': rm_score,
            'eval_score': eval_score,
            'response': response,
            folder_path: response
        })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    avg_rm_score = df['old_score'].mean() if not df.empty else 0
    avg_eval_score = df['new_score'].mean() if not df.empty else 0
    print(f"Evaluation complete.\nAverage rm_score: {avg_rm_score:.4f}")
    print(f"Evaluation complete.\nAverage eval_score: {avg_eval_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multiple MPC Generation Methods & Evaluation")
    parser.add_argument("--input_file", type=str, help="CSV with 'prompt' column")
    parser.add_argument("--output_folder", type=str, help="generation output name")
    parser.add_argument("--threshold", type=float, default=0.119, help="Reward threshold (for buffer method)")
    parser.add_argument("--max_iterations", type=int, default=5, help="Total iterations")
    parser.add_argument("--cuda_num", type=int, default=0, help="CUDA device index")
    parser.add_argument("--method", type=str,
                        choices=['p2a','mpc'], default='plan2align',
                        help="Generation method to run")
    parser.add_argument("--rm_path", type=str, default="/home/raychen/20241202/model_trained/nips_rm_hhrlhf_args", help="reward model folder path")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--eval_input_folder", type=str, help="evaluation folder name")
    parser.add_argument("--eval_it", type=int, default=5, help="Max iteration to eval")
    parser.add_argument("--eval_output_file", type=str, help="evaluation output file name", default='hh_eval_result.csv')
    parser.add_argument("--eval_range", type=int, default=2, help="evaluation stop index")
    parser.add_argument("--eval_cuda_num", type=int, default=2, help="CUDA device index")
    
    args = parser.parse_args()
    if args.evaluate:
        evaluate_results(args.eval_input_folder, args.eval_it, args.eval_output_file, args.eval_range, args.eval_cuda_num)
    else:
        if args.method=='p2a':
            run_p2a_generation(args)
        elif args.method=='mpc':
            run_mpc_generation(args)

"""
# Vanilla MPC
python plan2align_hh.py --input_file hhrlhf.csv --output_folder v-mpc --method mpc --cuda_num 1

# Plan2Align
python plan2align_hh.py --input_file hhrlhf.csv --output_folder p2a --method p2a --cuda_num 2

# evaluation 
python plan2align_hh.py --evaluate --eval_input_folder p2a --eval_it 5 --eval_range 400 --eval_cuda_num 0
"""