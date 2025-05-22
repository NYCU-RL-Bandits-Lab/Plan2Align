import os
import glob
import json
import random
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
from huggingface_hub import hf_hub_download

def load_reward_model_and_tokenizer(rm_path, device_map, device):
    # Load reward model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        rm_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    # Load value head weights
    value_head_file = hf_hub_download(repo_id=rm_path, filename="value_head.safetensors")
    v_weights = load_file(value_head_file)
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
    chat = []
    if sys_prompt or sys_prompt != "":
        chat.append({"role": "system", "content": sys_prompt})
    chat.append({"role": "user", "content": user_prompt})

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

def run_mpc_p2a_generation(args):
    """Run both MPC-style and buffer-based P2A-style iterative refinement using a shared initial response."""
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # Load reward model
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(
        args.rm_path, {"": device}, device
    )

    # Load generation model
    gen_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    ).to(device)
    gen_model.eval()

    threshold = args.threshold

    personas = [
        {
            "mpc": "Improve the given response. Make it more concise and clear. Respond only with the improved answer.",
            "p2a": "Rewrite the following partial responses as a single improved answer that is more concise and clear. "
        },
        {
            "mpc": "Improve the given response. Make it more detailed and informative. Respond only with the improved answer.",
            "p2a": "Rewrite the following partial responses as a single improved answer that is more detailed and informative. "
        },
        {
            "mpc": "Improve the given response. Make it balanced and well-structured. Respond only with the improved answer.",
            "p2a": "Rewrite the following partial responses as a single improved answer that is more balanced and well-structured. "
        }
    ]

    def generate_mpc_candidates(prompt, last_resp):
        base = ensure_conv(prompt)
        base += f"\n\n# Previous best:\n{last_resp}"
        return [
            generate_local(gen_model, gen_tokenizer, device, persona["mpc"], base)
            for persona in personas
        ]

    nlp = spacy.load("en_core_web_sm")
    def split_sentences(text):
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def generate_p2a_candidates(prompt, buffer, buffer_size):
        num_candidates = len(buffer)
        all_segments = []
        all_scores = []
        for i, item in enumerate(buffer):
            sentences = split_sentences(item['response'])
            total_sentences = len(sentences)
            segment_size = total_sentences // buffer_size
            remainder = total_sentences % buffer_size
            segments = []
            start = 0
            for j in range(buffer_size):
                end = start + segment_size + (1 if j < remainder else 0)
                segments.append(sentences[start:end])
                start = end
            all_segments.append(segments)
            all_scores.append(item['score'])
            
        chosen_segments = [[] for _ in range(buffer_size)]
        candidate_indices = list(range(num_candidates))
        used_candidates = [False] * num_candidates
        for i in range(buffer_size):
            random.shuffle(candidate_indices)
            segment_filled = False
            for candidate_idx in candidate_indices:
                if not used_candidates[candidate_idx]:
                    if all_segments[candidate_idx][i] and all_scores[candidate_idx] >= threshold:
                        chosen_segments[i].extend(all_segments[candidate_idx][i])
                        used_candidates[candidate_idx] = True
                        segment_filled = True
                        break
            if not segment_filled:
                chosen_segments[i].append("<Complete it>")

        buffer_lines = []
        combined_response = "\n".join(["\n".join(seg) for seg in chosen_segments])
        snippet = combined_response.replace("\n", " ")
        buffer_lines.append(f"(mixed, based on {sum(used_candidates)}/{num_candidates} candidates) {snippet}")
        context = ensure_conv(prompt)
        context += "\n\nPartial responses:\n" + "\n".join(buffer_lines) 
        print(context)
        return [
            generate_local(gen_model, gen_tokenizer, device, persona["p2a"], context)
            for persona in personas
        ]

    df = pd.read_csv(args.input_file)
    os.makedirs(args.output_folder, exist_ok=True)

    for idx, row in df.iterrows():
        if idx < args.start or idx >= args.end:
            continue
        
        prompt = row['prompt']
        sys_p = ""
        base_conv = ensure_conv(prompt)
        initial_resp = generate_local(gen_model, gen_tokenizer, device, sys_p, base_conv)
        initial_score = score_response(prompt, initial_resp, rm_model, rm_tokenizer, device)

        mpc_best_resp = p2a_best_resp = initial_resp
        mpc_best_score = p2a_best_score = initial_score

        p2a_buffer = [{"response": initial_resp, "score": initial_score}]

        history = {
            0: {
                "prompt": prompt,
                'mpc_response': initial_resp,
                'mpc_score': initial_score,
                'p2a_response': initial_resp,
                'p2a_score': initial_score
            }
        }

        for it in range(1, args.max_iterations + 1):
            if it != 1:
                mpc_cands = generate_mpc_candidates(prompt, mpc_best_resp)
                mpc_scored = [(cand, score_response(prompt, cand, rm_model, rm_tokenizer, device))
                            for cand in mpc_cands]
                mpc_best_resp, mpc_best_score = max(mpc_scored, key=lambda x: x[1])

                # P2A with buffer-based completion
                p2a_cands = generate_p2a_candidates(prompt, p2a_buffer, args.buffer_size)
                p2a_scored = [(cand, score_response(prompt, cand, rm_model, rm_tokenizer, device))
                            for cand in p2a_cands]
                top_cand, top_score = max(p2a_scored, key=lambda x: x[1])
                p2a_best_resp, p2a_best_score = top_cand, top_score
            elif it == 1:
                mpc_cands = generate_mpc_candidates(prompt, mpc_best_resp)
                mpc_scored = [(cand, score_response(prompt, cand, rm_model, rm_tokenizer, device))
                            for cand in mpc_cands]
                mpc_best_resp, mpc_best_score = max(mpc_scored, key=lambda x: x[1])
                top_cand, top_score = mpc_best_resp, mpc_best_score
                p2a_best_resp, p2a_best_score = top_cand, top_score
                 
            # Update buffer
            p2a_buffer.append({"response": top_cand, "score": top_score})
            p2a_buffer = sorted(p2a_buffer, key=lambda x: x['score'], reverse=True)[:args.buffer_size]

            history[it] = {
                "prompt": prompt,
                'mpc_response': mpc_best_resp,
                'mpc_score': mpc_best_score,
                'p2a_response': p2a_best_resp,
                'p2a_score': p2a_best_score
            }

        out_path = os.path.join(args.output_folder, f"prompt_{idx}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print("Combined MPC and buffer-based P2A generation done.")

def evaluate_results(folder_path: str, it: int, output_file: str, max_index: int):
    records = []
    for i in trange(max_index):
        file_name = f'prompt_{i}.json'
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid = {int(k): v for k, v in data.items() if int(k) <= it}
        if not valid:
            continue
        best_it_mpc = max(valid, key=lambda x: valid[x]['mpc_score'])
        best_it_p2a = max(valid, key=lambda x: valid[x]['p2a_score'])
        best_mpc = valid[best_it_mpc]
        best_p2a = valid[best_it_p2a]

        prompt = best_mpc['prompt']
        mpc_response = best_mpc['mpc_response']
        p2a_response = best_p2a['p2a_response']
        mpc_score = best_mpc['mpc_score']
        p2a_score = best_p2a['p2a_score']
        records.append({
            'file': file_name,
            'prompt': prompt,
            'best_iteration_mpc': best_it_mpc,
            'best_iteration_p2a': best_it_p2a,
            'mpc_score': mpc_score,
            'p2a_score': p2a_score,
            'mpc_response': mpc_response,
            'p2a_response': p2a_response
        })
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    mpc_avg_score = df['mpc_score'].mean() if not df.empty else 0
    p2a_avg_score = df['p2a_score'].mean() if not df.empty else 0
    print(f"Evaluation complete.\nAverage mpc_score: {mpc_avg_score:.4f}")
    print(f"Evaluation complete.\nAverage p2a_score: {p2a_avg_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multiple MPC Generation Methods & Evaluation")
    parser.add_argument("--input_file", type=str, help="CSV with 'prompt' column")
    parser.add_argument("--output_folder", type=str, help="generation output name")
    parser.add_argument("--threshold", type=float, default=4, help="Reward threshold (for buffer method)")
    parser.add_argument("--max_iterations", type=int, default=3, help="Total iterations")
    parser.add_argument("--buffer_size", type=int, default=3, help="Top n response in buffer")
    parser.add_argument("--cuda_num", type=int, default=0, help="CUDA device index")
    parser.add_argument("--start", type=int, default=0, help="start index")
    parser.add_argument("--end", type=int, default=1024, help="end index")
    parser.add_argument("--rm_path", type=str, help="reward model path(hugginface)")
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
        run_mpc_p2a_generation(args)