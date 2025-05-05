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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from safetensors.torch import load_file


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


def score_sentence(prompt, sentence, rm_model, rm_tokenizer, device):
    return reward_fn(prompt, sentence, rm_model, rm_tokenizer, device)


def score_response(prompt, response, rm_model, rm_tokenizer, device):
    return score_sentence(prompt, response, rm_model, rm_tokenizer, device)


def segment_sentences(nlp, text):
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    expanded = []
    for s in sents:
        if '\n' in s:
            expanded.extend([seg.strip() for seg in s.split('\n') if seg.strip()])
        else:
            expanded.append(s)
    return expanded

def score_n_sentence(prompt, response, rm_model, rm_tokenizer, device, nlp=None):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
        
    sentences = segment_sentences(nlp, response)
    if not sentences:
        return 0.0
    
    scores = [
        score_sentence(prompt, sent, rm_model, rm_tokenizer, device)
        for sent in sentences
    ]
    return sum(scores) / len(scores)


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


def generate_initial(gen_model, gen_tokenizer, device, nlp, prompt):
    base = ensure_conv(prompt)
    system_prompts = [
        "Write a clear and concise response to the prompt.",
        "Write a detailed and informative response to the prompt.",
        "Write a balanced and well-structured response to the prompt."
    ]
    return [generate_local(gen_model, gen_tokenizer, device, sp, base) for sp in system_prompts]


def generate_refined(gen_model, gen_tokenizer, device, nlp, prompt, segments):
    base = ensure_conv(prompt)
    if segments:
        combined = "\n".join(f"- {seg}" for seg in segments)
        base += f"\n\n# High-quality segments:\n{combined}"
        system_prompts = [
            "Craft a concise answer using the provided segments. Use only the most relevant ones, avoid repetition, and fill any gaps. Respond ONLY with the better answer.",
            "Craft a detailed answer using the provided segments. Use only the highest quality ones, avoid repetition, and fill any gaps. Respond ONLY with the better answer.",
            "Craft a balanced answer using the provided segments. Use only the most coherent ones, avoid repetition, and fill any gaps. Respond ONLY with the better answer."
        ]
    else:
        system_prompts = [
            "Provide a concise answer to the prompt. Respond ONLY with the final answer.",
            "Provide a detailed answer to the prompt. Respond ONLY with the final answer.",
            "Provide a balanced answer to the prompt. Respond ONLY with the final answer."
        ]
    return [generate_local(gen_model, gen_tokenizer, device, sp, base) for sp in system_prompts]


def generate_final_response(gen_model, gen_tokenizer, device, nlp, prompt, segments):
    base = ensure_conv(prompt)
    if segments:
        combined = "\n".join(f"- {seg}" for seg in segments)
        base += f"\n\n# All high-quality segments:\n{combined}"
    sys_p = (
        "Write a polished response by selectively using the provided segments. "
        "Choose only the most relevant ones, avoid repetition, combine similar ideas, and fill any gaps. Respond ONLY with the final answer."
    )
    return generate_local(gen_model, gen_tokenizer, device, sys_p, base)

def run_raw_generation(args):
    """Simple raw generation: one-shot with fixed system prompt."""
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
    for idx, row in df.iterrows():
        prompt = row['prompt']
        sys_p = "You are a helpful assistant."
        base = ensure_conv(prompt)
        resp = generate_local(gen_model, gen_tokenizer, device, sys_p, base)
        score = score_response(prompt, resp, rm_model, rm_tokenizer, device)
        history = {0: {'prompt': prompt, 'response': resp, 'score': score}}
        with open(f"{args.output_folder}/prompt_{idx}.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    print("Raw generation done.")

def run_best_of_n_generation(args):
    """Generate N times and pick best by reward."""
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
    for idx, row in df.iterrows():
        prompt = row['prompt']
        best_resp, best_score = None, float('-inf')
        sys_p = "You are a helpful assistant."
        base = ensure_conv(prompt)
        for i in range(args.max_iterations):
            resp = generate_local(gen_model, gen_tokenizer, device, sys_p, base)
            score = score_response(prompt, resp, rm_model, rm_tokenizer, device)
            if score > best_score:
                best_score, best_resp = score, resp
        history = {0: {'prompt': prompt, 'response': best_resp, 'score': best_score}}
        with open(f"{args.output_folder}/prompt_{idx}.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    print("Best-of-N generation done.")

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
            "Improve the above response: make it more concise and clear. Respond only with the improved answer.",
            "Improve the above response: make it more detailed and informative. Respond only with the improved answer.",
            "Improve the above response: make it balanced and well-structured. Respond only with the improved answer."
        ]
        return [generate_local(gen_model, gen_tokenizer, device, sp, base) for sp in sys_ps]
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        # initial raw
        sys_p = "You are a helpful assistant."
        base = ensure_conv(prompt)
        best_resp = generate_local(gen_model, gen_tokenizer, device, sys_p, base)
        best_score = score_response(prompt, best_resp, rm_model, rm_tokenizer, device)
        history = {}
        for it in range(args.max_iterations):
            candidates = generate_refined_mpc(prompt, best_resp)
            for cand in candidates:
                score = score_response(prompt, cand, rm_model, rm_tokenizer, device)
                if score > best_score:
                    best_score, best_resp = score, cand
            history[it] = {'prompt': prompt, 'response': best_resp, 'score': best_score}
        with open(f"{args.output_folder}/prompt_{idx}.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    print("MPC-style generation done.")


import re
def clean_segments(segments):
    cleaned = []
    for seg in segments:
        seg = seg.strip()
        seg = re.sub(r"^[\*\-\d]+\.\s*", "", seg)
        if len(seg) < 10:
            continue
        if seg.endswith(":") or seg.endswith("*"):
            continue
        if re.match(r"^\*\*[A-Za-z ]+\*\*$", seg):  # 單獨一個粗體標題
            continue
        cleaned.append(seg)
    return cleaned

def run_plan2align_generation(args):
    # Setup
    nlp = spacy.load("en_core_web_sm")
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # Reward model
    rm_path = args.rm_path
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(rm_path, {"": device}, device)

    # Generation model
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

    # Data & buffer
    df = pd.read_csv(args.input_file)
    embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    os.makedirs(args.output_folder, exist_ok=True)

    for idx, row in df.iterrows():
        prompt = row['prompt']
        segments, scores = [], {}

        # Initial gen
        inits = generate_initial(gen_model, gen_tokenizer, device, nlp, prompt)
        for resp in inits:
            for sent in segment_sentences(nlp, resp):
                if sent in scores: continue
                sc = score_sentence(prompt, sent, rm_model, rm_tokenizer, device)
                if sc >= args.threshold:
                    segments.append(sent)
                    scores[sent] = sc
        
        def dedupe_and_sort(segs, min_word_count=5, min_char_count=20, similarity_threshold=0.7):
            # 過濾太短和未打分的句子
            filtered_segs = [
                s for s in segs
                if len(s.split()) >= min_word_count and len(s) >= min_char_count and s in scores
            ]
            sorted_segs = sorted(filtered_segs, key=lambda s: scores[s], reverse=True)
            embs = embed_model.encode(sorted_segs, convert_to_numpy=True, normalize_embeddings=True)
            unique, unique_embs = [], []

            for emb, s in zip(embs, sorted_segs):
                if not unique_embs:
                    unique.append(s)
                    unique_embs.append(emb)
                else:
                    similarities = cosine_similarity([emb], unique_embs)[0]
                    if similarities.max() < similarity_threshold:
                        unique.append(s)
                        unique_embs.append(emb)

            return unique

        segments = dedupe_and_sort(segments)
        segments = clean_segments(segments)
        history = {}

        for it in range(args.max_iterations):
            # Refine
            refined = generate_refined(gen_model, gen_tokenizer, device, nlp, prompt, segments)
            for resp in refined:
                for sent in segment_sentences(nlp, resp):
                    if sent in scores: continue
                    sc = score_sentence(prompt, sent, rm_model, rm_tokenizer, device)
                    if sc >= args.threshold:
                        segments.append(sent)
                        scores[sent] = sc
            segments = dedupe_and_sort(segments)
            segments = clean_segments(segments)

            # Final
            final_resp = generate_final_response(gen_model, gen_tokenizer, device, nlp, prompt, segments)
            final_score = score_n_sentence(prompt, final_resp, rm_model, rm_tokenizer, device, nlp)
            history[it] = {"prompt": prompt, "response": final_resp, "score": final_score, "segments": segments.copy()}
            print(f"Iteration {it}: score={final_score}")

        # Save
        out_file = f"{args.output_folder}/prompt_{idx}.json"
        with open(out_file, 'w', encoding='utf-8') as fout:
            json.dump(history, fout, ensure_ascii=False, indent=2)

    print("Generation completed.")


def run_drop_plan2align_generation(args):
    # Setup
    nlp = spacy.load("en_core_web_sm")
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # Reward model
    rm_path = args.rm_path
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(rm_path, {"": device}, device)

    # Generation model
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

    # Data & buffer
    df = pd.read_csv(args.input_file)
    embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    os.makedirs(args.output_folder, exist_ok=True)

    for idx, row in df.iterrows():
        prompt = row['prompt']
        segments, scores = [], {}

        # Initial generation
        inits = generate_initial(gen_model, gen_tokenizer, device, nlp, prompt)

        for resp in inits:
            sentences = segment_sentences(nlp, resp)
            full_score = score_response(prompt, resp, rm_model, rm_tokenizer, device)

            drops = []
            for sent in sentences:
                partial_sentences = [s for s in sentences if s != sent]
                partial_resp = " ".join(partial_sentences)
                partial_score = score_response(prompt, partial_resp, rm_model, rm_tokenizer, device)
                drop = full_score - partial_score
                drops.append((sent, drop))

            # Sort sentences by score drop
            drops.sort(key=lambda x: x[1], reverse=True)

            # Pick most important sentences
            top_n = max(1, int(len(drops) // (args.max_iterations)))
            selected = [s for s, d in drops[:top_n]]

            for s in selected:
                if s not in scores:
                    scores[s] = full_score  # Save original full score as rough importance
                    segments.append(s)

        def dedupe_and_sort(segs, min_word_count=5, min_char_count=20, similarity_threshold=0.85):
            # 過濾太短和未打分的句子
            filtered_segs = [
                s for s in segs
                if len(s.split()) >= min_word_count and len(s) >= min_char_count and s in scores
            ]
            sorted_segs = sorted(filtered_segs, key=lambda s: scores[s], reverse=True)
            embs = embed_model.encode(sorted_segs, convert_to_numpy=True, normalize_embeddings=True)
            unique, unique_embs = [], []

            for emb, s in zip(embs, sorted_segs):
                if not unique_embs:
                    unique.append(s)
                    unique_embs.append(emb)
                else:
                    similarities = cosine_similarity([emb], unique_embs)[0]
                    if similarities.max() < similarity_threshold:
                        unique.append(s)
                        unique_embs.append(emb)

            return unique

        segments = dedupe_and_sort(segments)
        segments = clean_segments(segments)
        history = {}

        for it in range(args.max_iterations):
            refined = generate_refined(gen_model, gen_tokenizer, device, nlp, prompt, segments)
            for resp in refined:
                sentences = segment_sentences(nlp, resp)
                full_score = score_response(prompt, resp, rm_model, rm_tokenizer, device)

                drops = []
                for sent in sentences:
                    partial_resp = " ".join([s for s in sentences if s != sent])
                    partial_score = score_response(prompt, partial_resp, rm_model, rm_tokenizer, device)
                    drop = full_score - partial_score
                    drops.append((sent, drop))

                drops.sort(key=lambda x: x[1], reverse=True)
                top_n = max(1, int(len(drops) // (args.max_iterations)))
                selected = [s for s, d in drops[:top_n]]

                for s in selected:
                    if s not in scores:
                        scores[s] = full_score
                        segments.append(s)

            segments = dedupe_and_sort(segments)
            segments = clean_segments(segments)

            final_resp = generate_final_response(gen_model, gen_tokenizer, device, nlp, prompt, segments)
            final_score = score_response(prompt, final_resp, rm_model, rm_tokenizer, device)
            history[it] = {"prompt": prompt, "response": final_resp, "score": final_score, "segments": segments.copy()}
            print(f"Iteration {it}: score={final_score}")

        out_file = f"{args.output_folder}/prompt_{idx}.json"
        with open(out_file, 'w', encoding='utf-8') as fout:
            json.dump(history, fout, ensure_ascii=False, indent=2)

    print("Generation completed.")


def evaluate_results(folder_path: str, it: int, cuda_num: int, ):
    # Evaluation of best responses up to iteration it
    device = f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'
    rm_path = "nicolinho/QRM-Llama3.1-8B-v2"
    rm_model, rm_tokenizer = load_reward_model_and_tokenizer(rm_path, {"": device}, device)

    records = []
    for jf in glob.glob(os.path.join(folder_path, '*.json')):
        data = json.load(open(jf, 'r', encoding='utf-8'))
        valid = {int(k):v for k,v in data.items() if int(k) <= it}
        if not valid: continue
        best_it = max(valid, key=lambda x: valid[x]['score'])
        best = valid[best_it]
        prompt = best['prompt']
        response = best['response']
        old_score = best['score']
        new_score = reward_fn(prompt, response, rm_model, rm_tokenizer, device)
        records.append({
            'file': os.path.basename(jf),
            'best_iteration': best_it,
            'old_score': old_score,
            'new_score': new_score,
            'response': response.replace('\n', ' ')
        })
    df = pd.DataFrame(records)
    df.to_csv('hh_result.csv', index=False)
    print("Evaluation complete. hh_result.csv generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multiple MPC Generation Methods & Evaluation")
    parser.add_argument("--input_file", type=str, help="CSV with 'prompt' column")
    parser.add_argument("--output_folder", type=str, help="generation output name")
    parser.add_argument("--threshold", type=float, default=0.119, help="Reward threshold (for buffer method)")
    parser.add_argument("--max_iterations", type=int, default=5, help="Total iterations")
    parser.add_argument("--cuda_num", type=int, default=1, help="CUDA device index")
    parser.add_argument("--method", type=str,
                        choices=['p2a', 'drop_p2a','raw','best_of_n','mpc'], default='plan2align',
                        help="Generation method to run")
    parser.add_argument("--rm_path", type=str, default="/home/raychen/20241202/model_trained/nips_rm_hhrlhf_args", help="reward model folder path")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--eval_input_folder", type=str, help="evaluation folder name")
    parser.add_argument("--eval_it", type=int, default=5, help="Max iteration to eval")
    parser.add_argument("--eval_output_file", type=str, help="evaluation output file name")
    parser.add_argument("--eval_cuda_num", type=int, default=1, help="CUDA device index")
    args = parser.parse_args()
    if args.evaluate:
        evaluate_results(args.eval_input_folder, args.eval_it, args.eval_output_file, args.eval_cuda_num)
    else:
        if args.method=='p2a':
            run_plan2align_generation(args)
        elif args.method=='drop_p2a':
            run_drop_plan2align_generation(args)
        elif args.method=='raw':
            run_raw_generation(args)
        elif args.method=='best_of_n':
            run_best_of_n_generation(args)
        elif args.method=='mpc':
            run_mpc_generation(args)


"""
# Raw
python plan2align_hh.py --input_file hhrlhf.csv --output_folder raw --method raw

# Best-of-N
python plan2align_hh.py --input_file hhrlhf.csv --output_folder bfn --method best_of_n --rm_path "../model_trained/nips_rm_hhrlhf_args"

# MPC-style
python plan2align_hh.py --input_file hhrlhf.csv --output_folder mpc --method mpc --rm_path "../model_trained/nips_rm_hhrlhf_args"

# Plan2Align (select)
python plan2align_hh.py --input_file hhrlhf.csv --output_folder new_p2a --method p2a --rm_path "../model_trained/nips_rm_hhrlhf"

# Plan2Align (drop)
python plan2align_hh.py --input_file hhrlhf.csv --output_folder drop_p2a --method drop_p2a --rm_path "../model_trained/nips_rm_hhrlhf_args"

# evaluation 
python plan2align_hh.py --evaluate --eval_it 5 --eval_output_file p2a.csv
"""