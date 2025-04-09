import openai
from openai import OpenAI
import spacy
import pandas as pd
from collections import defaultdict
import random
import torch
import torch.nn as nn
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import shutil
import os
import subprocess
import json
from safetensors.torch import load_file
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import logging
import argparse

"""
## Dataset Requirements

The dataset must be in CSV format.  
Since segmentation is done using the SpaCy library, the program needs to specify the language and install the corresponding segmentation model. 
You can use the `lang_map` dictionary (Line 81) to define which languages are supported for translation.  
Additionally, for each CSV column, the columns need to specify the language code, such as `zh` and `en`, and a command should be used to specify which column is the source and which is the target.

## Reward Model and Preference Model

The Reward Model is implemented based on **Llama-Factory**, but due to the large size of the model, we provide an alternative **MetricX-QE** to replace the reward model.
The program is designed to work with any language model. For convenience, we offer the use of the API provided by **Deep Infra** (meta-llama/Meta-Llama-3.1-8B-Instruct) for inference. 
Please ensure you set the **API key** in the program.

## Example Command
During execution, the program will store the results of each iteration and different final translation tasks in the `MEMORY` folder.
To run the program with the necessary parameters, use the following command:

```bash
python plan2align.py \
    --input_file "valid_en_ja.csv" \
    --rm "metricx" \
    --src_language English \
    --task_language Japanese \
    --threshold 0.7 \
    --max_iterations 6 \
    --good_ref_contexts_num 5 \
    --cuda_num 0
"""

# Argument parser
parser = argparse.ArgumentParser(description="Set global variables from terminal.")
parser.add_argument("--input_file", type=str, help="Set the input file for the translation task.")
parser.add_argument("--rm", type=str, choices=['llama','metricx'], default='xcomet', help="Set the rm.")
parser.add_argument("--src_language", type=str, default="Japanese", help="Set the language for the task.")
parser.add_argument("--task_language", type=str, default="English", help="Set the language for the task.")
parser.add_argument("--threshold", type=float, default=0.7, help="Set the threshold value.")
parser.add_argument("--max_iterations", type=int, default=6, help="Set the maximum number of iterations.")
parser.add_argument("--good_ref_contexts_num", type=int, default=5, help="Set the number of good reference contexts.")
parser.add_argument("--cuda_num", type=int, default=0, help="Set the cuda.")
args = parser.parse_args()

TASK_LANGUAGE = args.task_language
SRC_LANGUAGE = args.src_language
cuda_num = args.cuda_num
csv_path = args.input_file

print(f"TASK_LANGUAGE: {TASK_LANGUAGE}")
print(f"SRC_LANGUAGE: {SRC_LANGUAGE}")
print(f"CUDA: {cuda_num}")

max_iterations = args.max_iterations
stop_memory = list(range(1, max_iterations))
MEMORY_FOLDER = (args.input_file).replace(".csv", "")
THRESHOLD = args.threshold
good_ref_contexts_num = args.good_ref_contexts_num 

device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

lang_map = {
    "English": ("en", "en_core_web_sm"),
    "Russian": ("ru", "ru_core_news_sm"),
    "German": ("de", "de_core_news_sm"),
    "Japanese": ("ja", "ja_core_news_sm"),
    "Korean": ("ko", "ko_core_news_sm"),
    "Spanish": ("es", "es_core_news_sm"),
    "Chinese": ("zh", "zh_core_web_sm")
}

def get_lang_and_nlp(language):
    if language not in lang_map:
        raise ValueError(f"Unsupported language: {language}")
    lang_code, model_name = lang_map[language]
    return lang_code, spacy.load(model_name)

src_lang, src_nlp = get_lang_and_nlp(SRC_LANGUAGE)
tgt_lang, mt_nlp = get_lang_and_nlp(TASK_LANGUAGE)

openai = OpenAI(
    api_key="your-api-key",
    base_url="",
)
MODEL_NAME= "google/gemma-2-9b-it" # "meta-llama/Meta-Llama-3.1-8B-Instruct"

################################# folder / file processing #################################

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

def delete_files_with_mt(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    for filename in os.listdir(folder_path):
        if "mt" in filename:
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

################################# reward model for ranking #################################

class metricx_RewardModel:
    def __init__(self, device):
        self.device = device
        self.json_path = os.path.join(os.getcwd(), f'{src_lang}_{tgt_lang}_json_for_metricx')
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)

    def get_entry(self, src, mt):
        return {"source": src, "hypothesis": mt, "reference": ""}

    def write_jsonl(self, src_list, mts):
        with open(os.path.join(self.json_path, 'input.jsonl'), 'w', encoding='utf-8') as output_file:
            for src, mt in zip(src_list, mts):
                entry = self.get_entry(src, mt)
                output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
    def run_command(self):
        devices_map = {'cuda:0':0, 'cuda:1':1, 'cuda:2':2, 'cuda:3':3}
        command = [
            "python", "-m", "metricx24.predict",
            "--tokenizer", "google/mt5-large",
            "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6",
            "--max_input_length", "1536",
            "--batch_size", "1",
            "--input_file", os.path.join(self.json_path, 'input.jsonl'),
            "--output_file", os.path.join(self.json_path, 'output.jsonl'),
            "--device", f"{devices_map.get(self.device, 0)}",
            "--qe"
        ]
        subprocess.run(command)

    def get_predict(self):
        scores = []
        with open(os.path.join(self.json_path, 'output.jsonl'), 'r', encoding='utf-8') as new_file:
            for line in new_file:
                entry = json.loads(line)
                score = entry.get('prediction', None)
                scores.append(score)
        return scores

    def reward_fn_batch(self, language, src_list, mts):
        self.write_jsonl(src_list, mts)
        self.run_command()
        scores = self.get_predict()
        rewards = [1 - (score / 25) for score in scores]
        return rewards

class RewardModel:
    def __init__(self, device, torch_dtype=torch.bfloat16):
        self.device = device
        self.LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.RM = AutoModelForCausalLMWithValueHead.from_pretrained(
            '../model_trained/rm',
            torch_dtype=torch_dtype
        ).to(self.device)
        self.RM.eval()
        self.RM.gradient_checkpointing_enable()
        
        value_head_weights = load_file("../LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/acl_rm/value_head.safetensors")
        new_state_dict = {
            key.replace("v_head.", "") if key.startswith("v_head.") else key: value 
            for key, value in value_head_weights.items()
        }
        self.RM.v_head.load_state_dict(new_state_dict)
    
    def _create_single_message(self, language, source, translation):
        return [
            {
                "role": "system",
                "content": "You are a helpful translator and only output the result."
            },
            {
                "role": "user",
                "content": f"### Translate this from Chinese to {language}, Chinese:\n{source}\n### {language}:"
            },
            {
                "role": "assistant",
                "content": translation
            }
        ]
    
    def _process_inputs(self, messages):
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            attention_mask = torch.ones_like(input_ids)
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
        except Exception as e:
            logging.error(f"Error processing inputs: {str(e)}")
            raise
    
    def reward_fn(self, language, source, translations):
        try:
            all_rewards = []
            for translation in translations:
                messages = self._create_single_message(language, source, translation)
                inputs = self._process_inputs(messages)
                with torch.no_grad():
                    outputs = self.RM(**inputs, return_value=True)
                    rewards = outputs[2]
                reward = rewards[0, -1].cpu().item()
                all_rewards.append(reward)
            return all_rewards
        except Exception as e:
            logging.error(f"Error in reward_fn: {str(e)}")
            raise

    
if args.rm=='llama':
    reward_model = llama_RewardModel(device=device)
elif args.rm=='metricx':
    reward_model = metricx_RewardModel(device=device)

def batch_rm_find_best_translation(evals, language):
    """
    evals: list of (src, [translation1, translation2, ...])
    Return the translation with the highest reward in each group that meets the THRESHOLD, along with its score.
    Otherwise, return (None, score), where score is the highest score in that group.
    """
    src_list = []
    mt_list = []
    counts = []
    for src, translations in evals:
        counts.append(len(translations))
        for mt in translations:
            src_list.append(src)
            mt_list.append(mt)
    rewards = reward_model.reward_fn_batch(language, src_list, mt_list)
    best_translations = []
    index = 0
    for (src, translations), count in zip(evals, counts):
        group_rewards = rewards[index: index+count]
        index += count
        if count < 2:
            if translations:
                best_translations.append((translations[0], group_rewards[0]))
            else:
                best_translations.append((None, None))
        else:
            best_index = group_rewards.index(max(group_rewards))
            best_score = group_rewards[best_index]
            if best_score >= THRESHOLD:
                best_translations.append((translations[best_index], best_score))
            else:
                best_translations.append((None, best_score))
    return best_translations

################################# generating translation #################################

def translate_with_deepinfra(source_sentence, buffer, good_sent_size, src_language, tgt_language):    
    system_prompts = [
        "You are a meticulous translator. Provide a literal, word-for-word translation that preserves the structure and meaning of each individual word.",
        "You are a professional translator. Deliver a clear, formal, and precise translation that faithfully conveys the original meaning.",
        "You are a creative and expressive translator. Render the text in a vivid and imaginative way, as if narrating a captivating story."
    ]
    
    context_prompt =  f"Below is a specialized, intermediate translation task. The input text is a mix of {src_language} and partial {tgt_language} translations. "
    context_prompt += f"In the text, some {src_language} sentences are already followed by preliminary {tgt_language} translations enclosed in parentheses. "
    context_prompt += f"These provided translations are rough references – they may be incomplete, inconsistent, or not fully aligned with the original meaning.\n\n"
    context_prompt += f"Your task is to produce an improved {tgt_language} translation according to the following guidelines:\n"
    context_prompt += f"1. **Refinement:** For sections with existing {tgt_language} translations (in parentheses), refine and polish them so that they are fluent, accurate, and coherent, fully capturing the meaning of the corresponding {src_language} text.\n"
    context_prompt += f"2. **Completion:** For sections that remain untranslated, translate the {src_language} text accurately and naturally in the specified style.\n"
    context_prompt += f"3. **Translation Order and Structure Preservation:** Maintain the original order and structure of the text. Every {src_language} sentence must appear in the same sequence as in the source text, with its corresponding {tgt_language} translation (if available) inserted immediately after it. Do not rearrange or reorder any part of the text.\n"
    context_prompt += f"4. **Consistency:** Ensure a uniform tone and style across the entire translation, adhering to the translator role specified.\n"
    context_prompt += f"5. **Final Output:** Provide the final output as a single, well-structured {tgt_language} text. Do not include any extraneous commentary, explanations, annotations, or headers – output only the translation in the correct order.\n\n"
    context_prompt += f"Note: This translation is an intermediate version that may later be merged with other translations. Focus on clarity, coherence, and fidelity to the source text.\n"

    # Process the buffer to extract relevant English translations
    processed_source = source_sentence
    if len(buffer) > 0:
        selected_keys = random.sample(buffer.keys(), min(len(buffer), good_sent_size))
        for key_sentence in selected_keys:
            key_sentence = key_sentence.strip()
            if key_sentence and (key_sentence in source_sentence) :
                translated_sentence =  buffer[key_sentence][0][0]          
                if f"\n({translated_sentence})\n" not in processed_source:
                    processed_source = processed_source.replace(
                        key_sentence, 
                        f"{key_sentence}\n({translated_sentence})\n"
                    )

    context_prompt += f"\nHere is the input data for translation:\n{processed_source}\n\n"
    context_prompt += "Apply the above guidelines to produce an improved, coherent translation that strictly follows the original order of the text.\n"
    
    if len(buffer) == 0:
        context_prompt = f"### Translate this from {src_language} to {tgt_language} and only output the result."
        context_prompt += f"\n### {src_language}:\n {source_sentence}"
        context_prompt += f"\n### {tgt_language}:\n"

    print("--------------------------------------------------------------------------------")
    print("\n context_prompt \n")
    print(context_prompt)
    print("--------------------------------------------------------------------------------")
    
    translations = []
    for prompt in system_prompts:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context_prompt}
            ]
        )
        translation = response.choices[0].message.content.strip()

        print("--------------------------------------------------------------------------------")
        print("\n rollout translation: \n")
        print(translation)
        print("--------------------------------------------------------------------------------")

        translations.append(translation)
    
    return translations

def process_buffer_sentences(source_sentences, buffer):
    translations = []    
    translation_map = {}
    for src_key, trans_list in buffer.items():
        if not trans_list or not isinstance(trans_list, list):
            continue
        src_sentences = [src_key]
        
        if len(src_sentences) > 0:
            for src_sent in src_sentences:
                if src_sent not in translation_map:
                    translation_map[src_sent] = []
                translation_map[src_sent] = trans_list[0]
    
    for src_sent in source_sentences:
        if src_sent in translation_map and translation_map[src_sent]:
            translations.append(translation_map[src_sent][0])
    return translations

def final_translate_with_deepinfra(source_sentence, source_segments, buffer, src_language, tgt_language):
    translations = process_buffer_sentences(source_segments, buffer)
    initial_translation = "\n".join(translations)

    rewrite_prompt = (
        f"Below is an initial translation of a {src_language} text into {tgt_language}. "
        f"This translation may include omissions, inaccuracies, or awkward phrasing. "
        f"Your task is to produce a refined version that is fluent, accurate, and coherent, "
        f"while faithfully preserving the full meaning of the original {src_language} text.\n\n"
        f"### Instructions:\n"
        f"1. Ensure that every detail in the original {src_language} text is accurately represented.\n"
        f"2. Correct any grammatical errors, unnatural expressions, or inconsistencies.\n"
        f"3. Improve the natural flow so that the translation reads as if written by a native speaker.\n"
        f"4. Do not add, omit, or change any essential details from the source text.\n"
        f"5. Output only the final refined translation without any additional commentary.\n\n"
        f"### Original {src_language} Text:\n{source_sentence}\n\n"
        f"### Initial {tgt_language} Translation:\n{initial_translation}\n\n"
        f"### Refined Translation:"
    )

    print("rewrite prompt:")
    print(rewrite_prompt)

    rewrite_response = openai.chat.completions.create(
        model=MODEL_NAME,  # Replace with your actual model name
        messages=[
            {"role": "system", "content": "You are a helpful translator and only output the result."},
            {"role": "user", "content": rewrite_prompt}
        ]
    )
    translation = rewrite_response.choices[0].message.content.strip()
    return translation


################################# alignment functions #################################


def save_sentences_to_txt(sentences, filename):
    i = 0
    with open(filename, "w", encoding="utf-8") as file:
        for sentence in sentences:
            print(sentence, i)
            file.write(sentence + "\n")
            i += 1

def segment_sentences_by_punctuation(text, lang):
    segmented_sentences = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            if lang == src_lang:
                doc = src_nlp(paragraph)
            if lang == tgt_lang:
                doc = mt_nlp(paragraph)
            for sent in doc.sents:
                segmented_sentences.append(sent.text.strip())
    return segmented_sentences

def generate_overlap_and_embedding(txt_file):
    overlaps_file = txt_file + ".overlaps"
    embed_file = txt_file + ".emb"
    subprocess.run(["./overlap.py", "-i", txt_file, "-o", overlaps_file, "-n", "10"])
    embed_command = [
        "$LASER/tasks/embed/embed.sh",
        overlaps_file,
        embed_file,
    ]
    subprocess.run(" ".join(embed_command), shell=True)
    return overlaps_file, embed_file

def run_vecalign(src_txt, tgt_txt, src_embed, tgt_embed):
    result = subprocess.run(
        [
            "./vecalign.py",
            "--alignment_max_size", "8",
            "--src", src_txt,
            "--tgt", tgt_txt,
            "--src_embed", src_txt + ".overlaps", src_embed,
            "--tgt_embed", tgt_txt + ".overlaps", tgt_embed,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    alignments = []
    for line in result.stdout.strip().split("\n"):
        if line:
            src_indices, tgt_indices, _ = line.split(":")
            src_indices = list(map(int, src_indices.strip("[]").split(","))) if src_indices.strip("[]") else []
            tgt_indices = list(map(int, tgt_indices.strip("[]").split(","))) if tgt_indices.strip("[]") else []
            alignments.append((src_indices, tgt_indices))
    return alignments

def compute_alignment_stats(alignment_results):
    costs = []
    zero_cost_count = 0

    for entry in alignment_results:
        try:
            cost = float(entry.split(":")[-1])  # Extract the cost value
            if cost == 0.0:
                zero_cost_count += 1
            else:
                costs.append(cost)
        except ValueError:
            continue  # Ignore invalid entries
    
    # Compute the average cost, ignoring zero-cost samples
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    zero_cost_ratio = zero_cost_count / len(alignment_results) if alignment_results else 0.0

    return avg_cost, zero_cost_ratio

def run_vecalign_explore(src_txt, tgt_txt, src_embed, tgt_embed):
    """
    Runs vecalign multiple times, exploring the best del_percentile_frac.
    Starts from 0.2 and decreases in 0.005 steps, stopping when zero-cost ratio increases sharply.
    
    :param src_txt: Source text file
    :param tgt_txt: Target text file
    :param src_embed: Source embeddings file
    :param tgt_embed: Target embeddings file
    :return: (best_del_percentile_frac, best_avg_cost, best_zero_cost_ratio, best_alignments)
    """
    del_percentile_frac = 0.2  # Starting value
    step_size = 0.005  # Exploration step
    prev_zero_cost_ratio = None
    prev_avg_cost = None

    best_avg_cost = float('inf')
    best_del_percentile_frac = del_percentile_frac
    best_zero_cost_ratio = 0.0
    best_alignments = []

    first_flag = True
    first_zero_cost_ratio = 0.0

    while del_percentile_frac > 0:
        result = subprocess.run(
            [
                "./vecalign.py",
                "--alignment_max_size", "8",
                "--del_percentile_frac", str(del_percentile_frac),
                "--src", src_txt,
                "--tgt", tgt_txt,
                "--costs_sample_size", "200000", 
                "--search_buffer_size", "20",
                "--src_embed", src_txt + ".overlaps", src_embed,
                "--tgt_embed", tgt_txt + ".overlaps", tgt_embed,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

        output_lines = result.stdout.strip().split("\n")
        avg_cost, zero_cost_ratio = compute_alignment_stats(output_lines)

        print(f"del_percentile_frac: {del_percentile_frac:.3f} | Avg Cost: {avg_cost:.6f} | Zero-Cost Ratio: {zero_cost_ratio:.6%}")

        if first_flag:
            first_zero_cost_ratio = zero_cost_ratio
            first_flag = False        

        if prev_zero_cost_ratio != 0 and prev_zero_cost_ratio is not None and (zero_cost_ratio / prev_zero_cost_ratio) > 1.5:
            print(f"Stopping exploration: Zero-cost ratio increased sharply at {del_percentile_frac:.3f}")
            break
        elif prev_zero_cost_ratio is not None and (
            (zero_cost_ratio - prev_zero_cost_ratio) > 0.15  or
            avg_cost > prev_avg_cost or
            avg_cost < 0.3 or zero_cost_ratio > 0.7
        ):
            print(f"Stopping exploration: Zero-cost ratio increased sharply at {del_percentile_frac:.3f}")
            break
        else:
            if avg_cost < best_avg_cost:
                best_avg_cost = avg_cost
                best_del_percentile_frac = del_percentile_frac
                best_zero_cost_ratio = zero_cost_ratio
                best_alignments = output_lines
        
        prev_zero_cost_ratio = zero_cost_ratio
        prev_avg_cost = avg_cost
        del_percentile_frac -= step_size

    final_avg_cost = best_avg_cost
    final_zero_cost_ratio = best_zero_cost_ratio 
    final_del_percentile_frac = best_del_percentile_frac
    final_alignments = best_alignments.copy()

    parsed_alignments = []
    for line in final_alignments:
        if line:
            src_indices, tgt_indices, _ = line.split(":")
            src_indices = list(map(int, src_indices.strip("[]").split(","))) if src_indices.strip("[]") else []
            tgt_indices = list(map(int, tgt_indices.strip("[]").split(","))) if tgt_indices.strip("[]") else []
            parsed_alignments.append((src_indices, tgt_indices))

    print("\nBest Found:")
    print(f"del_percentile_frac: {final_del_percentile_frac:.3f} | Avg Cost: {final_avg_cost:.6f} | Zero-Cost Ratio: {final_zero_cost_ratio:.6%}")

    return parsed_alignments

def standardize_common_alignments(common_alignments_list):
    # Reference alignment for standardization (use the shortest alignment set as baseline)
    reference_alignments = min(common_alignments_list, key=lambda alignments: len(alignments))

    # Standardized results to return
    standardized_results = []

    for alignments in common_alignments_list:
        standardized_alignment = []
        mt_idx_map = {tuple(src): mt for src, mt in alignments}
        for src_indices, _ in reference_alignments:  # Ignore ref_indices as it no longer exists
            # If src_indices exist in the current alignment, use them directly
            if tuple(src_indices) in mt_idx_map:
                mt_indices = mt_idx_map[tuple(src_indices)]
            else:
                # If not found, merge based on src alignment
                mt_indices = []
                for src in src_indices:
                    if (src,) in mt_idx_map:
                        mt_indices.extend(mt_idx_map[(src,)])
                # Ensure indices are unique and sorted after merging
                mt_indices = sorted(set(mt_indices))
            standardized_alignment.append((src_indices, mt_indices))
        standardized_results.append(standardized_alignment)
    return standardized_results

def generate_windows(source, translations):
    # Segment sentences
    source_segments = segment_sentences_by_punctuation(source, lang=src_lang)   

    temp_folder = f"{src_lang}_{tgt_lang}_temp"
    os.makedirs(temp_folder, exist_ok=True)

    # Generate overlaps and embeddings
    src_txt = f"{src_lang}_{tgt_lang}_temp/mpc_src.txt"
    mt_txt = f"{src_lang}_{tgt_lang}_temp/mpc_mt.txt"

    print("\n ----------------- source segmentation --------------------------- ")
    save_sentences_to_txt(source_segments, src_txt)
    print(" -------------------------------------------------------------------  \n")
    _, src_embed = generate_overlap_and_embedding(src_txt)
    mt_segments_list = [segment_sentences_by_punctuation(t, lang=tgt_lang) for t in translations]
    adjusted_mt_list = []
    
    common_alignments_list = []
    for mt_segments in mt_segments_list:
        print("\n ----------------- translation segmentation --------------------------- ")
        save_sentences_to_txt(mt_segments, mt_txt)
        print(" ------------------------------------------------------------------------  \n")
        _, mt_embed = generate_overlap_and_embedding(mt_txt)
        src_mt_alignments = run_vecalign_explore(src_txt, mt_txt, src_embed, mt_embed) # run_vecalign_explore, run_vecalign
        common_alignments_list.append(src_mt_alignments.copy())
        delete_files_with_mt(f"{src_lang}_{tgt_lang}_temp")
    
    common_alignments_list = standardize_common_alignments(common_alignments_list)

    mt_index = 0
   
    for common_alignments in common_alignments_list:
        adjusted_src = []
        adjusted_mt = []
        for src_indices, mt_indices in common_alignments:
            mt_indices = [x for x in mt_indices if x != -1]
            
            if len(src_indices) == 0:
                continue
            else:
                aligned_src = " ".join([source_segments[i] for i in src_indices])
            
            if len(mt_indices) > 0:
                aligned_mt = " ".join([mt_segments_list[mt_index][i] for i in mt_indices])
            else:
                aligned_mt = ""
            
            adjusted_src.append(aligned_src)
            adjusted_mt.append(aligned_mt)

        adjusted_mt_list.append(adjusted_mt.copy())
        mt_index += 1
    
    clear_folder(f"{src_lang}_{tgt_lang}_temp")
    return adjusted_src, adjusted_mt_list

################################# main function #################################

def saving_memory(buffer, index, iteration, final_translations_record):
    """
    Save the buffer, and final_translations_record to the Memory folder.
    """
    os.makedirs(f"{MEMORY_FOLDER}", exist_ok=True)
    buffer_file_path = f"{MEMORY_FOLDER}/buffer_{index}_iter_{iteration}.json"
    metadata_file_path = f"{MEMORY_FOLDER}/metadata_{index}_iter_{iteration}.json"

    buffer_to_save = {key: list(value) for key, value in buffer.items()}
    with open(buffer_file_path, "w", encoding="utf-8") as f:
        json.dump(buffer_to_save, f, ensure_ascii=False, indent=4)
    
    metadata = {
        "final_translations_record": final_translations_record
    }
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Buffer saved to {buffer_file_path}")
    print(f"Metadata saved to {metadata_file_path}")

def process_chunk():

    data = pd.read_csv(csv_path)
    for index, row in data.iterrows():
        print("::::::::::::::::::::::: index :::::::::::::::::::::::", index, " ::::::::::::::::::::::: index :::::::::::::::::::::::", )
        buffer = defaultdict(list)
                
        source_sentence = row[src_lang].replace('\n', ' ')
        source_segments = segment_sentences_by_punctuation(source_sentence, lang=src_lang)

        for iteration in range(max_iterations):
            print(f"\nStarting iteration {iteration + 1}/{max_iterations}...\n")
            
            if iteration in stop_memory:
                final_translations = final_translate_with_deepinfra(source_sentence, source_segments, buffer, SRC_LANGUAGE, TASK_LANGUAGE)
                print("Final Translation Method:")
                print(final_translations)
                final_translations_record = [final_translations]
                saving_memory(buffer, index, iteration, final_translations_record)
            
            if iteration == max_iterations - 1:
                break
            else:
                translations = translate_with_deepinfra(source_sentence, buffer, good_ref_contexts_num+iteration, SRC_LANGUAGE, TASK_LANGUAGE)
            
            src_windows, mt_windows_list = generate_windows(source_sentence, translations)

            ####################################### Evaluate translations and update buffer #######################################
            print("Evaluate translations and update buffer ..............")

            # First, store all sources and candidate translations as lists.
            src_context_list = list(src_windows)
            candidates_list = []
            for window_index in range(len(src_windows)):
                candidates = [mt_windows[window_index] for mt_windows in mt_windows_list]
                candidates_list.append(candidates)
            
            # Batch evaluate all candidate translations, returning the best translation and score for each source.
            best_candidate_results = batch_rm_find_best_translation(list(zip(src_context_list, candidates_list)), TASK_LANGUAGE)

            print("\n Our best candidate results:")
            print(best_candidate_results)
            print(" ------------------------------------------------------------------------  \n")

            print("\n===== Initial buffer state =====")
            for src, translations in buffer.items():
                print(f"Source '{src}': {[t[0] for t in translations]}")

            # Update the buffer for each source.
            for i, src in enumerate(src_context_list):
                best_tuple = best_candidate_results[i]  # (translation, score)
                if best_tuple[0] is not None:
                    # If the source is not yet in the buffer, initialize it.
                    if src not in buffer:
                        buffer[src] = [best_tuple]
                        print(f"[ADD] New Source '{src}' Add Translation: '{best_tuple[0]}', Score: {best_tuple[1]}")
                    else:
                        # Directly add the new translation to the buffer.
                        buffer[src].append(best_tuple)
                        print(f"[ADD] Source '{src}' Add Translation: '{best_tuple[0]}', Score: {best_tuple[1]}")
                    
                    # Sort by score to place the best translation (highest score) at the top.
                    buffer[src].sort(key=lambda x: x[1], reverse=True)
                    print(f"[UPDATE] Source '{src}' Best Translation: '{buffer[src][0][0]}'")

            print("\n===== Final buffer state =====")
            for src, translations in buffer.items():
                print(f"Source '{src}': {[t[0] for t in translations]}")


        print("Final Translation:")
        print(final_translations)

if __name__ == "__main__":
    process_chunk()