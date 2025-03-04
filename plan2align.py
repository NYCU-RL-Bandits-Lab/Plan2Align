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

"""
## Supported Translation Directions

The program currently supports the following translation directions:

- **zh / zh-tw -> en**
- **zh / zh-tw -> de**
- **zh / zh-tw -> ru**

## Dataset Requirements

The dataset must be in CSV format and **must contain a 'zh' column**. 

## Program Execution

During execution, the program will:

- Store the results of each iteration and different final translation tasks (annotation and reference methods) in the `MEMORY` folder.
- Merge the results of the final iteration into the dataset and save it as a new file.

## Reward Model and Preference Model

The Reward Model is implemented based on **Llama-Factory**, but due to the large size of the model, we provide an alternative **preference model** to replace the reward model.

The program is designed to work with any language model. For convenience, we offer the use of the API provided by **Deep Infra** (meta-llama/Meta-Llama-3.1-8B-Instruct) for inference. 

Please ensure you set the **API key** in the program.

## Example Command

To run the program with the necessary parameters, use the following command:

```bash
python plan2align.py \
    --task_language English \
    --start_index 0 \
    --end_index 30 \
    --cuda_num 0 \
    --threshold 2 \
    --max_iterations 6 \
    --good_ref_contexts_num 5 \
    --good_context_buffer_size 3 \
    --memory_folder "memory" \
    --output_suffix "t_2_d_6_chunk_0_30"
"""

# Argument parser
parser = argparse.ArgumentParser(description="Set global variables from terminal.")
parser.add_argument("--task_language", type=str, default="English", help="Set the language for the task.")
parser.add_argument("--start_index", type=int, default=0, help="Set the start index.")
parser.add_argument("--end_index", type=int, default=10, help="Set the end index.")
parser.add_argument("--cuda_num", type=int, default=0, help="Set the cuda number.")
parser.add_argument("--threshold", type=int, default=2, help="Set the threshold value.")
parser.add_argument("--max_iterations", type=int, default=6, help="Set the maximum number of iterations.")
parser.add_argument("--good_ref_contexts_num", type=int, default=5, help="Set the number of good reference contexts.")
parser.add_argument("--good_context_buffer_size", type=int, default=3, help="Set the size of good context buffer.")
parser.add_argument("--dataset", type=str, default="valid_en.csv", help="Set the dataset for inference. e.g., valid_en.csv")
parser.add_argument("--memory_folder", type=str, default="memory", help="Set the folder for memory storage.")
parser.add_argument("--output_suffix", type=str, default="", help="Suffix to append to the output file name.")
args = parser.parse_args()

# Setting variables based on arguments
TASK_LANGUAGE = args.task_language
START_INDEX = args.start_index
END_INDEX = args.end_index
cuda_num = args.cuda_num
THRESHOLD = args.threshold
max_iterations = args.max_iterations
good_ref_contexts_num = args.good_ref_contexts_num
good_context_buffer_size = args.good_context_buffer_size
output_suffix = args.output_suffix
MEMORY_FOLDER = args.memory_folder

csv_path = args.dataset
new_column_name = 'plan2align'
stop_memory = list(range(1, max_iterations))

# Output file path
output_path = f"{TASK_LANGUAGE}_t_{THRESHOLD}_d_{max_iterations}_chunk_{START_INDEX}_{END_INDEX}{output_suffix}.csv"

# Display the settings
print(f"DATASET: {csv_path}")
print(f"TASK_LANGUAGE: {TASK_LANGUAGE}")
print(f"START_INDEX: {START_INDEX}")
print(f"END_INDEX: {END_INDEX}")
print(f"CUDA: {cuda_num}")
print(f"THRESHOLD: {THRESHOLD}")
print(f"MAX_ITERATIONS: {max_iterations}")
print(f"GOOD_REF_CONTEXTS_NUM: {good_ref_contexts_num}")
print(f"GOOD_CONTEXT_BUFFER_SIZE: {good_context_buffer_size}")
print(f"MEMORY_FOLDER: {MEMORY_FOLDER}")
print(f"Output path: {output_path}")

device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

openai = OpenAI(
    api_key='your-api-key',
    base_url="",
)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

################################# path and folder preparation #################################

# Check if the folder exists, if not, create it
if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)
    print(f"Memory folder '{MEMORY_FOLDER}' has been created.")
else:
    print(f"Memory folder '{MEMORY_FOLDER}' already exists.") 

# Set the CSV path and alignment data temporary storage based on the task language.
if TASK_LANGUAGE == "English":
    if not os.path.exists(f"en_temp_{start_index}"):
        os.makedirs(f"en_temp_{start_index}")
        print(f"alignment data temporary storage, en_temp_{start_index} has been created.")
    else:
        print(f"alignment data temporary storage, en_temp_{start_index} already exists.") 
elif TASK_LANGUAGE == "Russian":
    if not os.path.exists(f"ru_temp_{start_index}"):
        os.makedirs(f"ru_temp_{start_index}")
        print(f"alignment data temporary storage, ru_temp_{start_index} has been created.")
    else:
        print(f"alignment data temporary storage, ru_temp_{start_index} already exists.") 
elif TASK_LANGUAGE == "German":
    if not os.path.exists(f"de_temp_{start_index}"):
        os.makedirs(f"de_temp_{start_index}")
        print(f"alignment data temporary storage, de_temp_{start_index} has been created.")
    else:
        print(f"alignment data temporary storage, de_temp_{start_index} already exists.") 
else:
    raise ValueError(f"Unsupported task language: {TASK_LANGUAGE}")

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

################################# preference model for ranking #################################

class PreferenceModel(nn.Module):
    def __init__(self, pretrained_model_name='google/mt5-small'):
        super(PreferenceModel, self).__init__()
        self.encoder = MT5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.encoder.config.d_model * 3, 1)  # Concatenation of source, good_mt, and bad_mt embeddings

    def forward(self, source, good_mt, bad_mt, source_mask, good_mt_mask, bad_mt_mask):
        source_embedding = self.encoder.encoder(input_ids=source, attention_mask=source_mask).last_hidden_state.mean(dim=1)
        good_mt_embedding = self.encoder.encoder(input_ids=good_mt, attention_mask=good_mt_mask).last_hidden_state.mean(dim=1)
        bad_mt_embedding = self.encoder.encoder(input_ids=bad_mt, attention_mask=bad_mt_mask).last_hidden_state.mean(dim=1)

        # Concatenate embeddings and pass through linear layer
        concat_embedding = torch.cat((source_embedding, good_mt_embedding, bad_mt_embedding), dim=-1)
        logits = self.fc(concat_embedding)
        return logits

def load_preference_model(model_path):
    model = PreferenceModel(pretrained_model_name='google/mt5-small')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()
    return model

def pm_predict_preference(source, translation0, translation1, language="English"):
    # Tokenize the inputs
    source_encoding = tokenizer(source, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(device)
    trans0_encoding = tokenizer(translation0, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(device)
    trans1_encoding = tokenizer(translation1, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(device)

    # Predict preference
    with torch.no_grad():
        preference_score = preference_model(
            source=source_encoding['input_ids'],
            good_mt=trans0_encoding['input_ids'],
            bad_mt=trans1_encoding['input_ids'],
            source_mask=source_encoding['attention_mask'],
            good_mt_mask=trans0_encoding['attention_mask'],
            bad_mt_mask=trans1_encoding['attention_mask']
        )
        preference = torch.sigmoid(preference_score).item()
    
    # Return result based on preference score
    if preference > 0.5:
        print("translation 0 is better")
        return 0
    else:
        print("translation 1 is better")
        return 1

def pm_find_best_translation(source, translations, language="English"):
    """
    Find the best translation among multiple candidates.
    Args:
        model: The loaded preference model.
        tokenizer: The tokenizer for the model.
        source: The source sentence (string).
        translations: A list of candidate translations (list of strings).
        device: The device to run the model on (CPU or GPU).
    Returns:
        The best translation string or None if no clear best is found.
    """
    
    if len(translations) < 2:
        return translations[0] if translations else None

    # Initialize scores for each translation
    scores = {''.join(translation): 0 for translation in translations}

    # Perform pairwise comparisons
    for i in range(len(translations)):
        for j in range(i + 1, len(translations)):
            translation0 = translations[i]
            translation1 = translations[j]
            # Predict preference between two translations
            result = pm_predict_preference(''.join(source), ''.join(translation0), ''.join(translation1), device)
            if result == 0:
                scores[''.join(translation0)] += 1
            elif result == 1:
                scores[''.join(translation1)] += 1

    # Find the translation with the highest score
    max_score = max(scores.values())
    best_translations = [k for k, v in scores.items() if v == max_score]

    # If there is a tie, return None
    if len(best_translations) > 1:
        return None

    return best_translations[0]

def pm_find_worst_buffer_translation(source, translations, language="English"):
    """
    Find the worst translation among multiple candidates.
    Args:
        source: The source sentence (string).
        translations: A list of candidate translations (list of strings).
        device: The device to run the model on (CPU or GPU).
    Returns:
        The worst translation string or None if no clear worst is found.
    """
    if len(translations) < 2:
        return translations[0] if translations else None

    # Initialize scores for each translation
    scores = {translation: 0 for translation in translations}

    # Perform pairwise comparisons
    for i in range(len(translations)):
        for j in range(i + 1, len(translations)):
            translation0 = translations[i]
            translation1 = translations[j]
            # Predict preference between two translations
            result = pm_predict_preference(''.join(source), translation0, translation1, device)

            if result == 0:
                scores[translation1] += 1  # translation1 is worse
            elif result == 1:
                scores[translation0] += 1  # translation0 is worse

    # Find the translation with the lowest score
    min_score = min(scores.values())
    worst_translations = [k for k, v in scores.items() if v == min_score]

    # If there is a tie, return None
    if len(worst_translations) > 1:
        return None

    return worst_translations[0]

def pm_find_best_buffer_translation(source, translations, language="English"):
    """
    Find the worst translation among multiple candidates.
    Args:
        source: The source sentence (string).
        translations: A list of candidate translations (list of strings).
        device: The device to run the model on (CPU or GPU).
    Returns:
        The worst translation string or None if no clear worst is found.
    """
    if len(translations) < 2:
        return translations[0] if translations else None

    # Initialize scores for each translation
    scores = {translation: 0 for translation in translations}

    # Perform pairwise comparisons
    for i in range(len(translations)):
        for j in range(i + 1, len(translations)):
            translation0 = translations[i]
            translation1 = translations[j]
            # Predict preference between two translations
            result = pm_predict_preference(''.join(source), translation0, translation1, device)

            if result == 0:
                scores[translation0] += 1  # translation1 is better
            elif result == 1:
                scores[translation1] += 1  # translation0 is better

    # Find the translation with the lowest score
    max_score = max(scores.values())
    best_translations = [k for k, v in scores.items() if v == max_score]

    # If there is a tie, return None
    if len(best_translations) > 1:
        return None

    return best_translations[0]

################################# reward model for ranking #################################

from safetensors.torch import load_file
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import logging

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

    def get_len(self, language, translations):
        try:
            len_ = 0
            for translation in translations:
                l = self.tokenizer(translation, return_tensors="pt").input_ids.to(device).shape[-1]
                len_ += l
            return len_
        except Exception as e:
            logging.error(f"Error in reward_fn: {str(e)}")
            raise

def get_token_length(translations, language="English"):
    token_length = reward_model.get_len(language, translations)
    return token_length

def rm_predict_preference(source, translation0, translation1, language="English"):
    translations = [translation0, translation1]
    for t_i in range(len(translations)):
        translations[t_i] = ''.join(translations[t_i]).replace('</s>',' ')
    rewards = reward_model.reward_fn(language, source.replace('</s>',' '), translations)
    best_index = rewards.index(max(rewards))
    return best_index

def rm_find_best_translation(source, translations, language="English"):
    copy_translations = translations.copy()
    
    if len(translations) < 2:
        return translations[0] if translations else None
    
    for t_i in range(len(translations)):
        translations[t_i] = ''.join(translations[t_i]).replace('</s>',' ')
    
    rewards = reward_model.reward_fn(language, ''.join(source).replace('</s>',' '), translations)
    
    print(rewards)
    
    best_index = rewards.index(max(rewards))

    print(f"Total translations length = {len(translations)}, and best translation index is: {best_index}")

    if rewards[best_index] >= THRESHOLD:
        return copy_translations[best_index]
    else:
        return None

def rm_find_worst_buffer_translation(source, translations, language="English"):
    copy_translations = translations.copy()
    
    if len(translations) < 2:
        return translations[0] if translations else None

    rewards = reward_model.reward_fn(language, ''.join(source).replace('</s>',' '), translations)
    
    print(rewards)

    worst_index = rewards.index(min(rewards))

    print(f"Total translations length = {len(translations)}, and worst translation index in buffer is: {worst_index}")

    return copy_translations[worst_index]

def rm_find_best_buffer_translation(source, translations, language="English"):
    copy_translations = translations.copy()
    
    if len(translations) < 2:
        return translations[0] if translations else None
    rewards = reward_model.reward_fn(language, ''.join(source).replace('</s>',' '), translations)
    
    print(rewards)
    
    best_index = rewards.index(max(rewards))
    
    print(f"Total translations length = {len(translations)}, and best translation index in buffer is: {best_index}")

    if rewards[best_index] >= THRESHOLD:
        return copy_translations[best_index]
    else:
        return None

################################# generating translation #################################

def translate_with_deepinfra(source_sentence, buffer, good_sent_size, language="English"):    
    system_prompts = [
        "You are a meticulous translator. Provide a literal, word-for-word translation that preserves the structure and meaning of each individual word.",
        "You are a professional translator. Deliver a clear, formal, and precise translation that faithfully conveys the original meaning.",
        "You are a creative and expressive translator. Render the text in a vivid and imaginative way, as if narrating a captivating story."
    ]
    
    context_prompt =  f"Below is a specialized, intermediate translation task. The input text is a mix of Chinese and partial {language} translations. "
    context_prompt += f"In the text, some Chinese sentences are already followed by preliminary {language} translations enclosed in parentheses. "
    context_prompt += f"These provided translations are rough references – they may be incomplete, inconsistent, or not fully aligned with the original meaning.\n\n"
    context_prompt += f"Your task is to produce an improved {language} translation according to the following guidelines:\n"
    context_prompt += f"1. **Refinement:** For sections with existing {language} translations (in parentheses), refine and polish them so that they are fluent, accurate, and coherent, fully capturing the meaning of the corresponding Chinese text.\n"
    context_prompt += f"2. **Completion:** For sections that remain untranslated, translate the Chinese text accurately and naturally in the specified style.\n"
    context_prompt += f"3. **Translation Order and Structure Preservation:** Maintain the original order and structure of the text. Every Chinese sentence must appear in the same sequence as in the source text, with its corresponding {language} translation (if available) inserted immediately after it. Do not rearrange or reorder any part of the text.\n"
    context_prompt += f"4. **Consistency:** Ensure a uniform tone and style across the entire translation, adhering to the translator role specified.\n"
    context_prompt += f"5. **Final Output:** Provide the final output as a single, well-structured {language} text. Do not include any extraneous commentary, explanations, annotations, or headers – output only the translation in the correct order.\n\n"
    context_prompt += f"Note: This translation is an intermediate version that may later be merged with other translations. Focus on clarity, coherence, and fidelity to the source text.\n"

    # Process the buffer to extract relevant English translations
    processed_source = source_sentence
    if len(buffer) > 0:
        selected_keys = random.sample(buffer.keys(), min(len(buffer), good_sent_size))
        for key in selected_keys:
            key_sentences = key.split('</s>')
            key_i = 0
            for key_sentence in key_sentences:
                key_sentence = key_sentence.strip()
                if key_sentence and (key_sentence in source_sentence) :
                    selected_sentences =  buffer[key][0]
                    if key_i >= len(selected_sentences):
                        translated_sentence = selected_sentences[-1].replace('</s>', '')
                    else:
                        translated_sentence = selected_sentences[key_i].replace('</s>', '')
                    
                    #  Ensure that the same sentence is not inserted for translation more than once.
                    if f"\n({translated_sentence})\n" not in processed_source:
                        processed_source = processed_source.replace(
                            key_sentence, 
                            f"{key_sentence}\n({translated_sentence})\n"
                        )
                key_i += 1


    context_prompt += f"\nHere is the input data for translation:\n{processed_source}\n\n"
    context_prompt += "Apply the above guidelines to produce an improved, coherent translation that strictly follows the original order of the text.\n"
    
    print("context_prompt")
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
        translations.append(translation)
    
    return translations

def process_buffer_sentences(source_sentences, buffer, key=None, idx=None, total=None):
    """
    Process the translation results in the buffer according to the original sentence order.

    Args:
        source_sentences (list): List of original Chinese sentences.
        buffer (dict): Dictionary containing the translation results.
        key (str, optional): The current key being processed.
        idx (int, optional): The current index.
        total (int, optional): The total count.

    Returns:
        list: A list of non-overlapping translation results arranged in the original order.
    """
    translations = []
    translation_map = {}
    for src_key, trans_list in buffer.items():
        if not trans_list or not isinstance(trans_list, list):
            continue
        src_sentences = [s.strip() for s in src_key.split('</s>') if s.strip()]
        if len(src_sentences) > 0:
            i = 0 
            for src_sent in src_sentences:
                if src_sent not in translation_map:
                    translation_map[src_sent] = []
                if i < len(trans_list[0]):
                    trans = trans_list[0][i].replace('</s>', '')
                else:
                    trans = trans_list[0][-1].replace('</s>', '')
                
                translation_map[src_sent].append(trans)
                i += 1
    
    for src_sent in source_sentences:
        src_sent = src_sent.replace('</s>', '')
        if src_sent in translation_map and translation_map[src_sent]:
            # Select the last translation (typically the most complete).
            translations.append(translation_map[src_sent][-1])
    
    return translations

################################# final_translate methods #################################
def remove_duplicate_paragraphs(text):
    lines = text.split("\n")
    seen = set()
    new_lines = []
    for line in reversed(lines):
        if line.strip() and line not in seen:
            seen.add(line)
            new_lines.append(line)
    return "\n".join(reversed(new_lines))

def final_translate_with_deepinfra_ann(source_sentence, source_segments, buffer, language="English"):
    translations = process_buffer_sentences(source_segments, buffer)
    initial_translation = remove_duplicate_paragraphs("\n".join(translations))

    rewrite_prompt = (
        f"Below is an initial translation of a Chinese text into {language}. "
        f"This translation may include omissions, inaccuracies, or awkward phrasing. "
        f"Your task is to produce a refined version that is fluent, accurate, and coherent, "
        f"while faithfully preserving the full meaning of the original Chinese text.\n\n"
        f"### Instructions:\n"
        f"1. Ensure that every detail in the original Chinese text is accurately represented.\n"
        f"2. Correct any grammatical errors, unnatural expressions, or inconsistencies.\n"
        f"3. Improve the natural flow so that the translation reads as if written by a native speaker.\n"
        f"4. Do not add, omit, or change any essential details from the source text.\n"
        f"5. Output only the final refined translation without any additional commentary.\n\n"
        f"### Original Chinese Text:\n{source_sentence}\n\n"
        f"### Initial {language} Translation:\n{initial_translation}\n\n"
        f"### Refined Translation:"
    )

    print("rewrite prompt (Method 1):")
    print(rewrite_prompt)

    rewrite_response = openai.chat.completions.create(
        model=MODEL_NAME, 
        messages=[
            {"role": "system", "content": "You are a helpful translator and only output the result."},
            {"role": "user", "content": rewrite_prompt}
        ]
    )
    translation = rewrite_response.choices[0].message.content.strip()
    return translation

def final_translate_with_deepinfra_ref(source_sentence, source_segments, buffer, language="English"):
    """
    Generate interleaved Chinese-English translation:
    - For each sliding window in the buffer, if translation content exists, output the original Chinese sentence first,
      then use process_buffer_sentences to extract the corresponding English translation, placing the English translation in parentheses following the Chinese sentence.
    - If the window has no translation content, only the original Chinese sentence will be output.
    - Combine all items to form the initial interleaved Chinese-English text, and then rewrite the English part using the LLaMA 3.1 8B model to improve quality,
      while ensuring that the Chinese-English format is preserved.

    Args:
        buffer (dict): A dictionary with the format {Chinese: [Translation sentence 1, Translation sentence 2, ...]}.
        language (str): The target language, default is English.

    Returns:
        str: The rewritten interleaved Chinese-English translation result.
    """

    translations = process_buffer_sentences(source_segments, buffer)
    lines = []
    for src, trans in zip(source_segments, translations):
        if src and trans:
            lines.append(src.replace('</s>', ''))
            lines.append(f"({trans})")
    initial_bilingual = "\n".join(lines)
    initial_bilingual = remove_duplicate_paragraphs(initial_bilingual)
    
    rewrite_prompt = (
        f"The following text is a bilingual translation generated from overlapping sliding windows. "
        f"Each Chinese sentence is followed by its corresponding {language} translation. "
        f"Your task is to refine the {language} portions for improved fluency, clarity, and accuracy. "
        f"Do not include the Chinese text in your output, only the refined {language} translation.\n\n"
        f"### Initial Translation ({language} only):\n{initial_bilingual}\n\n"
        f"### Refined Translation:"
    )
        
    print("rewrite prompt (Method 3):")
    print(rewrite_prompt)
    
    rewrite_response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful translator and only output the result."},
            {"role": "user", "content": rewrite_prompt}
        ]
    )
    translation = rewrite_response.choices[0].message.content.strip()
    return translation

################################# alignment functions #################################

if TASK_LANGUAGE == "English":
    mt_nlp = spacy.load("en_core_web_sm")
elif TASK_LANGUAGE == "Russian":
    mt_nlp = spacy.load("ru_core_news_sm")
elif TASK_LANGUAGE == "German":
    mt_nlp = spacy.load("de_core_news_sm")

zh_nlp = spacy.load("zh_core_web_sm")
WINDOW_SIZE = 3
SEPARATOR = "</s>"

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
            if lang == "zh":
                doc = zh_nlp(paragraph)
            else:
                doc = mt_nlp(paragraph)
            for sent in doc.sents:
                segmented_sentences.append(sent.text.strip() + SEPARATOR)
    return segmented_sentences

def generate_overlap_and_embedding(txt_file):
    overlaps_file = txt_file + ".overlaps"
    embed_file = txt_file + ".emb"
    subprocess.run(["overlap.py", "-i", txt_file, "-o", overlaps_file, "-n", "10"])
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
            "vecalign.py",
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
    """
    Computes the average alignment cost (ignoring zero-cost alignments) and the proportion of zero-cost alignments.

    :param alignment_results: List of alignment strings in the format "[src]:[tgt]:cost".
    :return: Tuple (average_cost, zero_cost_ratio)
    """
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
                "vecalign.py",
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

def clean_sentence(sentence):
    if sentence == "":
        return ""
    parts = sentence.split('</s>')
    unique_parts = list(dict.fromkeys(part.strip() for part in parts if part.strip()))
    return ' </s> '.join(unique_parts) + ' </s>'

def sliding_windows(sentences, window_size):
    cleaned_windows = []
    for i in range(len(sentences) - window_size + 1):
        window = [clean_sentence(sentence) for sentence in sentences[i:i + window_size]]
        unique_window = list(dict.fromkeys(window))
        cleaned_windows.append(unique_window)
    return cleaned_windows

def standardize_common_alignments(common_alignments_list):
    """
    Standardizes the alignments across different alignment results to ensure consistent src alignment.
    If src indices differ, merges mt alignments accordingly.

    Args:
        common_alignments_list (list): A list of alignment results, where each result is a list of tuples
                                       (src_indices, mt_indices).

    Returns:
        list: A list of standardized alignment results.
    """
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

def generate_windows(source, reference, translations, start_index):
    # Segment sentences
    source_segments = segment_sentences_by_punctuation(source, lang="zh")
    reference_segments = segment_sentences_by_punctuation(reference, lang="zh")
    
    if TASK_LANGUAGE == "English":
        lang = 'en'
    elif TASK_LANGUAGE == "Russian":
        lang = 'ru'
    elif TASK_LANGUAGE == "German":
        lang = 'de'
    
    src_txt = f"{lang}_temp_{start_index}/mpc_src.txt"
    mt_txt = f"{lang}_temp_{start_index}/mpc_mt.txt"

    save_sentences_to_txt(source_segments, src_txt)
    _, src_embed = generate_overlap_and_embedding(src_txt)
    mt_segments_list = [segment_sentences_by_punctuation(t, lang=lang) for t in translations]
    mt_windows_list = []
    
    common_alignments_list = []
    for mt_segments in mt_segments_list:
        save_sentences_to_txt(mt_segments, mt_txt)
        _, mt_embed = generate_overlap_and_embedding(mt_txt)
        src_mt_alignments = run_vecalign_explore(src_txt, mt_txt, src_embed, mt_embed) # run_vecalign_explore, run_vecalign
        common_alignments_list.append(src_mt_alignments.copy())
        delete_files_with_mt(f"{lang}_temp_{start_index}")
    
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

        src_windows = sliding_windows(adjusted_src, WINDOW_SIZE)
        mt_windows = sliding_windows(adjusted_mt, WINDOW_SIZE)
        mt_windows_list.append(mt_windows.copy())
        mt_index += 1
    
    clear_folder(f"{lang}_temp_{start_index}")
    return src_windows, mt_windows_list

################################# main function #################################

def set_translation_model(model_type="rm"):
    """
    - "rm" -> Use Reward Model
    - "pm" -> Use Preference Model
    """
    global find_best_translation, find_worst_buffer_translation, find_best_buffer_translation, predict_preference
    
    if model_type == "pm":
        
        global preference_model, tokenizer
        pm_model_path = "preference_model_small.pth"
        tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
        preference_model = load_preference_model(pm_model_path)

        print("Using Preference Model for translation ranking.")
        find_best_translation = pm_find_best_translation
        find_worst_buffer_translation = pm_find_worst_buffer_translation
        find_best_buffer_translation = pm_find_best_buffer_translation
        predict_preference = pm_predict_preference
    elif model_type == "rm":
        
        global reward_model
        reward_model = RewardModel(device=device)

        print("Using Reward Model for translation ranking.")
        find_best_translation = rm_find_best_translation
        find_worst_buffer_translation = rm_find_worst_buffer_translation
        find_best_buffer_translation = rm_find_best_buffer_translation
        predict_preference = rm_predict_preference
    else:
        raise ValueError("Invalid model type. Use 'rm' for Reward Model or 'pm' for Preference Model.")

def saving_memory(buffer, index, iteration, token_record, final_translations_record):
    """
    Save the buffer, token_record, and final_translations_record to the Memory folder.
    """
    os.makedirs(f"{MEMORY_FOLDER}/{TASK_LANGUAGE}", exist_ok=True)

    buffer_file_path = f"{MEMORY_FOLDER}/{TASK_LANGUAGE}/buffer_{index}_iter_{iteration}.json"
    metadata_file_path = f"{MEMORY_FOLDER}/{TASK_LANGUAGE}/metadata_{index}_iter_{iteration}.json"

    buffer_to_save = {key: list(value) for key, value in buffer.items()}
    with open(buffer_file_path, "w", encoding="utf-8") as f:
        json.dump(buffer_to_save, f, ensure_ascii=False, indent=4)
    
    metadata = {
        "token_record": token_record,
        "final_translations_record": final_translations_record
    }
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Buffer saved to {buffer_file_path}")
    print(f"Metadata saved to {metadata_file_path}")

def load_memory(index, iteration):
    """
    Read the buffer from the Memory folder. If no record is found, return a new buffer.
    Args:
        index (int): The current data index being processed.
        iteration (int): The iteration corresponding to the buffer to be read.
    Returns:
        buffer (defaultdict): The read buffer. If no historical record is found, an empty buffer is returned.
    """
    file_path = f"{MEMORY_FOLDER}/{TASK_LANGUAGE}/buffer_{index}_iter_{iteration}.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_buffer = json.load(f)
        print(f"Loaded buffer from {file_path}")
        return defaultdict(list, {key: list(value) for key, value in loaded_buffer.items()})
    else:
        print(f"No previous buffer found for index {index}, iteration {iteration-1}. Starting fresh.")
        return defaultdict(list)

def plan2align_process(start_index, end_index, output_path):

    data = pd.read_csv(csv_path)

    if start_index >= len(data):
        print("Starting index is out of range")
        return

    data[new_column_name + "_ann"] = ""
    data[new_column_name + "_ref"] = ""

    end_index = min(end_index, len(data))

    for index, row in data.iterrows():
        
        if index < start_index or index >= end_index:
            continue
        
        buffer = defaultdict(list)
        
        token_count = 0    
        
        source_sentence = row['zh']
        source_segments = segment_sentences_by_punctuation(source_sentence, lang="zh")

        if TASK_LANGUAGE == "English":
            lang = 'en'
        elif TASK_LANGUAGE == "Russian":
            lang = 'ru'
        elif TASK_LANGUAGE == "German":
            lang = 'de'

        reference = row[lang]
        for iteration in range(max_iterations):
            print(f"\nStarting iteration {iteration + 1}/{max_iterations}...\n")
            
            if iteration in stop_memory:
                final_translations_ann = final_translate_with_deepinfra_ann(source_sentence, source_segments, buffer, TASK_LANGUAGE)
                final_translations_ref = final_translate_with_deepinfra_ref(source_sentence, source_segments, buffer, TASK_LANGUAGE)

                token_count_ann = token_count + get_token_length(final_translations_ann, TASK_LANGUAGE)
                token_count_ref = token_count + get_token_length(final_translations_ref, TASK_LANGUAGE)

                token_record = [token_count_ann, token_count_ref]
                final_translations_record = [final_translations_ann, final_translations_ref]
                saving_memory(buffer, index, iteration, token_record, final_translations_record)
            
            if iteration == max_iterations - 1:
                break
            else:
                translations = translate_with_deepinfra(source_sentence, buffer, good_ref_contexts_num+iteration, TASK_LANGUAGE)
                token_count += get_token_length(translations, TASK_LANGUAGE)
            
            src_windows, mt_windows_list = generate_windows(source_sentence, reference, translations, start_index)

            # Evaluate translations and update buffer
            for window_index in range(len(src_windows)):
                candidates = []
                src_context = src_windows[window_index]

                for mt_windows in mt_windows_list:
                    candidates.append(mt_windows[window_index])

                if ''.join(src_context) not in buffer:
                    best_translation = find_best_translation(src_context, candidates, TASK_LANGUAGE)
                    if best_translation != None:
                        buffer[''.join(src_context)] = [best_translation]
                elif len(buffer[''.join(src_context)]) < good_context_buffer_size:
                    best_translation = find_best_translation(src_context, candidates, TASK_LANGUAGE)
                    if best_translation != None:
                        buffer[''.join(src_context)].append(best_translation)
                    best_buffer_translation = find_best_buffer_translation(src_context, buffer[''.join(src_context)], TASK_LANGUAGE)
                    if best_buffer_translation != None:
                        buffer[''.join(src_context)].remove(best_buffer_translation)
                        buffer[''.join(src_context)].insert(0, best_buffer_translation)
                else:
                    best_translation = find_best_translation(src_context, candidates, TASK_LANGUAGE)
                    weakest_translation = find_worst_buffer_translation(src_context, buffer[''.join(src_context)], TASK_LANGUAGE)
                    if best_translation != None and weakest_translation != None:
                        if predict_preference(''.join(src_context), best_translation, weakest_translation, TASK_LANGUAGE) == 0:
                            buffer[''.join(src_context)].remove(weakest_translation)
                            buffer[''.join(src_context)].append(best_translation)
                        best_buffer_translation = find_best_buffer_translation(src_context, buffer[''.join(src_context)], TASK_LANGUAGE)
                        if best_buffer_translation != None:
                            buffer[''.join(src_context)].remove(best_buffer_translation)
                            buffer[''.join(src_context)].insert(0, best_buffer_translation)
                    elif best_translation != None and weakest_translation == None:
                        if buffer[''.join(src_context)]:
                            buffer[''.join(src_context)].pop()
                        buffer[''.join(src_context)].append(best_translation)
                        
        print("Final Translation by Annotation:")
        print(final_translations_1)
        print("Final Translation by Reference:")
        print(final_translations_3)

        data.at[index, new_column_name + "_ann"] = final_translations_1    
        data.at[index, new_column_name + "_ref"] = final_translations_3
        
    data.iloc[start_index:end_index].to_csv(output_path, index=False)

if __name__ == "__main__":
    set_translation_model("pm") 
    plan2align_process(START_INDEX, ENDED_INDEX, output_path)