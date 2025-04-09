#!/usr/bin/env python3
"""
Long Context Evaluation

This script performs the following steps:
  1. Reads a specified CSV file containing the evaluation data.
  2. Segments source, reference, and MT texts into sentences and sliding windows.
  3. Generates overlaps and embeddings for alignment.
  4. Runs vector alignment exploration and computes COMET and COMET-QE scores.
  5. Aggregates the scores and saves the results.
"""

import os
import re
import json
import csv
import spacy
import torch
import random
import argparse
import numpy as np
import pandas as pd
import tempfile
import subprocess
import unicodedata
from multiprocessing import Pool
import datetime
from typing import Optional

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set the global random seed for reproducibility.

    Args:
        seed (int): Random seed (default is 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text: str) -> str:
    """
    Normalize text using Unicode normalization (NFKC) to convert full-width characters to half-width.
    Additional normalization (e.g., lowercasing) can be added if needed.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    normalized = unicodedata.normalize("NFKC", text)
    # Uncomment the following if lowercase conversion is desired:
    # normalized = normalized.lower()
    return normalized


def segment_sentences_by_punctuation(text: str, lang: str) -> list:
    """
    Segment text into sentences based on punctuation and add an end-of-sentence separator.

    Args:
        text (str): Input text (may contain multiple paragraphs).
        lang (str): Language code (e.g., "zh", "en", "ru", "de").

    Returns:
        list: List of segmented sentences with the SEPARATOR appended.
    """
    segmented_sentences = []
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            if lang == SRC_LANG:
                doc = src_nlp(paragraph)
            else:
                doc = mt_nlp(paragraph)
            for sent in doc.sents:
                segmented_sentences.append(normalize_text(sent.text.strip()) + SEPARATOR)
    return segmented_sentences


def preprocess_sentences(sentences: list) -> str:
    """
    Preprocess sentences by removing the end-of-sentence token and joining them with newline characters.

    Args:
        sentences (list): List of sentences.

    Returns:
        str: Preprocessed text.
    """
    processed = [sentence.replace(SEPARATOR, "").strip() for sentence in sentences]
    return "\n".join(processed)


def generate_overlap_and_embedding(text: str) -> tuple:
    """
    Generate overlap and embedding data from text using temporary files.

    Args:
        text (str): Input text.

    Returns:
        tuple: (overlap_content (str), embeddings_content (bytes))
    """
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".txt") as txt_file:
        txt_file.write(text)
        txt_file.flush()
        txt_filename = txt_file.name
        overlaps_file = txt_filename + ".overlaps"
        embed_file = txt_filename + ".emb"

        # Generate overlap data
        subprocess.run(["./overlap.py", "-i", txt_filename, "-o", overlaps_file, "-n", "10"], check=True)
        # Generate embedding data
        subprocess.run(" ".join(["$LASER/tasks/embed/embed.sh", overlaps_file, embed_file]),
                       shell=True, check=True)

        with open(embed_file, "rb") as f:
            embeddings_content = f.read()
        with open(overlaps_file, "r", encoding="utf-8") as f:
            overlap_content = f.read()

    for need_to_del_file in [overlaps_file, embed_file]:
        try:
            os.remove(need_to_del_file)
            print(f"Removed file: {need_to_del_file}")
        except Exception as e:
            print(f"Error removing {need_to_del_file}: {e}")

    return overlap_content, embeddings_content


def compute_alignment_stats(alignment_results: list) -> tuple:
    """
    Compute the average alignment cost (ignoring zero-cost alignments) and the zero-cost ratio.

    Args:
        alignment_results (list): List of alignment result strings in the format "[src]:[tgt]:cost".

    Returns:
        tuple: (average_cost (float), zero_cost_ratio (float))
    """
    costs = []
    zero_cost_count = 0

    for entry in alignment_results:
        try:
            cost = float(entry.split(":")[-1])
            if cost == 0.0:
                zero_cost_count += 1
            else:
                costs.append(cost)
        except ValueError:
            continue

    avg_cost = sum(costs) / len(costs) if costs else 0.0
    zero_cost_ratio = zero_cost_count / len(alignment_results) if alignment_results else 0.0

    return avg_cost, zero_cost_ratio


def run_vecalign_explore(src_text: str, tgt_text: str, src_overlap: str, tgt_overlap: str,
                         src_embed: bytes, tgt_embed: bytes) -> list:
    """
    Explore the best vector alignment parameters and return the best alignments.

    Args:
        src_text (str): Source text.
        tgt_text (str): Target text.
        src_overlap (str): Overlap data for the source.
        tgt_overlap (str): Overlap data for the target.
        src_embed (bytes): Embedding data for the source.
        tgt_embed (bytes): Embedding data for the target.

    Returns:
        list: Parsed best alignments as a list of tuples [(src_indices, tgt_indices), ...].
    """
    del_percentile_frac = 0.2
    step_size = 0.005
    prev_zero_cost_ratio = None
    prev_avg_cost = None

    best_avg_cost = float('inf')
    best_del_percentile_frac = del_percentile_frac
    best_zero_cost_ratio = 0.0
    best_alignments = []

    first_flag = True

    with tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".txt") as src_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".txt") as tgt_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".overlaps") as src_overlap_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="w+", encoding="utf-8", suffix=".overlaps") as tgt_overlap_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="wb", suffix=".emb") as src_embed_file, \
         tempfile.NamedTemporaryFile(delete=True, mode="wb", suffix=".emb") as tgt_embed_file:

        src_file.write(src_text)
        src_file.flush()
        tgt_file.write(tgt_text)
        tgt_file.flush()

        src_overlap_file.write(src_overlap)
        src_overlap_file.flush()
        tgt_overlap_file.write(tgt_overlap)
        tgt_overlap_file.flush()

        src_embed_file.write(src_embed)
        src_embed_file.flush()
        tgt_embed_file.write(tgt_embed)
        tgt_embed_file.flush()

        while del_percentile_frac > 0:
            result = subprocess.run(
                [
                    "./vecalign.py",
                    "--alignment_max_size", "8",
                    "--del_percentile_frac", str(del_percentile_frac),
                    "--src", src_file.name,
                    "--tgt", tgt_file.name,
                    "--src_embed", src_overlap_file.name, src_embed_file.name,
                    "--tgt_embed", tgt_overlap_file.name, tgt_embed_file.name,
                ],
                stdout=subprocess.PIPE,
                text=True,
            )

            output_lines = result.stdout.strip().split("\n")
            avg_cost, zero_cost_ratio = compute_alignment_stats(output_lines)
            print(f"del_percentile_frac: {del_percentile_frac:.3f} | Avg Cost: {avg_cost:.6f} | Zero-Cost Ratio: {zero_cost_ratio:.2%}")

            if first_flag:
                first_flag = False

            if prev_zero_cost_ratio is not None and prev_zero_cost_ratio != 0 and (zero_cost_ratio / prev_zero_cost_ratio) > 1.5:
                print(f"Stopping exploration: Zero-cost ratio increased sharply at {del_percentile_frac:.3f}")
                break
            elif prev_zero_cost_ratio is not None and (
                (zero_cost_ratio - prev_zero_cost_ratio) > 0.15 or
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

    # Parse the best alignments
    parsed_alignments = []
    for line in best_alignments:
        if line:
            src_part, tgt_part, _ = line.split(":")
            src_indices = list(map(int, src_part.strip("[]").split(","))) if src_part.strip("[]") else []
            tgt_indices = list(map(int, tgt_part.strip("[]").split(","))) if tgt_part.strip("[]") else []
            parsed_alignments.append((src_indices, tgt_indices))

    print("\nBest Found:")
    print(f"del_percentile_frac: {best_del_percentile_frac:.3f} | Avg Cost: {best_avg_cost:.6f} | Zero-Cost Ratio: {best_zero_cost_ratio:.2%}")
    return parsed_alignments


def clean_sentence(sentence: str) -> str:
    """
    Clean a sentence by removing duplicate parts and reconnecting with the separator.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Cleaned sentence.
    """
    if not sentence:
        return ""
    parts = sentence.split(SEPARATOR)
    unique_parts = list(dict.fromkeys(part.strip() for part in parts if part.strip()))
    return f" {SEPARATOR} ".join(unique_parts) + f" {SEPARATOR}"


def sliding_windows(sentences: list, window_size: int) -> list:
    """
    Create sliding windows from a list of sentences.

    Args:
        sentences (list): List of sentences.
        window_size (int): Window size.

    Returns:
        list: List of sliding windows (each is a list of sentences).
    """
    windows = []
    for i in range(len(sentences) - window_size + 1):
        window = [clean_sentence(s) for s in sentences[i:i + window_size]]
        # Remove duplicate window contents
        unique_window = list(dict.fromkeys(window))
        windows.append(unique_window)
    return windows


def save_windows_to_file(paragraph_id: int, aligned_src: list, aligned_ref: list, aligned_mt: list,
                         src_windows: list, ref_windows: list, mt_windows: list,
                         qe_src_windows: list, qe_mt_windows: list, output_dir: str,
                         output_name: str) -> None:
    """
    Save window information and alignment data as JSON files.

    Args:
        paragraph_id (int): Paragraph ID.
        aligned_src (list): Adjusted source alignment.
        aligned_ref (list): Adjusted reference alignment.
        aligned_mt (list): Adjusted MT alignment.
        src_windows (list): Source sliding windows.
        ref_windows (list): Reference sliding windows.
        mt_windows (list): MT sliding windows.
        qe_src_windows (list): QE source sliding windows.
        qe_mt_windows (list): QE MT sliding windows.
        output_dir (str): Output directory path.
        output_name (str): Identifier for the output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    windows_data = {
        "paragraph_id": paragraph_id,
        "src_windows": src_windows,
        "ref_windows": ref_windows,
        "mt_windows": mt_windows,
    }
    windows_file = os.path.join(output_dir, f"windows_{paragraph_id}_{output_name}.json")
    with open(windows_file, "w", encoding="utf-8") as f:
        json.dump(windows_data, f, ensure_ascii=False, indent=2)

    qe_windows_data = {
        "paragraph_id": paragraph_id,
        "src_windows": qe_src_windows,
        "mt_windows": qe_mt_windows,
    }
    qe_windows_file = os.path.join(output_dir, f"qe_windows_{paragraph_id}_{output_name}.json")
    with open(qe_windows_file, "w", encoding="utf-8") as f:
        json.dump(qe_windows_data, f, ensure_ascii=False, indent=2)

    aligned_info = {
        "src": aligned_src,
        "ref": aligned_ref,
        "mt": aligned_mt,
    }
    aligned_file = os.path.join(output_dir, f"aligned_{paragraph_id}_{output_name}.json")
    with open(aligned_file, "w", encoding="utf-8") as f:
        json.dump(aligned_info, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Alignment Gap Processing Functions
# -----------------------------------------------------------------------------
def process_gaps(alignments: list) -> tuple:
    """
    Process alignment list blocks where the source is empty but target is non-empty,
    converting them into gap alignments (source converted to a negative gap key).

    Args:
        alignments (list): Original alignment list (each element is (src_indices, tgt_indices)).

    Returns:
        tuple: (new_alignments (list), gap_counts (dict))
    """
    new_alignments = []
    gap_counts = {}
    n = len(alignments)
    i = 0
    while i < n:
        src, tgt = alignments[i]
        if not src and tgt:
            block = []
            while i < n and not alignments[i][0] and alignments[i][1]:
                block.append(alignments[i])
                i += 1
            # Get the left neighbor's source index if available
            left_src = new_alignments[-1][0][-1] if new_alignments and new_alignments[-1][0] else None
            # Get the first non-empty source index on the right
            right_src = None
            j = i
            while j < n:
                if alignments[j][0]:
                    right_src = alignments[j][0][0]
                    break
                j += 1
            gap_key = left_src if left_src is not None else (right_src - 1 if right_src is not None else 0)
            for item in block:
                new_alignments.append(([-gap_key], item[1]))
            gap_counts[gap_key] = gap_counts.get(gap_key, 0) + len(block)
        else:
            new_alignments.append(alignments[i])
            i += 1
    return new_alignments, gap_counts


def complement_gaps(processed: list, gap_counts: dict, desired_gaps: dict) -> list:
    """
    Complement the gaps in the processed alignment list by inserting dummy alignments until
    the desired gap count is met.

    Args:
        processed (list): Processed alignment list.
        gap_counts (dict): Counts of each gap key in the processed list.
        desired_gaps (dict): Desired counts for each gap key from the other alignment list.

    Returns:
        list: Processed alignment list after gap completion.
    """
    all_keys = set(gap_counts.keys()) | set(desired_gaps.keys())
    for gap in all_keys:
        current = gap_counts.get(gap, 0)
        desired = desired_gaps.get(gap, 0)
        if current < desired:
            indices = [i for i, (src, _) in enumerate(processed) if src and src[0] == -gap]
            insert_idx = indices[0] if indices else next((i for i, (src, _) in enumerate(processed) if src and src[0] > gap), len(processed))
            for _ in range(desired - current):
                processed.insert(insert_idx, ([-gap], []))
            gap_counts[gap] = desired
    return processed


def custom_sort_key(item: tuple) -> tuple:
    """
    Custom sort key:
      - For non-gap alignments (positive), key = (source, 0).
      - For gap alignments (negative), key = (abs(source), 1).

    Args:
        item (tuple): Alignment tuple (src_indices, tgt_indices).

    Returns:
        tuple: Sorting key.
    """
    src, _ = item
    if src:
        val = src[0]
        return (val, 0) if val >= 0 else (abs(val), 1)
    return (float('inf'), 2)


def fill_empty_alignments(src_ref_alignments: list, src_mt_alignments: list) -> tuple:
    """
    Fill the empty alignments (gaps) in both source-reference and source-MT alignments so that
    the gap key counts match, then sort them.

    Args:
        src_ref_alignments (list): Alignment list for source-reference.
        src_mt_alignments (list): Alignment list for source-MT.

    Returns:
        tuple: (filled_src_ref_alignments, filled_src_mt_alignments)
    """
    proc_ref, gaps_ref = process_gaps(src_ref_alignments)
    proc_mt, gaps_mt = process_gaps(src_mt_alignments)
    proc_ref = complement_gaps(proc_ref, gaps_ref, gaps_mt)
    proc_mt = complement_gaps(proc_mt, gaps_mt, gaps_ref)
    proc_ref.sort(key=custom_sort_key)
    proc_mt.sort(key=custom_sort_key)
    return proc_ref, proc_mt


def find_common_alignments(src_ref_alignments: list, src_mt_alignments: list) -> list:
    """
    Find common alignments between source-reference and source-MT alignment lists and remove duplicates.

    Args:
        src_ref_alignments (list): Alignment list for source-reference.
        src_mt_alignments (list): Alignment list for source-MT.

    Returns:
        list: List of common alignments as (common_src_indices, common_ref_indices, common_mt_indices).
    """
    common_alignments = []
    src_ref_alignments, src_mt_alignments = fill_empty_alignments(src_ref_alignments, src_mt_alignments)

    for ref_align in src_ref_alignments:
        for mt_align in src_mt_alignments:
            common_src = sorted(list(set(ref_align[0]) & set(mt_align[0])))
            if common_src:
                common_ref = sorted(list(set(ref_align[1]))) if ref_align[1] else [-1]
                common_mt = sorted(list(set(mt_align[1]))) if mt_align[1] else [-1]
                common_alignments.append((common_src, common_ref, common_mt))

    # Remove duplicate triples
    unique = []
    seen = set()
    for triple in common_alignments:
        key = (tuple(triple[0]), tuple(triple[1]), tuple(triple[2]))
        if key not in seen:
            seen.add(key)
            unique.append(triple)
    print("Unique common alignments:")
    print(unique)
    return unique


def args_to_dict(args: argparse.Namespace, prefix: str, strip_prefix: bool = False) -> dict:
    """
    Convert an argparse Namespace to a dictionary, optionally filtering by a prefix and stripping it.

    Args:
        args (argparse.Namespace): Input arguments.
        prefix (str): Prefix to filter keys.
        strip_prefix (bool): Whether to remove the prefix from keys (default is False).

    Returns:
        dict: Filtered dictionary.
    """
    d = vars(args)
    prefix_key = prefix + '_'
    filtered = {k: v for k, v in d.items() if k.startswith(prefix_key)}
    if strip_prefix:
        return {k[len(prefix_key):]: v for k, v in filtered.items()}
    return filtered


# -----------------------------------------------------------------------------
# Metrics Computation
# -----------------------------------------------------------------------------
def compute_metrics(paragraph_src: str, paragraph_ref: str, paragraph_mt: str,
                    src_windows: list, ref_windows: list, mt_windows: list,
                    qe_src_windows: list, qe_mt_windows: list,
                    paragraph_id: int, mt_col: str) -> dict:
    """
    Compute COMET and COMET-QE scores, then save the scores and related window information as a JSON file.

    Args:
        paragraph_src (str): Source paragraph text.
        paragraph_ref (str): Reference paragraph text.
        paragraph_mt (str): MT paragraph text.
        bleu_adjusted_ref: (Placeholder) BLEU adjusted parameter.
        bleu_adjusted_mt: (Placeholder) BLEU adjusted parameter.
        src_windows (list): Source sliding windows.
        ref_windows (list): Reference sliding windows.
        mt_windows (list): MT sliding windows.
        qe_src_windows (list): QE source sliding windows.
        qe_mt_windows (list): QE MT sliding windows.
        paragraph_id (int): Paragraph ID.
        mt_col (str): MT column name.

    Returns:
        dict: Dictionary containing various computed scores.
    """
    comet_zero_score_windows = []
    comet_qe_zero_score_windows = []

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as src_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True) as ref_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True) as mt_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True) as qe_src_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True) as qe_mt_file:

        # Write each window on a separate line
        for idx, (src_win, ref_win, mt_win) in enumerate(zip(src_windows, ref_windows, mt_windows)):
            src_line = " ".join(src_win)
            ref_line = " ".join(ref_win)
            mt_line = " ".join(mt_win)
            if src_line and mt_line:
                src_file.write(src_line + "\n")
                ref_file.write(ref_line + "\n")
                mt_file.write(mt_line + "\n")
            else:
                comet_zero_score_windows.append(idx)

        src_file.flush()
        ref_file.flush()
        mt_file.flush()

        comet_command = [
            "comet-score",
            "-s", src_file.name,
            "-t", mt_file.name,
            "-r", ref_file.name,
            "--model", COMET_MODEL,
            "--enable-context",
            "--gpus", GPU_ID,
            "--quiet",
        ]
        result = subprocess.run(comet_command, stdout=subprocess.PIPE, text=True)
        print(result.stdout)
        comet_scores = [float(s) for s in re.findall(r"score:\s(-?[0-9.]+)", result.stdout.strip())][:-1]

        for idx, (src_win, mt_win) in enumerate(zip(qe_src_windows, qe_mt_windows)):
            src_line = " ".join(src_win)
            mt_line = " ".join(mt_win)
            if src_line and mt_line:
                qe_src_file.write(src_line + "\n")
                qe_mt_file.write(mt_line + "\n")
            else:
                comet_qe_zero_score_windows.append(idx)

        qe_src_file.flush()
        qe_mt_file.flush()

        qe_command = [
            "comet-score",
            "-s", qe_src_file.name,
            "-t", qe_mt_file.name,
            "--model", COMET_QE_MODEL,
            "--enable-context",
            "--gpus", GPU_ID,
            "--quiet",
        ]
        qe_result = subprocess.run(qe_command, stdout=subprocess.PIPE, text=True)
        print(qe_result.stdout)
        comet_qe_scores = [float(s) for s in re.findall(r"score:\s(-?[0-9.]+)", qe_result.stdout.strip())][:-1]

    # Insert zero scores for windows that had missing scores
    for idx in comet_zero_score_windows:
        comet_scores.insert(idx, 0.0)
    for idx in comet_qe_zero_score_windows:
        comet_qe_scores.insert(idx, 0.0)

    # Placeholder values for sentence-level metrics
    sentences_length = len(paragraph_mt.splitlines())
    sentences_zero_ratio = 0.0

    scores_data = {
        'paragraph_id': paragraph_id,
        'comet_scores': comet_scores,
        'comet_qe_scores': comet_qe_scores,
        'sentences_length': sentences_length,
        'windows_length': len(comet_scores),
        'windows_qe_length': len(comet_qe_scores),
        'sentences_zero_ratio': sentences_zero_ratio,
        'windows_zero_ratio': len(comet_zero_score_windows) / len(comet_scores) if comet_scores else 0,
        'windows_qe_zero_ratio': len(comet_qe_zero_score_windows) / len(comet_qe_scores) if comet_qe_scores else 0,
        'avg_comet': sum(comet_scores) / len(comet_scores) if comet_scores else 0,
        'avg_comet_qe': sum(comet_qe_scores) / len(comet_qe_scores) if comet_qe_scores else 0
    }

    scores_file = os.path.join(SAVE_FOLDER, 'scores', f'scores_{paragraph_id}_{mt_col}.json')
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    with open(scores_file, 'w', encoding='utf-8') as f:
        json.dump(scores_data, f, ensure_ascii=False, indent=2)

    return scores_data


def compute_metrics_reference_free(src_windows: list, mt_windows: list,
                                   qe_src_windows: list, qe_mt_windows: list,
                                   paragraph_id: int, mt_col: str) -> dict:
    """
    Compute reference-free evaluation metrics (only QE scores) when no reference is provided.

    Args:
        src_windows (list): (Unused) Source sliding windows.
        mt_windows (list): (Unused) MT sliding windows.
        qe_src_windows (list): QE source sliding windows.
        qe_mt_windows (list): QE MT sliding windows.
        paragraph_id (int): Paragraph ID.
        mt_col (str): MT column name.

    Returns:
        dict: Dictionary containing computed QE scores.
    """
    comet_qe_zero_score_windows = []

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as qe_src_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True) as qe_mt_file:

        for idx, (src_win, mt_win) in enumerate(zip(qe_src_windows, qe_mt_windows)):
            src_line = " ".join(src_win)
            mt_line = " ".join(mt_win)
            if src_line and mt_line:
                qe_src_file.write(src_line + "\n")
                qe_mt_file.write(mt_line + "\n")
            else:
                comet_qe_zero_score_windows.append(idx)
        qe_src_file.flush()
        qe_mt_file.flush()

        qe_command = [
            "comet-score",
            "-s", qe_src_file.name,
            "-t", qe_mt_file.name,
            "--model", COMET_QE_MODEL,
            "--enable-context",
            "--gpus", GPU_ID,
            "--quiet",
        ]
        qe_result = subprocess.run(qe_command, stdout=subprocess.PIPE, text=True)
        print(qe_result.stdout)
        comet_qe_scores = [float(s) for s in re.findall(r"score:\s(-?[0-9.]+)", qe_result.stdout.strip())][:-1]

    for idx in comet_qe_zero_score_windows:
        comet_qe_scores.insert(idx, 0.0)

    scores_data = {
        'paragraph_id': paragraph_id,
        'comet_scores': 0.0,  # Not computed in reference-free mode.
        'comet_qe_scores': comet_qe_scores,
        'windows_length': len(comet_qe_scores),
        'windows_qe_length': len(comet_qe_scores),
        'avg_comet': 0.0,
        'avg_comet_qe': sum(comet_qe_scores) / len(comet_qe_scores) if comet_qe_scores else 0,
    }
    return scores_data


# -----------------------------------------------------------------------------
# Paragraph-Level Processing
# -----------------------------------------------------------------------------
def paragraph_level_score(row: pd.Series, paragraph_id: int, src_col: str = None,
                          ref_col: str = None, mt_col: str = None) -> None:
    """
    Process alignment and scoring for a single paragraph. Steps include:
      1. Sentence segmentation and preprocessing.
      2. Generating overlaps and embeddings.
      3. Running vector alignment exploration.
      4. Computing COMET and COMET-QE scores and saving window information.

    Args:
        row (pd.Series): A single data row.
        paragraph_id (int): Paragraph identifier.
        src_col (str): Source column name (default is "zh").
        ref_col (str): Reference column name (default is set based on language).
        mt_col (str): MT column name (default is set based on TARGET).
    """
    global mt_nlp, src_nlp

    # Set default columns if not provided
    if ref_col is None:
        ref_col = LANG
    if mt_col is None:
        mt_col = TARGET

    # Sentence segmentation and preprocessing
    src_sentences = segment_sentences_by_punctuation(row[src_col], src_col)
    ref_sentences = segment_sentences_by_punctuation(row[ref_col], ref_col)
    mt_sentences = segment_sentences_by_punctuation(row[mt_col], ref_col)

    src_txt = preprocess_sentences(src_sentences)
    ref_txt = preprocess_sentences(ref_sentences)
    mt_txt = preprocess_sentences(mt_sentences)

    # Generate overlap and embedding data
    src_overlap, src_embed = generate_overlap_and_embedding(src_txt)
    ref_overlap, ref_embed = generate_overlap_and_embedding(ref_txt)
    mt_overlap, mt_embed = generate_overlap_and_embedding(mt_txt)

    # Run vector alignment exploration
    src_ref_alignments = run_vecalign_explore(src_txt, ref_txt, src_overlap, ref_overlap, src_embed, ref_embed)
    src_mt_alignments = run_vecalign_explore(src_txt, mt_txt, src_overlap, mt_overlap, src_embed, mt_embed)

    # For reference-free evaluation: get non-adjusted alignments
    non_adjusted_src = []
    non_adjusted_mt = []
    for src_indices, mt_indices in src_mt_alignments:
        mt_indices = [x for x in mt_indices if x != -1]
        aligned_src = " ".join([src_sentences[i] for i in src_indices]) if src_indices else ""
        aligned_mt = " ".join([mt_sentences[i] for i in mt_indices]) if mt_indices else ""
        non_adjusted_src.append(aligned_src)
        non_adjusted_mt.append(aligned_mt)

    # Find common alignments between src-ref and src-mt
    common_alignments = find_common_alignments(src_ref_alignments, src_mt_alignments)

    adjusted_src, adjusted_ref, adjusted_mt = [], [], []
    for src_indices, ref_indices, mt_indices in common_alignments:
        ref_indices = [x for x in ref_indices if x != -1]
        mt_indices = [x for x in mt_indices if x != -1]
        aligned_src = "" if (src_indices and src_indices[0] < 0) else " ".join([src_sentences[i] for i in src_indices])
        aligned_ref = " ".join([ref_sentences[i] for i in ref_indices]) if ref_indices else ""
        aligned_mt = " ".join([mt_sentences[i] for i in mt_indices]) if mt_indices else ""
        adjusted_src.append(aligned_src)
        adjusted_ref.append(aligned_ref)
        adjusted_mt.append(aligned_mt)

    # Create sliding windows
    src_windows = sliding_windows(adjusted_src, WINDOW_SIZE)
    ref_windows = sliding_windows(adjusted_ref, WINDOW_SIZE)
    mt_windows = sliding_windows(adjusted_mt, WINDOW_SIZE)
    qe_src_windows = sliding_windows(non_adjusted_src, WINDOW_SIZE)
    qe_mt_windows = sliding_windows(non_adjusted_mt, WINDOW_SIZE)

    # Compute metrics and save window information
    compute_metrics(
        row[src_col], row[ref_col], row[mt_col],
        src_windows, ref_windows, mt_windows,
        qe_src_windows, qe_mt_windows,
        paragraph_id, mt_col
    )

    output_dir = os.path.join(SAVE_FOLDER, "windows")
    save_windows_to_file(paragraph_id, adjusted_src, adjusted_ref, adjusted_mt,
                         src_windows, ref_windows, mt_windows,
                         qe_src_windows, qe_mt_windows, output_dir, output_name=mt_col)


def parallel_paragraph_level_score(args: tuple) -> None:
    """
    Process a single paragraph in parallel. If an exception occurs, it prints an error message.

    Args:
        args (tuple): (row (pd.Series), paragraph_id (int))
    """
    row, paragraph_id = args
    try:
        paragraph_level_score(row, paragraph_id, mt_col=TARGET, src_col= SRC_LANG)
    except Exception as e:
        print(f"Error processing paragraph {paragraph_id}: {e}")
        print(f"{TARGET} result cannot be aligned in paragraph {paragraph_id}\n")


# -----------------------------------------------------------------------------
# New Function: Flexible Evaluation (Reference-Free or Full Evaluation)
# -----------------------------------------------------------------------------
def evaluate_score(src: str, tgt: str, ref: Optional[str] = None) -> dict:
    """
    Evaluate quality scores for given source and target texts.
    If a reference is provided, full evaluation is performed (including src-ref alignment);
    otherwise, reference-free evaluation is conducted using only src and tgt.

    Args:
        src (str): Source text.
        tgt (str): Target (MT) text.
        ref (Optional[str]): Reference text (if provided).

    Returns:
        dict: Dictionary of evaluation scores.
    """
    # Full evaluation (with reference)
    if ref is not None:
        src_sentences = segment_sentences_by_punctuation(src, SRC_LANG)
        ref_sentences = segment_sentences_by_punctuation(ref, LANG)
        tgt_sentences = segment_sentences_by_punctuation(tgt, LANG)

        src_txt = preprocess_sentences(src_sentences)
        ref_txt = preprocess_sentences(ref_sentences)
        tgt_txt = preprocess_sentences(tgt_sentences)

        src_overlap, src_embed = generate_overlap_and_embedding(src_txt)
        ref_overlap, ref_embed = generate_overlap_and_embedding(ref_txt)
        tgt_overlap, tgt_embed = generate_overlap_and_embedding(tgt_txt)

        src_ref_alignments = run_vecalign_explore(src_txt, ref_txt, src_overlap, ref_overlap, src_embed, ref_embed)
        src_mt_alignments = run_vecalign_explore(src_txt, tgt_txt, src_overlap, tgt_overlap, src_embed, tgt_embed)

        non_adjusted_src = []
        non_adjusted_mt = []
        for s_indices, t_indices in src_mt_alignments:
            filtered_t_indices = [x for x in t_indices if x != -1]
            aligned_src = " ".join([src_sentences[i] for i in s_indices]) if s_indices else ""
            aligned_mt = " ".join([tgt_sentences[i] for i in filtered_t_indices]) if filtered_t_indices else ""
            non_adjusted_src.append(aligned_src)
            non_adjusted_mt.append(aligned_mt)

        common_alignments = find_common_alignments(src_ref_alignments, src_mt_alignments)
        adjusted_src, adjusted_ref, adjusted_mt = [], [], []
        for s_indices, r_indices, t_indices in common_alignments:
            r_indices = [x for x in r_indices if x != -1]
            t_indices = [x for x in t_indices if x != -1]
            aligned_src = "" if (s_indices and s_indices[0] < 0) else " ".join([src_sentences[i] for i in s_indices])
            aligned_ref = " ".join([ref_sentences[i] for i in r_indices]) if r_indices else ""
            aligned_mt = " ".join([tgt_sentences[i] for i in t_indices]) if t_indices else ""
            adjusted_src.append(aligned_src)
            adjusted_ref.append(aligned_ref)
            adjusted_mt.append(aligned_mt)

        src_windows = sliding_windows(adjusted_src, WINDOW_SIZE)
        ref_windows = sliding_windows(adjusted_ref, WINDOW_SIZE)
        tgt_windows = sliding_windows(adjusted_mt, WINDOW_SIZE)
        qe_src_windows = sliding_windows(non_adjusted_src, WINDOW_SIZE)
        qe_mt_windows = sliding_windows(non_adjusted_mt, WINDOW_SIZE)

        # Use paragraph_id=0 for single evaluation
        scores_data = compute_metrics(src, ref, tgt,
                                      src_windows, ref_windows, tgt_windows,
                                      qe_src_windows, qe_mt_windows,
                                      paragraph_id=0, mt_col=TARGET)
        return scores_data

    # Reference-free evaluation
    else:
        src_sentences = segment_sentences_by_punctuation(src, SRC_LANG)
        tgt_sentences = segment_sentences_by_punctuation(tgt, LANG)

        src_txt = preprocess_sentences(src_sentences)
        tgt_txt = preprocess_sentences(tgt_sentences)

        src_overlap, src_embed = generate_overlap_and_embedding(src_txt)
        tgt_overlap, tgt_embed = generate_overlap_and_embedding(tgt_txt)

        src_mt_alignments = run_vecalign_explore(src_txt, tgt_txt, src_overlap, tgt_overlap, src_embed, tgt_embed)

        non_adjusted_src = []
        non_adjusted_mt = []
        for s_indices, t_indices in src_mt_alignments:
            filtered_t_indices = [x for x in t_indices if x != -1]
            aligned_src = " ".join([src_sentences[i] for i in s_indices]) if s_indices else ""
            aligned_mt = " ".join([tgt_sentences[i] for i in filtered_t_indices]) if filtered_t_indices else ""
            non_adjusted_src.append(aligned_src)
            non_adjusted_mt.append(aligned_mt)

        # In reference-free mode, only compute QE evaluation.
        qe_src_windows = sliding_windows(non_adjusted_src, WINDOW_SIZE)
        qe_mt_windows = sliding_windows(non_adjusted_mt, WINDOW_SIZE)

        scores_data = compute_metrics_reference_free(src_windows=[], mt_windows=[],
                                                     qe_src_windows=qe_src_windows, qe_mt_windows=qe_mt_windows,
                                                     paragraph_id=0, mt_col=TARGET)
        return scores_data


def aggregate_scores_and_merge(evaluated_file_path: str, save_folder: str, target: str) -> dict:
    """
    Read scores for each paragraph, aggregate the results, and save them as a CSV.

    Args:
        evaluated_file_path (str): Path to the original CSV file.
        save_folder (str): Folder where scores are saved.
        target (str): MT target name.

    Returns:
        dict: Overall average scores for each metric.
    """
    df = pd.read_csv(evaluated_file_path)
    df['comet'] = 0.0
    df['comet_qe'] = 0.0
    df['sentences_zero_ratio'] = 0.0
    df['windows_zero_ratio'] = 0.0
    df['windows_qe_zero_ratio'] = 0.0

    scores_dir = os.path.join(save_folder, 'scores')
    total_scores = {
        'comet': [],
        'comet_qe': [],
        'sentences_zero_ratio': [],
        'windows_zero_ratio': [],
        'windows_qe_zero_ratio': []
    }

    for idx in df.index:
        score_file = os.path.join(scores_dir, f'scores_{idx}_{target}.json')
        if os.path.exists(score_file):
            with open(score_file, 'r', encoding='utf-8') as f:
                scores = json.load(f)
                df.at[idx, 'comet'] = scores.get('avg_comet', 0)
                df.at[idx, 'comet_qe'] = scores.get('avg_comet_qe', 0)
                df.at[idx, 'sentences_zero_ratio'] = scores.get('sentences_zero_ratio', 0)
                df.at[idx, 'windows_zero_ratio'] = scores.get('windows_zero_ratio', 0)
                df.at[idx, 'windows_qe_zero_ratio'] = scores.get('windows_qe_zero_ratio', 0)

                total_scores['comet'].append(scores.get('avg_comet', 0))
                total_scores['comet_qe'].append(scores.get('avg_comet_qe', 0))
                total_scores['sentences_zero_ratio'].append(scores.get('sentences_zero_ratio', 0))
                total_scores['windows_zero_ratio'].append(scores.get('windows_zero_ratio', 0))
                total_scores['windows_qe_zero_ratio'].append(scores.get('windows_qe_zero_ratio', 0))

    overall_scores = {metric: (sum(vals) / len(vals) if vals else 0) for metric, vals in total_scores.items()}

    output_path = os.path.join(save_folder, f'evaluated_results_{target}.csv')
    df.to_csv(output_path, index=False)
    return overall_scores


# -----------------------------------------------------------------------------
# Global Parameters
# -----------------------------------------------------------------------------
set_seed(42)

# Set up argparse with defaults for file, target_column, and save folder.
parser = argparse.ArgumentParser(description="Set TARGET_FILE, TARGET_COLUMN, and TASK_LANGUAGE")
parser.add_argument("--file", type=str, default="", help="(Optional) Set the MT target file")
parser.add_argument("--target_column", type=str, default="", help="(Optional) Set the MT target column")
parser.add_argument("--save", type=str, default="./", help="(Optional) Set the save folder")
parser.add_argument("--src_language", type=str, required=True, help="Set the task language (English, Russian, German)")
parser.add_argument("--task_language", type=str, required=True, help="Set the task language (English, Russian, German)")

args = parser.parse_args()

TARGET = args.target_column
TASK_LANGUAGE = args.task_language
SRC_LANGUAGE = args.src_language
print(f"TARGET: {TARGET}")
print(f"TASK_LANGUAGE: {TASK_LANGUAGE}")

if TASK_LANGUAGE == "English":
    LANG = 'en'
elif TASK_LANGUAGE == "Russian":
    LANG = 'ru'
elif TASK_LANGUAGE == "German":
    LANG = 'de'
elif TASK_LANGUAGE == "Japanese":
    LANG = 'ja'
elif TASK_LANGUAGE == "Spanish":
    LANG = 'es'
elif TASK_LANGUAGE == "Chinese":
    LANG = 'zh'
else:
    raise ValueError("Unsupported TASK_LANGUAGE.")

if SRC_LANGUAGE == "English":
    SRC_LANG = 'en'
elif SRC_LANGUAGE == "Russian":
    SRC_LANG = 'ru'
elif SRC_LANGUAGE == "German":
    SRC_LANG = 'de'
elif SRC_LANGUAGE == "Japanese":
    SRC_LANG = 'ja'
elif SRC_LANGUAGE == "Spanish":
    SRC_LANG = 'es'
elif SRC_LANGUAGE == "Chinese":
    SRC_LANG = 'zh'
else:
    raise ValueError("Unsupported TASK_LANGUAGE.")

# File and folder path settings
evaluated_file_path = args.file  # May be empty if not provided
WINDOW_SIZE = 3
SEPARATOR = "</s>"
SAVE_FOLDER = args.save
GPU_ID = "1"
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_QE_MODEL = "Unbabel/wmt22-cometkiwi-da"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Folder '{SAVE_FOLDER}' created")
else:
    print(f"Folder '{SAVE_FOLDER}' already exists")

# Load Spacy models based on task language
if TASK_LANGUAGE == "English":
    mt_nlp = spacy.load("en_core_web_sm")
elif TASK_LANGUAGE == "Russian":
    mt_nlp = spacy.load("ru_core_news_sm")
elif TASK_LANGUAGE == "German":
    mt_nlp = spacy.load("de_core_news_sm")
elif TASK_LANGUAGE == "Japanese":
    mt_nlp = spacy.load("ja_core_news_sm")
elif TASK_LANGUAGE == "Spanish":
    mt_nlp = spacy.load("es_core_news_sm")
elif TASK_LANGUAGE == "Chinese":
    mt_nlp = spacy.load("zh_core_web_sm")
    
if SRC_LANGUAGE == "English":
    src_nlp = spacy.load("en_core_web_sm")
elif SRC_LANGUAGE == "Russian":
    src_nlp = spacy.load("ru_core_news_sm")
elif SRC_LANGUAGE == "German":
    src_nlp = spacy.load("de_core_news_sm")
elif SRC_LANGUAGE == "Japanese":
    src_nlp = spacy.load("ja_core_news_sm")
elif SRC_LANGUAGE == "Spanish":
    src_nlp = spacy.load("es_core_news_sm")
elif SRC_LANGUAGE == "Chinese":
    src_nlp = spacy.load("zh_core_web_sm")

# -----------------------------------------------------------------------------
# Main Process: Parallel processing of paragraphs and score aggregation
# Command: export LASER="/path/to/laser/"
# Command for evaluate csv: 
# python long_context_eval.py --file eval_en_ja.csv --target_column mpc --save eval_en_ja --src_language English --task_language Japanese
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data = pd.read_csv(evaluated_file_path)
    pool_args = [(row, idx) for idx, row in data.iterrows()]
    with Pool(2) as pool:
        pool.map(parallel_paragraph_level_score, pool_args)
    overall_scores = aggregate_scores_and_merge(evaluated_file_path, SAVE_FOLDER, TARGET)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_result = f"{TARGET}: {TASK_LANGUAGE} Overall scores: {overall_scores}, time: {timestamp}\n"
    print(output_result)