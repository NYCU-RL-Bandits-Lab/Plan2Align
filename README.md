# Plan2Align

This is the official implementation for the paper **"Plan2Align: Predictive Planning Based Test-Time Preference Alignment in Paragraph-Level Machine Translation"**.

## Environment Setup Guide for Plan2Align

This document provides a step-by-step guide for setting up the environment required to run Plan2Align efficiently. Please follow the instructions below to ensure a smooth installation process.

### 1. Create a Conda Virtual Environment (Recommended)

It is highly recommended to use a Conda virtual environment to manage dependencies and avoid conflicts. Execute the following commands:

```bash
conda create --name plan2align python=3.9
conda activate plan2align
```

### 2. Install Dependencies: VecAlign & SpaCy

Plan2Align relies on **VecAlign** for alignment tasks. Follow the installation instructions in the [VecAlign GitHub Repository](https://github.com/thompsonb/vecalign).

Text segmentation is handled by **SpaCy**. Please refer to the [spaCy Installation Guide](https://spacy.io/usage) for installing the relevant language models. You can update the supported translation languages via the `lang_map` dictionary in `plan2align.py` (Line 81).

### 3. Configure Environment Variables for LASER

LASER must be properly configured by setting up the required environment variables. Use the following steps:

```bash
nano ~/.bashrc
export LASER="{PATH_TO_LASER}"
source ~/.bashrc
```

Make sure to replace `{PATH_TO_LASER}` with the actual path where LASER is installed.

### 4. Set Up API Keys for OpenAI Services

Plan2Align uses an API key to access OpenAI services. Configure your credentials as follows:

```python
openai = OpenAI(
    api_key='your-api-key',
    base_url='your-base_url'
)
```

Replace `'your-api-key'` and `'your-base_url'` with your actual credentials. Alternatively, you may opt for a locally deployed language model if available.

### 5. Configure Reward Model

Plan2Align utilizes a reward model for alignment tasks. Ensure that you modify the following paths in your reward model setup before use:

For HH_RLHF task: Set args.rm_path to our huggingface repository.

For Translation task: Set args.rm to our huggingface repository.
```python
# for HH_RLHF
parser.add_argument("--rm_path", type=str, default='', help="reward model path")

# for Translation
parser.add_argument("--rm_path", type=str, default='', help="reward model path")
```

### 6. Running Plan2Align

For testing, an alternative metric—**MetricX-QE**—is available to replace the reward model.

Plan2Align accepts a CSV file as input with each column designated by a language code (e.g., `zh`, `en`). Specify source and target languages via command-line arguments. For example, to perform a Chinese-to-English translation task using `valid_en_ja.csv`, execute:

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
```

Results from each iteration and final outputs will be saved in a folder named after the input file (e.g., `valid_en_ja`).

### 7. Evaluation Process

Translation results from each iteration are stored in separate folders. To merge results from a specific iteration into a single CSV file, use the following command. For example, to merge iteration 5 results into `valid_en_ja.csv` with the output column named `plan2align`:

```bash
python memory2csv.py \
    --num 5 \
    --input_csv valid_en_ja.csv  \
    --output_csv eval_en_ja.csv \
    --column_name plan2align
```

Then, evaluate the `plan2align` column with:

```bash
python long_context_eval.py \
    --file valid_en_ja.csv \
    --target_column plan2align \
    --save eval_en_ja \
    --src_language English \
    --task_language Japanese
```

The evaluation scores will be saved in the `eval_en_ja` folder as `evaluated_results_plan2align.csv`.

---

## Citation

If you would like to cite this work, please use the following BibTeX entry:

```bibtex
@article{wang2025plan2align,
  title={Plan2Align: Predictive Planning Based Test-Time Preference Alignment in Paragraph-Level Machine Translation},
  author={Wang, Kuang-Da and Chen, Teng-Ruei and Hung, Yu Heng and Ding, Shuoyang and Wu, Yueh-Hua and Wang, Yu-Chiang Frank and Yang, Chao-Han Huck and Peng, Wen-Chih and Hsieh, Ping-Chun},
  journal={arXiv preprint arXiv:2502.20795},
  year={2025}
}
```