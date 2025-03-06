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

### 2. Install VecAlign

Plan2Align relies on VecAlign for alignment tasks. Please follow the installation instructions provided in the official repository:  
[VecAlign GitHub Repository](https://github.com/thompsonb/vecalign)

### 3. Configure Environment Variables for LASER

LASER must be properly configured by setting up the required environment variables. Use the following steps:

```bash
nano ~/.bashrc
export LASER="{PATH_TO_LASER}"
source ~/.bashrc
```

Make sure to replace `{PATH_TO_LASER}` with the actual path where LASER is installed.

### 4. Prepare API Key

Plan2Align requires an API key for OpenAI services. Ensure that you have the necessary credentials set up:

```python
openai = OpenAI(
    api_key='your-api-key',
    base_url='your-base_url',
)
```

Replace `'your-api-key'` and `'your-base_url'` with your actual API key and endpoint.

### 5. Configure Reward Model

Plan2Align utilizes a reward model for alignment tasks. Ensure that you modify the following paths in your reward model setup before use:

```python
self.RM = AutoModelForCausalLMWithValueHead.from_pretrained(
    '../<path-to-rm>',
    torch_dtype=torch.bfloat16
).to(self.device)

value_head_weights = load_file("../<path-to-value_head>")
```

Replace `<path-to-rm>` and `<path-to-value_head>` with the correct file paths in your system.

Before running the program, you can use `set_translation_model("rm")` to make Plan2Align perform alignment based on the reward model.

### 6. Running Plan2Align

For ease of testing Plan2Align, we provide a small preference model for alignment. You can download its weights from the following link:  
[Download Weights](https://drive.google.com/file/d/1us3pBmnJseI0-lozh999dDraql9m03im/view?usp=sharing).  
Place it directly in the project directory, and use `set_translation_model("pm")` in `plan2align.py` to utilize it.

Regarding datasets, we used the dataset from [Hugging Face](https://huggingface.co/datasets/huckiyang/zh-tw-en-us-nv-tech-blog-v1) for validation. We selected longer, semantically structured samples from it, created a `valid_en.csv`, and performed Chinese-to-English translation tasks.

To validate that Plan2Align is correctly installed and configured, execute the following command:

```bash
python plan2align.py \
    --task_language English \
    --dataset "valid_en.csv" \
    --start_index 0 \
    --end_index 5 \
    --cuda_num 0 \
    --threshold 2 \
    --max_iterations 6 \
    --good_ref_contexts_num 5 \
    --good_context_buffer_size 3 \
    --memory_folder "memory" \
    --output_suffix "t_2_d_6_chunk_0_5"
```

If the script runs successfully, the installation is complete.

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