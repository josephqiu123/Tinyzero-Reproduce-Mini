# Tinyzero-Reproduce-Mini

This repository demonstrates Reinforcement Learning from Human Feedback (RLHF) for LLMs using Group Relative Policy Optimization (GRPO). It specifically implements a concept similar to **TinyZero**, which is an effort to reproduce aspects of powerful self-improving models like DeepSeek R1 Zero ([https://arxiv.org/abs/2501.12948]).

Leveraging the efficiency of the `unsloth` library and `trl`, this project aims to make advanced RLHF techniques significantly more accessible. While many RLHF pipelines demand massive GPU clusters, **this implementation is designed to run effectively on a single GPU with approximately 40GB of VRAM** (e.g., an NVIDIA A100 40GB, H100 40GB).

The goal is to train a model to solve a specific mathematical task (inspired by the "Countdown" numbers game): generating an equation using a given set of numbers to reach a target value, while adhering to a predefined `<think>`/`<answer>` reasoning format. 

## Background: TinyZero and Accessible RLHF

Traditional RLHF methods, especially those aiming for self-improvement loops seen in state-of-the-art models, often require substantial computational resources. Projects like DeepSeek R1 Zero showcase remarkable capabilities achieved through complex RL techniques but necessitate large-scale infrastructure.

**TinyZero** ([GitHub Repo: Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero)) emerged as an initiative to replicate some of these advanced self-improvement ideas on a much smaller, more manageable scale. 

This repository builds upon the spirit of TinyZero and utilizes an `unsloth`-based framework, directly adapted from the **Unsloth Qwen2.5 GRPO Tutorial** ([Colab Notebook Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9)). It provides a concrete, runnable example of GRPO-based RLHF. The underlying RL algorithm used is Group Relative Policy Optimization (GRPO); you can learn more about its theory in the [Hugging Face LLM Course - GRPO Section](https://huggingface.co/learn/llm-course/en/chapter12/3a). By focusing on efficiency (4-bit quantization, PEFT/LoRA, optimized kernels via Unsloth, VLLM integration), it demonstrates the feasibility of performing this type of training on a smaller scale.

## Features

*   **Accessible RLHF:** Code run on a single GPU, significantly lowering the barrier for RLHF experimentation compared to large-scale setups.
*   **Reinforcement Learning (GRPO):** Uses `trl.GRPOTrainer` to fine-tune the LLM based on custom reward signals. 
*   **Efficiency (`unsloth` + `vllm`):** Leverages `unsloth` for optimized training/LoRA and `vllm` (integrated via `unsloth` and `trl`) for faster generation during RL sampling. 
*   **Custom Rewards:** Implements two reward functions focused on format adherence and mathematical correctness for the Countdown task.


## Requirements

*   Python 3.10+
*   PyTorch (compatible with your CUDA version)
*   **A CUDA-enabled GPU with ~40GB VRAM minimum recommended** (e.g., A100 40GB, H100 PCIe 40GB, RTX 4090/3090 - memory usage may vary slightly based on config). Required for Unsloth, VLLM, and 4-bit quantization.
*   **Key Python Libraries:**
    *   `unsloth`: For optimized training, LoRA, and 4-bit support.
    *   `torch`: The core deep learning framework.
    *   `trl`: For the GRPO trainer and RLHF utilities.
    *   `datasets`: For data loading and handling.
    *   `vllm`: For accelerated inference during RL generation (used via `use_vllm=True`).
*   **Supporting Libraries (Often installed as dependencies or needed explicitly):**
    *   `transformers`: Required by `unsloth`, `trl`.
    *   `bitsandbytes`: Required for 4-bit quantization (`load_in_4bit=True`).
    *   `accelerate` (Optional, but recommended): Often used by `trl` and `unsloth` for efficient execution, especially multi-GPU or mixed precision.
    *   `deepspeed` (Optional): An alternative to `accelerate` for large-scale training optimizations.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[Your GitHub Username]/[Your Repository Name].git
    cd [Your Repository Name]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    *   **Install PyTorch:** Follow instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) for your specific CUDA version. This is critical for compatibility with other libraries.

    *   **Install Unsloth:** Follow the specific instructions on the [Unsloth GitHub](https://github.com/unslothai/unsloth). This often depends on your environment (PyTorch/CUDA version) and will pull in many dependencies like `transformers` and potentially `bitsandbytes`, `accelerate`.
        ```bash
        pip install "unsloth"
        ```

    *   **Install VLLM:** VLLM installation can be sensitive to CUDA/PyTorch versions. Check the [VLLM documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html) for the recommended method for your setup. It might be:
        ```bash
        # Example command - verify compatibility first!
        pip install vllm
        ```
        *Note: If `pip install vllm` fails or causes issues, you might need to build from source or use a specific wheel matching your environment.*


    *   **Install Quantization Library (if not installed by Unsloth):**
        ```bash
        pip install bitsandbytes
        ```


## Dataset

The training script expects a dataset (e.g., in Parquet format) with at least two columns:

*   `nums`: A list of integers available to form the equation.
*   `target`: The integer target value the equation should equal.

The specific dataset used in the example script (`data/Countdown-Tasks-3to4.parquet`) can be found on the Hugging Face Hub:
*   **Dataset Source:** [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)

You will need to:

1.  **Obtain the dataset:** You can load it directly using the `load_dataset` function from the Hugging Face Hub identifier above, or download the Parquet file and place it locally (e.g., in a `data/` directory).
2.  **Update the path (if local):** If you download the file, modify the `data_files` argument in the `get_countdown_questions` function within the script to point to your local path.

```python
# Inside get_countdown_questions function in the script:
# Example using Hub ID (recommended):
data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
# Example using local file:
# data = load_dataset("parquet", data_files="data/Countdown-Tasks-3to4.parquet", split="train")
