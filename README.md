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
    git clone https://github.com/josephqiu123/Tinyzero-Reproduce-Mini
    cd Tinyzero-Reproduce-Mini
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
# Inside get_questions function in the script:
# Example using Hub ID (recommended):
data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
# Example using local file:
# data = load_dataset("parquet", data_files="data/Countdown-Tasks-3to4.parquet", split="train")

## Dataset

Reinforcement Learning algorithms like GRPO rely on reward functions to guide the model towards desired behaviors. These functions evaluate the model's generated completions based on specific criteria and assign a numerical score (reward). Higher rewards encourage the model to produce similar outputs in the future.

This project utilizes two distinct reward functions applied sequentially or combined by the `GRPOTrainer`:

### 1. `strict_format_reward_func`

This function focuses solely on the structural and syntactical correctness of the model's output, specifically checking adherence to the predefined `<think>`/`<answer>` format. It provides partial credit to guide the model incrementally.

**Logic & Scoring:**

*   **Input:** Takes the list of model completions (`completions`).
*   **Checks:**
    1.  **Overall Structure:** Uses regex to verify if the output matches the pattern `^<think>([\s\S]*?)<\/think>\n<answer>([\s\S]*?)<\/answer>$`.
    2.  **Answer Content Format:** If the overall structure is correct, it examines the content within the `<answer>` tags. It checks if this content consists of *exactly one* non-empty line.
    3.  **Equation Syntax:** If there is exactly one line in `<answer>`, it further checks if this line looks like a valid mathematical equation by ensuring it only contains numbers, operators (`+`, `-`, `*`, `/`), parentheses, periods, whitespace, and crucially, includes an equals sign (`=`).
*   **Scoring:**
    *   **`0.0` points:** If the overall `<think>`/`<answer>` structure is incorrect.
    *   **`0.1` points:** If the overall structure is correct, BUT the content within `<answer>` does *not* meet the criteria (e.g., multiple lines, empty line, disallowed characters, missing '=').
    *   **`1.0` points:** If the overall structure is correct AND the `<answer>` tag contains exactly one line that follows the basic equation syntax rules.

### 2. `equation_reward_func`

This function evaluates the mathematical correctness and validity of the equation provided within the `<answer>` tag, ensuring it meets the specific constraints of the Countdown task.

**Logic & Scoring:**

*   **Input:** Takes model completions (`completions`), the corresponding original prompts (`prompts`), the target values (`target`), and the list of available numbers (`nums`) for each example.
*   **Checks:**
    1.  **Answer Extraction:** Extracts the content within the `<answer>` tags.
    2.  **Equation Parsing:** Splits the extracted content into a Left-Hand Side (LHS) and a Right-Hand Side (RHS) based on the `=` sign. Fails if no `=` is found or RHS is not a number.
    3.  **Character Validation (LHS):** Ensures the LHS contains only allowed characters (digits, operators `+-*/`, parentheses, dots, whitespace).
    4.  **Number Usage Validation (LHS):** Extracts all numbers used on the LHS. It then compares this list (sorted) against the required input `nums` list (sorted). They must match exactly – meaning each required number is used precisely once on the LHS.
    5.  **LHS Evaluation:** Uses Python's `eval()` function in a restricted environment (`{"__builtins__": None}, {}`) to calculate the numerical result of the LHS.
    6.  **Correctness Check:** Compares the calculated LHS result against both the numerical value of the RHS *and* the ground truth `target` value from the dataset. It uses a small tolerance (`1e-5`) for floating-point comparisons.
*   **Scoring:**
    *   **`1.0` points:** If the answer is correctly extracted, the equation is parseable, the LHS uses only allowed characters, *all* required numbers are used exactly once on the LHS, the LHS evaluates correctly, *and* the result matches both the RHS and the target value.
    *   **`0.0` points:** If *any* of the above checks fail (e.g., format error, incorrect number usage, disallowed characters, evaluation error, incorrect final value).

These two functions work together: `strict_format_reward_func` encourages the model to learn the basic output structure, while `equation_reward_func` pushes it towards generating mathematically sound and correct solutions for the specific task.
