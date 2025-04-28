# Tinyzero-Reproduce-Mini

This repository demonstrates Reinforcement Learning from Human Feedback (RLHF) for Large Language Models (LLMs) using Group Relative Policy Optimization (GRPO). It specifically implements a concept similar to **TinyZero**, which itself is an effort to reproduce aspects of powerful self-improving models like DeepSeek R1 Zero ([Link Placeholder for DeepSeek R1 Zero reference/paper]).

Leveraging the efficiency of the `unsloth` library and `trl`, this project aims to make advanced RLHF techniques significantly more accessible. While many RLHF pipelines demand massive GPU clusters, **this implementation is designed to run effectively on a single GPU with approximately >30GB of VRAM** (e.g., an NVIDIA A100 40GB, H100 40GB, or potentially consumer cards like RTX 4090/3090 with careful configuration).

The goal is to train a model to solve a specific mathematical task (inspired by the "Countdown" numbers game): generating an equation using a given set of numbers to reach a target value, while adhering to a predefined `<think>`/`<answer>` response format. This serves as a practical example of applying GRPO in a lower-resource setting.

## Background: TinyZero and Accessible RLHF

Traditional RLHF methods, especially those aiming for self-improvement loops seen in state-of-the-art models, often require substantial computational resources, putting them out of reach for many researchers and developers. Projects like DeepSeek R1 Zero showcase remarkable capabilities achieved through complex RL techniques but necessitate large-scale infrastructure.

**TinyZero** emerged as an initiative (often associated with community efforts or specific tutorials) to replicate some of these advanced self-improvement ideas on a much smaller, more manageable scale. It demonstrates that the core principles can be explored and yield improvements even without massive compute.

This repository builds upon the spirit of TinyZero and utilizes an `unsloth`-based framework (inspired by or directly adapted from an Unsloth tutorial) to provide a concrete, runnable example of GRPO-based RLHF. By focusing on efficiency (4-bit quantization, PEFT/LoRA, optimized kernels via Unsloth), it demonstrates the feasibility of performing this type of training on a single, moderately powerful GPU (~40GB VRAM).

## Features

*   **Accessible RLHF:** Designed to run on a single GPU with ~40GB VRAM, significantly lowering the barrier for RLHF experimentation compared to large-scale setups.
*   **TinyZero Concept:** Implements a GRPO-based RLHF loop inspired by efforts to reproduce advanced RL techniques (like DeepSeek R1 Zero) on a smaller scale.
*   **Reinforcement Learning (GRPO):** Uses `trl.GRPOTrainer` to fine-tune the LLM based on custom reward signals.
*   **Efficiency (`unsloth`):** Leverages `unsloth` for:
    *   Faster training and inference.
    *   Memory savings via 4-bit quantization (`load_in_4bit`).
    *   Efficient PEFT (LoRA) implementation.
*   **Custom Rewards:** Implements two reward functions focused on format adherence and mathematical correctness for the Countdown task.
*   **PEFT/LoRA:** Trains lightweight LoRA adapters for efficient model saving and sharing.
*   **Configurable:** Easily change the base model, LoRA parameters, training arguments, and dataset.

## Requirements

*   Python 3.9+
*   PyTorch (compatible with your CUDA version)
*   **A CUDA-enabled GPU with ~40GB VRAM minimum recommended** (e.g., A100 40GB, H100 PCIe 40GB, RTX 4090/3090 - memory usage may vary slightly based on config). Required for Unsloth and 4-bit quantization.
*   Key Python Libraries:
    *   `unsloth`
    *   `trl`
    *   `torch`
    *   `transformers`
    *   `datasets`
    *   `accelerate`
    *   `bitsandbytes` (for 4-bit/8-bit)
    *   `xformers` (optional, can provide speedups)
