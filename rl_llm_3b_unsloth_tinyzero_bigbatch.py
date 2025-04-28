from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "model/Qwen2.5-3B-Instruct" ,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)




# Load and prep dataset
SYSTEM_PROMPT = """
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

def create_question(numbers: list, target:int) -> str | None:
    return f"""
Using the numbers [{numbers}], create an equation that equals {target}.You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
"""

# uncomment middle messages for 1-shot prompting
def get_questions(split = "train") -> Dataset:
    #data = load_dataset("parquet", data_files="data/gsm8k-train-00000-of-00001.parquet", split="train")
    data = load_dataset("parquet", data_files="data/Countdown-Tasks-3to4.parquet", split="train")
    data = data.shuffle(seed=42).select(range(10000))

    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': create_question(x['nums'],x['target'])}
        ],
        #'answer': x['target']
    })
    return data 

dataset = get_questions()
print(dataset)
print(dataset[0])

# Reward functions



def equation_reward_func(prompts, completions, target, nums, **kwargs) -> list[float]:
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"\nQuestion:\n{q}", f"\nAnswer:\n{target[0]}", f"\nResponse:\n{responses[0]}")

    for completion, gt_val, numbers in zip(completions, target, nums):
      try:
        # Check if the format is correct
        # Use re.DOTALL in case the answer spans multiple lines
        match = re.search(r"<answer>(.*?)<\/answer>", completion[0]['content'], re.DOTALL)
        if match is None:
            #print('no answer tag found') # Clarified the print message
            rewards.append(0.0)
            continue

        # Extract the "answer" part from the completion
        full_equation = match.group(1).strip()
        #print(f"Extracted full equation: {full_equation}")

        # --- Modification Start ---
        # Split equation into LHS and RHS
        if '=' not in full_equation:
            #print("Equation missing '=' sign")
            rewards.append(0.0)
            continue
        
        parts = full_equation.split('=', 1) # Split only on the first '='
        lhs = parts[0].strip()
        rhs_str = parts[1].strip()

        # Try converting RHS to float for comparison
        try:
            rhs_val = float(rhs_str)
        except ValueError:
            #print(f"Could not convert RHS '{rhs_str}' to float")
            rewards.append(0.0)
            continue

        #print(f"LHS: {lhs}, RHS: {rhs_str}")
        # --- Modification End ---

        # Extract all numbers from the *Left Hand Side* (LHS) only
        used_numbers = [int(n) for n in re.findall(r'\d+', lhs)]
        #print(f"Numbers used in LHS: {used_numbers}")
        #print(f"Required numbers: {numbers}")

        # Check if all required numbers are used exactly once on the LHS
        if sorted(used_numbers) != sorted(numbers):
            #print("Mismatch between used numbers and required numbers")
            rewards.append(0.0)
            continue

        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace for the LHS
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, lhs):
           #print(f"LHS '{lhs}' contains disallowed characters")
           rewards.append(0.0)
           continue
        
        # Evaluate the *LHS* with restricted globals and locals
        result = eval(lhs, {"__builtins__": None}, {})
        #print(f"Evaluated LHS result: {result}")
        #print(f"Expected target value (gt): {gt_val}")
        #print(f"Value on RHS: {rhs_val}")

        # Check if the evaluated LHS result equals the RHS value AND the ground truth target
        # Use a small tolerance for float comparison
        if abs(float(result) - rhs_val) < 1e-5 and abs(float(result) - float(gt_val)) < 1e-5 :
            rewards.append(1.0)
        else:
            #print("Calculation result does not match RHS or target value")
            rewards.append(0.0)

      except Exception as e:
            # If evaluation or any other step fails, reward is 0
            print(f"An error occurred: {e}")
            rewards.append(0.0)


    print('correctness:', rewards[0])
    print('correctness all:', rewards)
    print('-'*20)
    return rewards



def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks format with partial and full credit:
    - 0.0 points: If the overall <think>...</think>\n<answer>...</answer> structure is incorrect.
    - 0.1 points: If the overall structure is correct, BUT the <answer> tag does NOT
                  contain exactly one valid mathematical equation line.
    - 1.0 points: If the overall structure is correct AND the <answer> tag contains
                  exactly one non-empty line which is a valid mathematical equation
                  (contains only allowed characters and includes '=').

    Args:
        completions (List[List[Dict[str, str]]]): Generated outputs from the model (batch).
        **kwargs: Additional keyword arguments.

    Returns:
        List[float]: Reward scores (0.0, 0.1, or 1.0) for each completion.
    """
    rewards = []
    # Regex for the overall structure: Needs <think>, newline, <answer>
    structure_regex = r"^<think>([\s\S]*?)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

    # Regex to check if a non-empty line ONLY contains characters typical of an equation.
    equation_line_pattern = r'[0-9+\-*/().=\s]+' 

    for completion_item in completions:
        current_reward = 0.0  # Start with zero reward
        try:
            content = completion_item[0]['content']

            # --- 1. Check overall structure ---
            match = re.search(structure_regex, content, re.DOTALL)

            if match:
                # Structure is OK - Grant partial credit (0.1 points)
                current_reward = 0.1

                # --- Now check the <answer> content for the additional points ---
                answer_content = match.group(2) # Extract content within <answer>

                # Split into lines and filter out lines that are empty or contain only whitespace
                lines = answer_content.splitlines()
                non_empty_lines = [line.strip() for line in lines if line.strip()]

                # --- 2. Check if there is *exactly one* non-empty line ---
                if len(non_empty_lines) == 1:
                    single_equation_line = non_empty_lines[0]

                    # --- 3. Check if the single line looks like a valid equation ---
                    # Check a) it only contains allowed characters AND b) it includes an '=' sign
                    if re.fullmatch(equation_line_pattern, single_equation_line) and '=' in single_equation_line:
                        # Both structure and answer content are correct - Add the remaining 
                        current_reward += 0.9 # Total becomes 1.0


        except Exception as e:
            # In case of any unexpected error during processing for this completion
            print(f"Error processing completion: {e}\nContent snippet: {completion_item[0]['content'][:100]}...")
            current_reward = 0.0 # Ensure reward is 0 on error

        rewards.append(current_reward)

    print('format rewards:',rewards[0])
    print('format rewards all:',rewards)
    return rewards


training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 32, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 1024,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 500,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "unsloth_outputs_tinyzero_bigbatch",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        strict_format_reward_func,
        equation_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_lora("grpo_saved_lora_tinyzero_bigbatch")