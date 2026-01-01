"""
Fine-tune Llama Example
=======================

Complete workflow: Generate data → Fine-tune with Tinker → Evaluate

Prerequisites:
    pip install tinker
    Sign up at https://thinkingmachines.ai/tinker/ for API access
    Set TINKER_API_KEY environment variable
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from synkro.pipelines import create_pipeline
from synkro.models.google import Google
from synkro.types import DatasetType
from synkro.core.policy import Policy

# =============================================================================
# STEP 1: Generate Training Data (skip if already exists)
# =============================================================================

train_file = Path("train_sft.jsonl")

if train_file.exists():
    print("📁 Found existing train_sft.jsonl, skipping data generation...")
else:
    print("📚 Step 1: Loading policy...")
    policy_path = Path(__file__).parent / "policies" / "Expense-Reimbursement-Policy.docx"
    policy = Policy.from_file(policy_path)

    print("🔄 Step 2: Generating training data...")
    pipeline = create_pipeline(
        model=Google.GEMINI_25_FLASH,       # Fast generation
        grading_model=Google.GEMINI_25_PRO, # Quality grading
        dataset_type=DatasetType.SFT,       # Chat format
        max_iterations=3,                   # Up to 3 refinement attempts
    )

    dataset = pipeline.generate(policy, traces=100)

    # Save dataset
    dataset.save("train_sft.jsonl", format="sft")  # For supervised fine-tuning

    print(f"✅ Generated {len(dataset)} traces ({dataset.passing_rate:.0%} pass rate)")

# =============================================================================
# STEP 2: Fine-tune with Tinker (remote GPU training - no local GPU needed)
# =============================================================================

print("\n🔧 Step 3: Fine-tuning with Tinker...")

from datasets import load_dataset
import tinker
from tinker import types

# Load our generated dataset
train_data = load_dataset("json", data_files="train_sft.jsonl", split="train")

# Model to fine-tune
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Initialize Tinker client (training happens on remote GPUs)
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=16,  # LoRA rank
)

# Get tokenizer from Tinker
tokenizer = training_client.get_tokenizer()


def apply_llama3_chat_template(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Apply Llama 3 chat template manually."""
    output = "<|begin_of_text|>"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        output += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    if add_generation_prompt:
        output += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return output


def messages_to_datum(messages: list[dict]) -> types.Datum:
    """Convert chat messages to a Tinker Datum for supervised learning."""
    # Find the last assistant message (completion we want to learn)
    prompt_messages = []
    completion_text = ""

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and i == len(messages) - 1:
            completion_text = msg["content"]
        else:
            prompt_messages.append(msg)

    # Tokenize prompt (using Llama 3 chat template)
    prompt_text = apply_llama3_chat_template(prompt_messages, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_weights = [0] * len(prompt_tokens)  # Don't compute loss on prompt

    # Tokenize completion (with end token)
    completion_with_end = completion_text + "<|eot_id|>"
    completion_tokens = tokenizer.encode(completion_with_end, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)  # Compute loss on completion

    # Combine tokens and weights
    all_tokens = prompt_tokens + completion_tokens
    all_weights = prompt_weights + completion_weights

    # Create shifted targets for next-token prediction
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = all_weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            target_tokens=target_tokens,
            weights=weights,
        ),
    )


# Training loop with Tinker
num_epochs = 3
batch_size = 4
learning_rate = 1e-4

print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    for i in range(0, len(train_data), batch_size):
        batch_end = min(i + batch_size, len(train_data))

        # Convert batch to Datum objects
        batch_data = []
        for j in range(i, batch_end):
            example = train_data[j]
            datum = messages_to_datum(example["messages"])
            batch_data.append(datum)

        # Forward pass and gradient computation on remote GPUs
        fwd_bwd_result = training_client.forward_backward(
            data=batch_data,
            loss_fn="cross_entropy",
        ).result()

        # Update model weights
        training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        ).result()

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {batch_end}/{len(train_data)} examples")

# Save trained model and get sampling client
sampling_client = training_client.save_weights_and_get_sampling_client(
    name="policy-llama"
)

print("✅ Fine-tuning complete! Model saved to Tinker as 'policy-llama'")

# =============================================================================
# STEP 3: Quick Evaluation
# =============================================================================

print("\n📊 Step 4: Quick evaluation...")

# Test with a sample scenario
test_scenario = """
I need to expense a $350 software subscription for a project management tool
that's not on the pre-approved list. The tool is critical for our team's
workflow. What's the process?
"""

messages = [
    {"role": "system", "content": "You are a helpful policy compliance assistant."},
    {"role": "user", "content": test_scenario},
]

# Format prompt for sampling
prompt_text = apply_llama3_chat_template(messages, add_generation_prompt=True)
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

# Generate response using Tinker's sampling API (runs on remote GPUs)
result = sampling_client.sample(
    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
    sampling_params=types.SamplingParams(
        max_tokens=500,
        temperature=0.7,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    ),
    num_samples=1,
).result()

# Decode the response
response_tokens = result.sequences[0].tokens
response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

print("\n🤖 Fine-tuned model response:")
print(response_text)

# Save output to file
output_file = Path("finetune_output.txt")
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("Fine-tuned Model Evaluation Output\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model: {BASE_MODEL}\n")
    f.write(f"LoRA Rank: 16\n")
    f.write(f"Training Epochs: {num_epochs}\n")
    f.write(f"Training Examples: {len(train_data)}\n\n")
    f.write("-" * 80 + "\n")
    f.write("Test Scenario:\n")
    f.write("-" * 80 + "\n")
    f.write(test_scenario.strip() + "\n\n")
    f.write("-" * 80 + "\n")
    f.write("Model Response:\n")
    f.write("-" * 80 + "\n")
    f.write(response_text + "\n")

print(f"\n📁 Saved output to: {output_file}")
