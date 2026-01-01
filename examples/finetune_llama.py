"""
Fine-tune Llama Example
=======================

Complete workflow: Generate data → Fine-tune with Unsloth → Evaluate
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
# STEP 1: Generate Training Data
# =============================================================================

print("📚 Step 1: Loading policy...")
policy = Policy.from_file("company_handbook.pdf")

print("🔄 Step 2: Generating training data...")
pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,       # Fast generation
    grading_model=Google.GEMINI_25_PRO, # Quality grading
    dataset_type=DatasetType.SFT,       # Chat format
    max_iterations=3,                   # Up to 3 refinement attempts
)

dataset = pipeline.generate(policy, traces=1000)

# Save dataset
dataset.save("train_sft.jsonl", format="sft")  # For supervised fine-tuning

print(f"✅ Generated {len(dataset)} traces ({dataset.passing_rate:.0%} pass rate)")

# =============================================================================
# STEP 2: Fine-tune with Unsloth (fast LoRA training)
# =============================================================================

print("\n🔧 Step 3: Fine-tuning with Unsloth...")

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Load our generated dataset
train_data = load_dataset("json", data_files="train_sft.jsonl", split="train")

# Load base model with Unsloth (4-bit quantized for speed)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Format for chat
def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="./policy-llama",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
    ),
    formatting_func=format_chat,
    max_seq_length=4096,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./policy-llama-final")
tokenizer.save_pretrained("./policy-llama-final")

print("✅ Fine-tuning complete! Model saved to ./policy-llama-final")

# =============================================================================
# STEP 3: Quick Evaluation
# =============================================================================

print("\n📊 Step 4: Quick evaluation...")

# Load fine-tuned model for inference
FastLanguageModel.for_inference(model)

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

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=500, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🤖 Fine-tuned model response:")
print(response.split("assistant")[-1].strip())
