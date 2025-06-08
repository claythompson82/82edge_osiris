#!/usr/bin/env python
import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def format_prompt(sample):
    """Formats a sample from the dataset into a single string for training.
    Assumes 'prompt' and 'response' keys in the dataset.
    """
    return f"Prompt:\n{sample['prompt']}\n\nResponse:\n{sample['response']}"


def main():
    parser = argparse.ArgumentParser(
        description="Run QLoRA fine-tuning for Phi-3-mini."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/tmp/training_data.jsonl",
        help="Path to the training dataset in JSONL format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/qlora_adapters",
        help="Directory to save the trained LoRA adapters.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",  # More practical default for HF training
        help="Path or Hugging Face identifier for the base model.",
    )
    # Add arguments for LoRA configuration if needed, but using fixed values from prompt for now
    # parser.add_argument("--lora_r", type=int, default=16)
    # parser.add_argument("--lora_alpha", type=int, default=32)
    # parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    if args.base_model_path.endswith(".onnx"):
        print("Warning: The provided --base_model_path is an ONNX file.")
        print(
            "QLoRA fine-tuning with TRL/PEFT typically requires a PyTorch-based Hugging Face model."
        )
        print(
            f"Attempting to use '{args.base_model_path}' but it might fail if not convertible."
        )
        print(
            "It is recommended to use a Hugging Face model identifier like 'microsoft/Phi-3-mini-4k-instruct'."
        )

    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Please ensure the dataset is a valid JSONL file with 'prompt' and 'response' keys."
        )
        return

    # 2. Determine compute dtype (bf16 or fp32)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using compute dtype: {'bfloat16' if use_bf16 else 'float32'}")

    # 3. Configure BitsAndBytes for QLoRA (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # As per prompt
        bnb_4bit_quant_type="nf4",  # As per prompt
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 4. Load Model and Tokenizer
    print(f"Loading base model: {args.base_model_path} with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically distribute model on available GPUs
            trust_remote_code=True,  # Required for some models like Phi-3
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,  # Required for some models like Phi-3
        )
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # 5. PEFT Configuration (LoRA)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,  # LoRA rank - from prompt (implicitly, as it was one of the params mentioned)
        lora_alpha=32,  # LoRA alpha - from prompt
        lora_dropout=0.05,  # LoRA dropout - from prompt
        bias="none",
        task_type="CAUSAL_LM",
        # Example target_modules for Phi-3, might need adjustment based on model architecture
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention layers
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP layers
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Training Arguments
    # These are minimal arguments. More can be added for better control.
    training_args = TrainingArguments(
        output_dir=os.path.join(
            args.output_dir, "trainer_checkpoints"
        ),  # Checkpoints for trainer state
        per_device_train_batch_size=1,  # Keep low for memory
        gradient_accumulation_steps=4,  # Accumulate gradients
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,  # For a quick test, adjust as needed
        save_strategy="epoch",  # Save adapters at the end of each epoch
        bf16=use_bf16,
        fp16=not use_bf16
        and torch.cuda.is_available(),  # Use fp16 if bf16 not available but CUDA is
        report_to="none",  # Disable wandb/tensorboard for this example
    )

    # 7. SFTTrainer (Supervised Fine-tuning Trainer)
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=None,  # We use formatting_func instead
        formatting_func=format_prompt,  # Use our custom formatting function
        max_seq_length=2048,  # Adjust based on model and data
        tokenizer=tokenizer,
        peft_config=lora_config,  # Pass peft_config here
    )

    # 8. Start Training
    print("Starting QLoRA fine-tuning...")
    try:
        trainer.train()
        print("Training completed.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 9. Save Adapters
    print(f"Saving trained LoRA adapters to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        # Save the PEFT adapter model
        trainer.save_model(
            args.output_dir
        )  # This saves the adapter_model.bin and adapter_config.json
        # The tokenizer is usually saved alongside if it was modified or to ensure consistency.
        # SFTTrainer with PEFT model should handle saving adapter correctly.
        # If tokenizer needs to be saved explicitly:
        # tokenizer.save_pretrained(args.output_dir)
        print(f"Adapters saved successfully to {args.output_dir}")
    except Exception as e:
        print(f"Error saving adapters: {e}")
        return

    print("QLoRA fine-tuning script finished successfully.")


if __name__ == "__main__":
    main()
    # Example usage from bash, assuming data is ready:
    # python scripts/run_qlora.py \
    #   --dataset_path /tmp/training_data.jsonl \
    #   --output_dir /tmp/qlora_adapters \
    #   --base_model_path "microsoft/Phi-3-mini-4k-instruct"
    # (This would typically be run by nightly_qlora.sh which sets these paths)
