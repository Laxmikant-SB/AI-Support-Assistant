"""
Stage 2: QLoRA Fine-Tuning on Qwen/Qwen2.5-1.5B-Instruct
- Quantization: 4-bit NF4 via bitsandbytes
- LoRA configuration: rank=8, alpha=16, targets=[q_proj, k_proj, v_proj, o_proj]
- Optimized for 6GB VRAM (gradient accumulation, batch size 1/2, fp16)
- Tracks train_loss and eval_loss per epoch
- Evaluates on held-out 50 eval examples: per-field exact match, invalid-JSON rate, invalid-enum rate
"""

import json
import os
import sys
import argparse
from pathlib import Path
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import (
    IssueCategory,
    Priority,
    CustomerSentiment,
    RequestedAction,
    TicketExtraction,
)
from src.extraction import SYSTEM_PROMPT, clean_json_response


def format_chat_record(entry: dict, tokenizer) -> str:
    """Formats a labeled ticket entry into Qwen chat template."""
    extraction_json_str = json.dumps(entry["extraction"], ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer ticket:\n{entry['text']}"},
        {"role": "assistant", "content": extraction_json_str}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def load_dataset_from_jsonl(file_path: Path):
    """Loads records from JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def train_qlora(
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 3,
    lr: float = 2e-4
):
    """Fine-tunes Qwen2.5-1.5B with QLoRA on 6GB VRAM."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. Load train and eval records
    print(f"Loading datasets from {train_path} and {eval_path}...")
    raw_train = load_dataset_from_jsonl(train_path)
    raw_eval = load_dataset_from_jsonl(eval_path)
    print(f"Loaded {len(raw_train)} train records, {len(raw_eval)} eval records.")

    train_texts = [format_chat_record(r, tokenizer) for r in raw_train]
    eval_texts = [format_chat_record(r, tokenizer) for r in raw_eval]

    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})

    # 2. 4-bit Quantization Config (NF4)
    print("Configuring 4-bit NF4 Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    device_map = "auto" if torch.cuda.is_available() else None
    print(f"Loading base model (Device: {'CUDA' if torch.cuda.is_available() else 'CPU'})...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 3. LoRA Configuration
    print("Setting up LoRA adapter (r=8, alpha=16, targets=[q_proj, k_proj, v_proj, o_proj])...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Ensure trainable parameters are in float32 for stable gradient scaling on Windows
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            
    model.print_trainable_parameters()

    # 4. Training Arguments tuned for 6GB VRAM
    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_train_epochs=epochs,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch"
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args
    )

    print("\n--- Starting Training ---")
    train_result = trainer.train()
    print("\nTraining Finished!")
    print(f"Global steps: {train_result.global_step}, Train Loss: {train_result.training_loss:.4f}")

    # 6. Save LoRA Adapter
    adapter_path = output_dir / "qwen2.5_lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving fine-tuned adapter to {adapter_path}...")
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print("Saved successfully!")

    return adapter_path


def evaluate_model(eval_path: Path, adapter_path: Path, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Evaluates fine-tuned model on held-out 50 examples:
    - Per-field exact match accuracy
    - Invalid-JSON rate
    - Invalid-enum rate
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n=== Evaluating on Held-Out Test Set ({eval_path}) ===")
    raw_eval = load_dataset_from_jsonl(eval_path)
    
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    total = len(raw_eval)
    invalid_json_count = 0
    invalid_enum_count = 0
    
    correct_counts = {
        "issue_category": 0,
        "priority": 0,
        "customer_sentiment": 0,
        "requested_action": 0,
        "product_or_service": 0,
    }

    valid_categories = {c.value for c in IssueCategory}
    valid_priorities = {p.value for p in Priority}
    valid_sentiments = {s.value for s in CustomerSentiment}
    valid_actions = {a.value for a in RequestedAction}

    print(f"Running inference on {total} eval examples...")
    for idx, item in enumerate(raw_eval):
        ground_truth = item["extraction"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer ticket:\n{item['text']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # 1. Parse JSON
        try:
            pred_dict = clean_json_response(raw_text)
        except Exception:
            invalid_json_count += 1
            print(f"[{idx+1}/{total}] Invalid JSON: {raw_text[:100]}")
            continue

        # 2. Check Enums
        has_invalid_enum = False
        if pred_dict.get("issue_category") not in valid_categories:
            has_invalid_enum = True
        if pred_dict.get("priority") not in valid_priorities:
            has_invalid_enum = True
        if pred_dict.get("customer_sentiment") not in valid_sentiments:
            has_invalid_enum = True
        if pred_dict.get("requested_action") not in valid_actions:
            has_invalid_enum = True
            
        if has_invalid_enum:
            invalid_enum_count += 1

        # 3. Check Exact Matches
        for field in correct_counts.keys():
            pred_val = str(pred_dict.get(field, "")).strip().lower()
            gt_val = str(ground_truth.get(field, "")).strip().lower()
            if pred_val == gt_val:
                correct_counts[field] += 1

    # Report Results
    results = {
        "total_eval_samples": total,
        "invalid_json_rate": round(invalid_json_count / total, 4),
        "invalid_enum_rate": round(invalid_enum_count / total, 4),
        "exact_match_accuracy": {
            field: round(correct_counts[field] / total, 4)
            for field in correct_counts
        }
    }
    
    print("\n" + "="*50)
    print("STAGE 2 EVALUATION METRICS ON HELD-OUT 50 EXAMPLES:")
    print("="*50)
    print(f"Invalid JSON Rate: {results['invalid_json_rate'] * 100:.1f}%")
    print(f"Invalid Enum Rate: {results['invalid_enum_rate'] * 100:.1f}%")
    print("Per-Field Exact Match Accuracy:")
    for field, acc in results["exact_match_accuracy"].items():
        print(f"  - {field:20s}: {acc * 100:.1f}%")
    print("="*50)
    
    # Save evaluation scorecard
    eval_scorecard_path = adapter_path.parent / "eval_scorecard_stage2.json"
    with open(eval_scorecard_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved eval scorecard to {eval_scorecard_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning on Qwen2.5-1.5B")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    train_file = data_dir / "train_tickets.jsonl"
    eval_file = data_dir / "eval_tickets.jsonl"
    adapter_dir = models_dir / "qwen2.5_lora_adapter"

    if args.eval_only:
        evaluate_model(eval_file, adapter_dir)
    else:
        adapter_path = train_qlora(
            train_path=train_file,
            eval_path=eval_file,
            output_dir=models_dir,
            epochs=args.epochs,
            lr=args.lr
        )
        evaluate_model(eval_file, adapter_path)


if __name__ == "__main__":
    main()
