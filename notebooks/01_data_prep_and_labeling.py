"""
Stage 1: Data Preparation & Teacher Model Labeling
- Downloads Bitext customer support dataset from HuggingFace
- Samples 400 tickets and splits 350 train / 50 eval BEFORE labeling (preventing data leakage)
- Uses Groq (llama-3.3-70b-versatile, temp=0) as teacher model to produce structured labels
- Saves labeled datasets to data/train_tickets.jsonl and data/eval_tickets.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.schema import IssueCategory, Priority, CustomerSentiment, RequestedAction, TicketExtraction

SYSTEM_PROMPT = f"""You are an expert customer support triage assistant.
Extract structured metadata from the support ticket into valid JSON matching this exact schema:

Fields:
1. "issue_category": One of {[c.value for c in IssueCategory]}
2. "priority": One of {[p.value for p in Priority]}
   - "Low": general questions, feedback, informational
   - "Medium": non-urgent problems, minor delivery delays, general requests
   - "High": blockers, account locked, payment failed, unable to use service, urgent request
3. "customer_sentiment": One of {[s.value for s in CustomerSentiment]}
4. "requested_action": One of {[a.value for a in RequestedAction]}
5. "product_or_service": The specific product, service, or feature mentioned (or "General" if not specified)

Output ONLY a JSON object with these 5 keys. Do not include markdown codeblocks or extra text."""


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in .env or your environment.")
    from groq import Groq
    return Groq(api_key=api_key)


def get_preferred_groq_model(client) -> str:
    """Find the best available teacher model on the current Groq account."""
    try:
        models = [m.id for m in client.models.list().data]
        preferences = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
        ]
        for pref in preferences:
            if pref in models:
                return pref
        return models[0]
    except Exception:
        return "openai/gpt-oss-120b"


def label_ticket_with_groq(client, text: str, model_name: str = "openai/gpt-oss-120b") -> dict:
    """Uses Groq teacher model to label a single ticket into structured JSON."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ticket message:\n{text}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)


def load_raw_dataset(sample_count: int = 400):
    """Loads Bitext customer support dataset and takes sample_count items."""
    from datasets import load_dataset
    print("Fetching 'bitext/Bitext-customer-support-llm-chatbot-training-dataset' from HuggingFace...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    
    # Shuffle with fixed seed for reproducibility
    shuffled = ds.shuffle(seed=42)
    sampled = shuffled.select(range(min(sample_count, len(shuffled))))
    
    records = []
    for i, item in enumerate(sampled):
        # The bitext dataset has 'instruction' or 'query' representing the customer's message
        text = item.get("instruction") or item.get("message") or item.get("user_query") or item.get("text")
        records.append({
            "id": f"ticket_{i:04d}",
            "text": text.strip(),
            "raw_intent": item.get("intent", ""),
            "raw_category": item.get("category", "")
        })
    return records


def label_and_save(client, records: list, output_path: Path, model_name: str):
    """Labels records with resume support and saves to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load already labeled records if resuming
    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_ids.add(data["id"])
        print(f"Resuming {output_path.name}: {len(existing_ids)} already labeled.")

    with open(output_path, "a", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            if rec["id"] in existing_ids:
                continue
            
            # Label with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    extraction_dict = label_ticket_with_groq(client, rec["text"], model_name=model_name)
                    # Validate against Pydantic schema
                    validated = TicketExtraction(**extraction_dict)
                    
                    labeled_entry = {
                        "id": rec["id"],
                        "text": rec["text"],
                        "extraction": validated.model_dump(),
                        "raw_metadata": {
                            "raw_intent": rec.get("raw_intent", ""),
                            "raw_category": rec.get("raw_category", "")
                        }
                    }
                    f.write(json.dumps(labeled_entry, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{idx+1}/{len(records)}] Labeled {rec['id']}: {validated.issue_category.value} | {validated.priority.value}")
                    # Brief sleep to stay within free tier rate limits
                    time.sleep(0.3)
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for {rec['id']}: {e}")
                    time.sleep(2.0 * (attempt + 1))
            else:
                print(f"Failed to label {rec['id']} after {max_retries} attempts.")


def main():
    print("=== Stage 1: Data Preparation & Labeling ===")
    client = get_groq_client()
    model_name = get_preferred_groq_model(client)
    print(f"Using Groq Teacher Model: {model_name}")
    
    # 1. Load 400 raw samples
    all_sampled = load_raw_dataset(sample_count=400)
    print(f"Loaded {len(all_sampled)} raw samples.")
    
    # 2. Split 350 train / 50 eval BEFORE any labeling/formatting
    train_records = all_sampled[:350]
    eval_records = all_sampled[350:400]
    print(f"Split dataset: {len(train_records)} Train / {len(eval_records)} Eval.")
    
    # 3. Label and save
    data_dir = PROJECT_ROOT / "data"
    print("\n--- Labeling Train Set (350 tickets) ---")
    label_and_save(client, train_records, data_dir / "train_tickets.jsonl", model_name=model_name)
    
    print("\n--- Labeling Eval Set (50 tickets) ---")
    label_and_save(client, eval_records, data_dir / "eval_tickets.jsonl", model_name=model_name)
    
    print("\nStage 1 Data Preparation and Labeling complete!")


if __name__ == "__main__":
    main()
