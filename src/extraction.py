"""
Extraction module: Performs structured customer support ticket extraction.
Supports:
1. Groq API inference (llama-3.3-70b-versatile / openai/gpt-oss-120b) for lightweight deployment (USE_LOCAL_MODEL=false).
2. Local Qwen2.5-1.5B + LoRA adapter inference on local GPU (USE_LOCAL_MODEL=true).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from src.schema import (
    IssueCategory,
    Priority,
    CustomerSentiment,
    RequestedAction,
    TicketExtraction,
)

SYSTEM_PROMPT = """You are an expert customer support classifier.
Extract structured metadata from the customer ticket into a valid JSON object matching this schema:
- "issue_category": One of [Payment Issue, Delivery Issue, Login Issue, Account Issue, Refund Request, Order Cancellation, Order Modification, Invoice/Billing Request, Product Defect, Technical Issue, Agent/Human Handoff, Other]
- "priority": One of [Low, Medium, High]
- "customer_sentiment": One of [Positive, Neutral, Negative]
- "requested_action": One of [Refund, Replacement, Cancel Order, Technical Support, Information Request, Account Recovery, Other]
- "product_or_service": Specific product, feature or service mentioned (or "General")

Output ONLY raw JSON with these 5 keys. No extra commentary."""


def build_chat_prompt(ticket_text: str) -> list:
    """Builds standard chat messages for Qwen / Groq chat models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer ticket:\n{ticket_text}"}
    ]


def clean_json_response(raw_text: str) -> dict:
    """Extracts and parses JSON object from model output text."""
    text = raw_text.strip()
    
    # Remove markdown code fences if present
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
        
    # Attempt direct json load
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Regex find the outermost {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse valid JSON from response: {raw_text[:200]}")


class TicketExtractor:
    """
    Handles ticket metadata extraction.
    Loads local QLoRA model ONLY if USE_LOCAL_MODEL=true; otherwise uses Groq API.
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        use_teacher_fallback: bool = False
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = device
        
        # Check environment flag (default: "false" for deployment compatibility)
        use_local_env = os.environ.get("USE_LOCAL_MODEL", "false").lower() in ("true", "1", "t", "yes")
        
        if use_local_env and not use_teacher_fallback and adapter_path and Path(adapter_path).exists():
            self._load_local_model()
        else:
            print("TicketExtractor: Configured for lightweight API mode (USE_LOCAL_MODEL=false).")

    def _load_local_model(self):
        """Lazy loads PyTorch, Transformers & LoRA adapter on demand."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading tokenizer from {self.base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading base model {self.base_model_name} (4-bit QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        print(f"Loading LoRA adapter from {self.adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        print("Local QLoRA Model successfully loaded!")

    def extract(self, ticket_text: str) -> Tuple[TicketExtraction, bool]:
        """
        Extracts structured fields from ticket.
        Returns (TicketExtraction, is_valid).
        """
        if self.model is not None and self.tokenizer is not None:
            return self._extract_local(ticket_text)
        
        return self._extract_groq(ticket_text)

    def _extract_local(self, ticket_text: str) -> Tuple[TicketExtraction, bool]:
        import torch
        messages = build_chat_prompt(ticket_text)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        try:
            parsed = clean_json_response(raw_output)
            validated = TicketExtraction(**parsed)
            return validated, True
        except Exception:
            return safe_default_extraction(), False

    def _extract_groq(self, ticket_text: str) -> Tuple[TicketExtraction, bool]:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY is not set. Returning safe default extraction.")
            return safe_default_extraction(), False
            
        client = Groq(api_key=api_key)
        models_to_try = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=build_chat_prompt(ticket_text),
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw = response.choices[0].message.content.strip()
                parsed = json.loads(raw)
                return TicketExtraction(**parsed), True
            except Exception as e:
                continue

        return safe_default_extraction(), False


def safe_default_extraction() -> TicketExtraction:
    """Fallback default if extraction fails."""
    return TicketExtraction(
        issue_category=IssueCategory.OTHER,
        priority=Priority.MEDIUM,
        customer_sentiment=CustomerSentiment.NEUTRAL,
        requested_action=RequestedAction.OTHER,
        product_or_service="General"
    )
