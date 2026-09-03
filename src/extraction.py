"""
Extraction module: Loads fine-tuned Qwen2.5-1.5B-Instruct LoRA model
and performs structured customer support ticket extraction.
Includes fallback support for teacher model / local inference.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple
import torch

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
    """Builds standard chat messages for Qwen / chat models."""
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
    """Handles inference with the fine-tuned LoRA model or fallback teacher model."""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        use_teacher_fallback: bool = False
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_teacher_fallback = use_teacher_fallback
        self.model = None
        self.tokenizer = None
        
        if not use_teacher_fallback and adapter_path and Path(adapter_path).exists():
            self._load_local_model()
            
    def _load_local_model(self):
        """Loads Qwen2.5-1.5B base model + LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
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
        print("Model successfully loaded!")

    def extract(self, ticket_text: str) -> Tuple[TicketExtraction, bool]:
        """
        Extracts structured fields from ticket.
        Returns (TicketExtraction, is_valid).
        """
        # If model is loaded locally, run local inference
        if self.model is not None and self.tokenizer is not None:
            return self._extract_local(ticket_text)
        
        # Fallback to teacher model via Groq
        return self._extract_groq(ticket_text)

    def _extract_local(self, ticket_text: str) -> Tuple[TicketExtraction, bool]:
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
            # Return safe fallback if parsing fails
            return TicketExtraction(
                issue_category=IssueCategory.OTHER,
                priority=Priority.MEDIUM,
                customer_sentiment=CustomerSentiment.NEUTRAL,
                requested_action=RequestedAction.OTHER,
                product_or_service="General"
            ), False

    def _extract_groq(self, ticket_text: str) -> Tuple[TicketExtraction, bool]:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        client = Groq(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=build_chat_prompt(ticket_text),
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return TicketExtraction(**parsed), True
        except Exception:
            return TicketExtraction(
                issue_category=IssueCategory.OTHER,
                priority=Priority.MEDIUM,
                customer_sentiment=CustomerSentiment.NEUTRAL,
                requested_action=RequestedAction.OTHER,
                product_or_service="General"
            ), False
