"""
Stage 7: FastAPI Deployment for AI Support Ticket Assistant
- POST /api/ticket/process : End-to-end processing of customer support tickets
- POST /api/ticket/approve : Human-in-the-loop authorization for escalated/sensitive actions
- GET /health : System readiness and component status
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.schema import TicketExtraction, IssueCategory, Priority, CustomerSentiment, RequestedAction
from src.guardrails import validate_and_sanitize_input, validate_extraction_schema, safe_default_extraction
from src.extraction import TicketExtractor
from src.retrieval import HybridRAGPipeline
from src.agent import SupportTicketAgent, execute_order_cancellation

app = FastAPI(
    title="AI Support Ticket Assistant API",
    description="Automated triage, structured extraction, hybrid RAG retrieval, and LangGraph agent workflow.",
    version="1.0.0"
)

# Global component singletons
rag_pipeline: Optional[HybridRAGPipeline] = None
agent: Optional[SupportTicketAgent] = None
extractor: Optional[TicketExtractor] = None


@app.on_event("startup")
def startup_event():
    """Initializes models and retrieval pipeline on app startup."""
    global rag_pipeline, agent, extractor
    print("Initializing AI Support Ticket Assistant services...")
    
    docs_dir = PROJECT_ROOT / "docs_kb"
    adapter_path = PROJECT_ROOT / "models" / "qwen2.5_lora_adapter"
    
    try:
        rag_pipeline = HybridRAGPipeline(docs_dir=docs_dir)
    except Exception as e:
        print(f"Warning: RAG pipeline init: {e}")

    try:
        agent = SupportTicketAgent(rag_pipeline=rag_pipeline)
    except Exception as e:
        print(f"Warning: Agent init: {e}")

    # Respect USE_LOCAL_MODEL environment variable (default "false" for cloud deployment)
    use_local = os.environ.get("USE_LOCAL_MODEL", "false").lower() in ("true", "1", "t", "yes")
    try:
        extractor = TicketExtractor(
            adapter_path=str(adapter_path) if (use_local and adapter_path.exists()) else None,
            use_teacher_fallback=not use_local
        )
        mode = "Local QLoRA adapter" if use_local else "Groq API (Lightweight Cloud Mode)"
        print(f"Extractor initialized: {mode}")
    except Exception as e:
        print(f"Warning: Extractor init: {e}")
        extractor = TicketExtractor(use_teacher_fallback=True)

    print("All backend services initialized successfully!")


class TicketRequest(BaseModel):
    message: str = Field(..., description="Raw text message from customer", min_length=3)
    ticket_id: Optional[str] = Field(default="ticket_001", description="Unique ticket tracking ID")


class TicketResponse(BaseModel):
    ticket_id: str
    sanitized_message: str
    extraction: TicketExtraction
    is_safe: bool
    requires_human_approval: bool
    escalation_reason: Optional[str]
    tools_called: List[str]
    agent_response: str
    processing_time_ms: float


class ApprovalRequest(BaseModel):
    ticket_id: str
    action: str = Field(..., description="Action to approve: 'REFUND' or 'CANCEL_ORDER'")
    order_id: str
    approved: bool = Field(..., description="True to approve action, False to deny")
    manager_notes: Optional[str] = None


class ApprovalResponse(BaseModel):
    ticket_id: str
    status: str
    action_result: Dict[str, Any]
    message: str


@app.get("/health")
def health_check():
    """Returns system status and component health."""
    adapter_path = PROJECT_ROOT / "models" / "qwen2.5_lora_adapter"
    return {
        "status": "healthy",
        "service": "AI Support Ticket Assistant",
        "version": "1.0.0",
        "components": {
            "rag_knowledge_base": rag_pipeline is not None,
            "agent_router": agent is not None,
            "lora_fine_tuned_model": adapter_path.exists(),
            "groq_api_connected": bool(os.environ.get("GROQ_API_KEY"))
        }
    }


@app.post("/api/ticket/process", response_model=TicketResponse)
def process_ticket(payload: TicketRequest):
    """
    Main customer support pipeline:
    1. Guardrails: Injection check & PII redaction
    2. Extraction: Fine-tuned QLoRA structured extraction
    3. LangGraph Agent: Specialized routing, RAG retrieval & business logic
    4. Output Safety & Schema enforcement
    """
    start_time = time.time()
    raw_text = payload.message

    # 1. Input Guardrails
    sanitized_text, is_safe, security_msg = validate_and_sanitize_input(raw_text)
    if not is_safe:
        return TicketResponse(
            ticket_id=payload.ticket_id,
            sanitized_message="[BLOCKED BY SECURITY GUARDRAIL]",
            extraction=safe_default_extraction(),
            is_safe=False,
            requires_human_approval=True,
            escalation_reason="Prompt injection / security trigger detected.",
            tools_called=["input_security_guardrail"],
            agent_response="Your message could not be processed due to a security violation.",
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )

    # 2. Structured Extraction (QLoRA / Groq Fallback)
    global extractor, agent
    if extractor is None:
        extractor = TicketExtractor(use_teacher_fallback=True)
    if agent is None:
        agent = SupportTicketAgent(rag_pipeline=rag_pipeline)

    extraction, valid = extractor.extract(sanitized_text)

    # 3. Agent Execution
    agent_state = agent.process_ticket(
        raw_message=sanitized_text,
        extraction=extraction,
        ticket_id=payload.ticket_id
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return TicketResponse(
        ticket_id=payload.ticket_id,
        sanitized_message=sanitized_text,
        extraction=extraction,
        is_safe=True,
        requires_human_approval=agent_state.get("requires_human_approval", False),
        escalation_reason=agent_state.get("escalation_reason"),
        tools_called=agent_state.get("tools_called", []),
        agent_response=agent_state.get("agent_response", ""),
        processing_time_ms=elapsed_ms
    )


@app.post("/api/ticket/approve", response_model=ApprovalResponse)
def approve_consequential_action(payload: ApprovalRequest):
    """
    Human-in-the-loop endpoint: Approves or denies paused actions (refunds, cancellations).
    """
    if not payload.approved:
        return ApprovalResponse(
            ticket_id=payload.ticket_id,
            status="Denied",
            action_result={"approved": False, "reason": payload.manager_notes or "Action denied by reviewer."},
            message=f"Action '{payload.action}' for ticket {payload.ticket_id} was rejected."
        )

    if payload.action.upper() in ["CANCEL_ORDER", "REFUND"]:
        res = execute_order_cancellation(payload.order_id)
        return ApprovalResponse(
            ticket_id=payload.ticket_id,
            status="Executed",
            action_result=res,
            message=f"Action '{payload.action}' successfully executed for order {payload.order_id}."
        )

    raise HTTPException(status_code=400, detail=f"Unsupported action: {payload.action}")
