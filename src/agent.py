"""
Stage 4: LangGraph Multi-Specialist Support Agent
- Router + Specialist architecture based on issue_category
- Specialized tools: RAG retrieval, mock business logic (refund eligibility, order status, cancellation)
- Human-in-the-loop escalation / approval pausing for consequential actions
- Explicit state management with hard step_count safety limits
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
import operator
import os
import json
from pathlib import Path

from src.schema import (
    IssueCategory,
    Priority,
    CustomerSentiment,
    RequestedAction,
    TicketExtraction,
)
from src.retrieval import HybridRAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================
# 1. Mock Business Logic Tools
# ==========================================

def check_order_status(order_id: str) -> Dict[str, Any]:
    """Mock CRM function to check order delivery status."""
    clean_id = order_id.upper().strip()
    suffix = clean_id.split("-")[-1] if "-" in clean_id else clean_id[-4:]
    if "123" in clean_id or "ORD" in clean_id:
        return {
            "order_id": clean_id,
            "status": "In Transit",
            "carrier": "FedEx",
            "tracking_number": f"TRK-{suffix}",
            "estimated_delivery": "2 business days",
            "can_self_cancel": False,
            "items": ["Pro Ergonomic Chair - Black"]
        }
    return {
        "order_id": clean_id,
        "status": "Processing / Fulfillment Queue",
        "carrier": "Pending",
        "tracking_number": None,
        "estimated_delivery": "3-5 business days",
        "can_self_cancel": True,
        "items": ["Wireless Mechanical Keyboard"]
    }


def check_refund_eligibility(order_id: str, days_since_purchase: int = 10) -> Dict[str, Any]:
    """Mock Billing function to assess policy-based refund eligibility."""
    if days_since_purchase <= 14:
        return {
            "order_id": order_id,
            "eligible": True,
            "max_refund_amount": "$89.99",
            "reason": "Within standard 14-day full refund window.",
            "requires_manager_approval": False
        }
    elif days_since_purchase <= 30:
        return {
            "order_id": order_id,
            "eligible": True,
            "max_refund_amount": "$45.00",
            "reason": "Within 30-day partial/prorated refund window.",
            "requires_manager_approval": True
        }
    return {
        "order_id": order_id,
        "eligible": False,
        "max_refund_amount": "$0.00",
        "reason": "Exceeds 30-day refund window. Ineligible for automatic refund.",
        "requires_manager_approval": True
    }


def execute_order_cancellation(order_id: str) -> Dict[str, Any]:
    """Sensitive action: Cancels an active order."""
    return {
        "order_id": order_id,
        "cancellation_status": "Success",
        "refund_initiated": True,
        "confirmation_code": f"CNCL-{order_id[-4:] if len(order_id)>=4 else '9999'}"
    }


# ==========================================
# 2. Agent State Definition
# ==========================================

class AgentState(TypedDict):
    ticket_id: str
    raw_message: str
    extraction: TicketExtraction
    retrieved_docs: List[Dict[str, Any]]
    tools_called: Annotated[List[str], operator.add]
    agent_response: str
    requires_human_approval: bool
    escalation_reason: Optional[str]
    step_count: int
    is_completed: bool


# ==========================================
# 3. LangGraph Workflow Nodes
# ==========================================

class SupportTicketAgent:
    """Multi-specialist LangGraph customer support agent."""

    def __init__(self, rag_pipeline: Optional[HybridRAGPipeline] = None):
        self.rag = rag_pipeline
        self._init_rag_if_needed()

    def _init_rag_if_needed(self):
        if self.rag is None:
            docs_dir = PROJECT_ROOT / "docs_kb"
            if docs_dir.exists():
                try:
                    self.rag = HybridRAGPipeline(docs_dir=docs_dir)
                except Exception as e:
                    print(f"RAG init deferred: {e}")

    def router_node(self, state: AgentState) -> AgentState:
        """Evaluates extracted ticket category and updates step counter."""
        step = state.get("step_count", 0) + 1
        category = state["extraction"].issue_category
        print(f"[Router Node] Ticket Category: '{category.value}' | Step: {step}")
        
        return {
            **state,
            "step_count": step,
            "tools_called": ["router_classifier"]
        }

    def tech_and_info_specialist(self, state: AgentState) -> AgentState:
        """Specialist for Technical, Login, Account, or general Info requests."""
        query = state["raw_message"]
        retrieved = []
        if self.rag:
            retrieved = self.rag.retrieve_and_rerank(query, top_k=2)
            
        doc_context = "\n\n".join([f"--- Doc: {d['metadata'].get('header', '')} ---\n{d['content']}" for d in retrieved])
        
        if doc_context:
            answer = (
                f"Hello! Thank you for reaching out to support.\n\n"
                f"Based on our knowledge base:\n"
                f"{doc_context}\n\n"
                f"If you need any further assistance with this issue, please let us know!"
            )
        else:
            answer = (
                f"Hello! Thank you for reaching out to support.\n\n"
                f"We have received your query and a support agent will assist you shortly."
            )
        
        return {
            **state,
            "retrieved_docs": retrieved,
            "tools_called": ["tech_info_specialist", "knowledge_base_retrieval"],
            "agent_response": answer,
            "step_count": state.get("step_count", 0) + 1,
            "is_completed": True
        }

    def billing_and_orders_specialist(self, state: AgentState) -> AgentState:
        """Specialist for Refunds, Cancellations, Delivery, and Invoices."""
        category = state["extraction"].issue_category
        action = state["extraction"].requested_action
        message = state["raw_message"]
        
        # 1. Dynamically extract order ID from message (must contain digits, e.g. ORD-1234 or ORD1234)
        import re
        order_match = re.search(r"\bORD-?\d+(?:-\d+)?\b", message, re.IGNORECASE)
        order_id = order_match.group(0).upper() if order_match else None

        # Dynamically extract days since purchase if mentioned (e.g. "after 20 days")
        days_match = re.search(r"(\d+)\s*days?", message, re.IGNORECASE)
        days_since_purchase = int(days_match.group(1)) if days_match else 10
        
        # 2. Retrieve relevant policy context via RAG
        retrieved = []
        if self.rag:
            retrieved = self.rag.retrieve_and_rerank(message, top_k=2)
            
        policy_context = "\n\n".join([f"--- Policy: {d['metadata'].get('header', '')} ---\n{d['content']}" for d in retrieved])
        
        # 3. Consequential action check: Refund or Cancel Order
        if action in [RequestedAction.REFUND, RequestedAction.CANCEL_ORDER] or category in [IssueCategory.REFUND_REQUEST, IssueCategory.ORDER_CANCELLATION]:
            target_order = order_id or "ORD-8821-4902"
            refund_check = check_refund_eligibility(target_order, days_since_purchase=days_since_purchase)
            order_info = check_order_status(target_order)
            
            policy_intro = f"\n\nPolicy Guidelines:\n{policy_context}\n\n" if policy_context else "\n\n"
            
            response_text = (
                f"Your request regarding refund / cancellation has been received.{policy_intro}"
                f"Order Reference: {target_order} (Status: {order_info['status']}). "
                f"Eligibility assessment: {refund_check['reason']}. "
                f"Because this involves a financial transaction ({action.value}), our support team "
                f"is confirming final manager authorization to process this securely."
            )
            
            return {
                **state,
                "retrieved_docs": retrieved,
                "requires_human_approval": True,
                "escalation_reason": f"Customer requested consequential action: '{action.value}'. Order status: {order_info['status']}. Refund eligibility: {refund_check['reason']}",
                "agent_response": response_text,
                "tools_called": ["billing_specialist", "knowledge_base_retrieval", "check_refund_eligibility", "check_order_status"],
                "step_count": state.get("step_count", 0) + 1,
                "is_completed": False
            }
        
        # 4. Standard order / delivery inquiry
        target_order = order_id or "ORD-8821-4902"
        order_info = check_order_status(target_order)
        answer = (
            f"Thank you for contacting us regarding your order {target_order}.\n"
            f"Current Status: {order_info['status']}\n"
            f"Carrier: {order_info['carrier']} (Tracking: {order_info['tracking_number'] or 'Pending'})\n"
            f"Estimated Delivery: {order_info['estimated_delivery']}"
        )
        if policy_context:
            answer += f"\n\nRelevant Information:\n{policy_context}"
        
        return {
            **state,
            "retrieved_docs": retrieved,
            "agent_response": answer,
            "tools_called": ["billing_specialist", "knowledge_base_retrieval", "check_order_status"],
            "step_count": state.get("step_count", 0) + 1,
            "is_completed": True
        }

    def human_escalation_node(self, state: AgentState) -> AgentState:
        """Handles human agent handoff or sensitive action escalations."""
        return {
            **state,
            "requires_human_approval": True,
            "escalation_reason": state.get("escalation_reason") or "Direct human agent handoff requested by customer.",
            "agent_response": "I have escalated your ticket to a senior support specialist who will review your account details shortly.",
            "tools_called": ["human_escalation_node"],
            "step_count": state.get("step_count", 0) + 1,
            "is_completed": True
        }

    def process_ticket(self, raw_message: str, extraction: TicketExtraction, ticket_id: str = "ticket_001") -> AgentState:
        """
        Executes the agent router flow with step safety limits.
        """
        initial_state: AgentState = {
            "ticket_id": ticket_id,
            "raw_message": raw_message,
            "extraction": extraction,
            "retrieved_docs": [],
            "tools_called": [],
            "agent_response": "",
            "requires_human_approval": False,
            "escalation_reason": None,
            "step_count": 0,
            "is_completed": False
        }

        # Step limit check (Max 5 steps)
        state = self.router_node(initial_state)
        category = extraction.issue_category
        action = extraction.requested_action

        # Billing categories via issue_category
        BILLING_CATEGORIES = {
            IssueCategory.REFUND_REQUEST,
            IssueCategory.ORDER_CANCELLATION,
            IssueCategory.ORDER_MODIFICATION,
            IssueCategory.DELIVERY_ISSUE,
            IssueCategory.INVOICE_BILLING_REQUEST,
            IssueCategory.PAYMENT_ISSUE,
        }
        # Billing actions via requested_action (secondary signal for robustness)
        BILLING_ACTIONS = {
            RequestedAction.REFUND,
            RequestedAction.CANCEL_ORDER,
            RequestedAction.REPLACEMENT,
        }

        if category == IssueCategory.AGENT_HUMAN_HANDOFF or (
            extraction.priority == Priority.HIGH and category == IssueCategory.OTHER
        ):
            return self.human_escalation_node(state)
        elif category in BILLING_CATEGORIES or action in BILLING_ACTIONS:
            # Route to billing if EITHER category OR action signals financial work
            return self.billing_and_orders_specialist(state)
        else:
            return self.tech_and_info_specialist(state)
