"""
Tests for LangGraph Agent: Routing, Tools, and Human-in-the-loop safety.
"""

from src.agent import SupportTicketAgent, check_refund_eligibility, check_order_status
from src.schema import (
    IssueCategory,
    Priority,
    CustomerSentiment,
    RequestedAction,
    TicketExtraction
)


def test_order_status_tool():
    status = check_order_status("ORD-1234")
    assert status["status"] == "In Transit"
    assert status["carrier"] == "FedEx"


def test_refund_eligibility_tool():
    within_window = check_refund_eligibility("ORD-1234", days_since_purchase=7)
    assert within_window["eligible"] is True
    assert within_window["requires_manager_approval"] is False

    outside_window = check_refund_eligibility("ORD-1234", days_since_purchase=45)
    assert outside_window["eligible"] is False
    assert outside_window["requires_manager_approval"] is True


def test_agent_routes_to_human_escalation():
    agent = SupportTicketAgent(rag_pipeline=None)
    extraction = TicketExtraction(
        issue_category=IssueCategory.AGENT_HUMAN_HANDOFF,
        priority=Priority.HIGH,
        customer_sentiment=CustomerSentiment.NEGATIVE,
        requested_action=RequestedAction.OTHER,
        product_or_service="General"
    )
    state = agent.process_ticket("I need to speak to a real person right now!", extraction)
    assert state["requires_human_approval"] is True
    assert "human_escalation_node" in state["tools_called"]


def test_agent_pauses_on_refund_request():
    agent = SupportTicketAgent(rag_pipeline=None)
    extraction = TicketExtraction(
        issue_category=IssueCategory.REFUND_REQUEST,
        priority=Priority.HIGH,
        customer_sentiment=CustomerSentiment.NEUTRAL,
        requested_action=RequestedAction.REFUND,
        product_or_service="Product"
    )
    state = agent.process_ticket("I want a refund for my order.", extraction)
    # Sensitive action must require approval
    assert state["requires_human_approval"] is True
    assert "check_refund_eligibility" in state["tools_called"]
