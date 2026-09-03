"""
Tests for Schema and Enum validations.
"""

import pytest
from pydantic import ValidationError
from src.schema import (
    IssueCategory,
    Priority,
    CustomerSentiment,
    RequestedAction,
    TicketExtraction,
)


def test_valid_ticket_extraction():
    data = {
        "issue_category": "Payment Issue",
        "priority": "High",
        "customer_sentiment": "Negative",
        "requested_action": "Refund",
        "product_or_service": "Annual Pro Subscription"
    }
    obj = TicketExtraction(**data)
    assert obj.issue_category == IssueCategory.PAYMENT_ISSUE
    assert obj.priority == Priority.HIGH
    assert obj.customer_sentiment == CustomerSentiment.NEGATIVE
    assert obj.requested_action == RequestedAction.REFUND
    assert obj.product_or_service == "Annual Pro Subscription"


def test_invalid_enum_raises_validation_error():
    data = {
        "issue_category": "InvalidCategory123",
        "priority": "High",
        "customer_sentiment": "Negative",
        "requested_action": "Refund",
        "product_or_service": "General"
    }
    with pytest.raises(ValidationError):
        TicketExtraction(**data)


def test_default_product_or_service():
    data = {
        "issue_category": "Technical Issue",
        "priority": "Medium",
        "customer_sentiment": "Neutral",
        "requested_action": "Technical Support"
    }
    obj = TicketExtraction(**data)
    assert obj.product_or_service == "General"
