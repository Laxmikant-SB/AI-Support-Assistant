"""
Shared schema definitions for ticket classification and extraction.
Uses Pydantic enums to enforce strict evaluation boundaries.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    PAYMENT_ISSUE = "Payment Issue"
    DELIVERY_ISSUE = "Delivery Issue"
    LOGIN_ISSUE = "Login Issue"
    ACCOUNT_ISSUE = "Account Issue"
    REFUND_REQUEST = "Refund Request"
    ORDER_CANCELLATION = "Order Cancellation"
    ORDER_MODIFICATION = "Order Modification"
    INVOICE_BILLING_REQUEST = "Invoice/Billing Request"
    PRODUCT_DEFECT = "Product Defect"
    TECHNICAL_ISSUE = "Technical Issue"
    AGENT_HUMAN_HANDOFF = "Agent/Human Handoff"
    OTHER = "Other"


class Priority(str, Enum):
    LOW = "Low"          # General inquiries, feedback, or non-urgent queries
    MEDIUM = "Medium"    # Minor inconveniences or standard support requests
    HIGH = "High"        # Blockers: locked accounts, payment failures, urgent requests


class CustomerSentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class RequestedAction(str, Enum):
    REFUND = "Refund"
    REPLACEMENT = "Replacement"
    CANCEL_ORDER = "Cancel Order"
    TECHNICAL_SUPPORT = "Technical Support"
    INFORMATION_REQUEST = "Information Request"
    ACCOUNT_RECOVERY = "Account Recovery"
    OTHER = "Other"


class TicketExtraction(BaseModel):
    """Structured extraction output for customer support tickets."""
    issue_category: IssueCategory = Field(
        description="Primary topic or department category for the ticket."
    )
    priority: Priority = Field(
        description="Urgency level based on customer impact."
    )
    customer_sentiment: CustomerSentiment = Field(
        description="Customer emotional tone in the message."
    )
    requested_action: RequestedAction = Field(
        description="Explicit or implicit action the customer wants taken."
    )
    product_or_service: str = Field(
        default="General",
        description="Specific product, feature, or service mentioned, or 'General'."
    )


class TicketRecord(BaseModel):
    """Represents a labeled ticket in the dataset."""
    id: str
    text: str
    extraction: TicketExtraction
    metadata: Optional[dict] = None
