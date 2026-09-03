# Billing, Payments, and Refund Policies

This document outlines official payment handling, subscription management, and refund evaluation criteria.

## Refund Eligibility Policy
Refund eligibility depends on subscription tier and billing cadence:
- **Monthly Subscriptions**: Eligible for a full refund if requested within 14 calendar days of initial billing or auto-renewal, provided total API/compute consumption does not exceed 10% of monthly quotas.
- **Annual Subscriptions**: Eligible for a 100% refund within 30 calendar days of initial purchase. After 30 days, pro-rated refunds are calculated based on unused remaining full calendar months, minus standard non-discounted monthly rates for active months.
- **Hardware & Physical Merchandise**: Full refund within 30 days of physical delivery if returned in original condition with intact tamper seals.
- **Exceptions**: Setup fees, custom integration contracts, and consumed pay-as-you-go credits are non-refundable unless service downtime exceeded SLA commitments (99.9% monthly uptime).

## Subscription Cancellation and Proration
- Customers can cancel recurring plans at any time via **Billing > Manage Subscription**.
- Upon cancellation, access remains active until the end of the current paid billing cycle.
- Downgrading from Enterprise to Starter tier takes effect at the next renewal date, preserving accumulated data until downgrade.

## Failed Payments and Dunning Process
When an automated invoice charge fails:
1. **Day 1**: System sends an immediate payment failure notification and retries the card.
2. **Days 3 & 7**: Smart retry logic attempts billing at alternative bank settlement windows.
3. **Day 14**: Grace period ends; account transitions to restricted read-only mode.
4. **Day 30**: Account suspended and scheduled for data retention archival.

## Invoicing and Tax Identification
- Invoices are generated automatically on the 1st of each calendar month or on the renewal anniversary.
- VAT / GST numbers can be added under **Billing > Tax Information**. Validated tax IDs remove domestic sales tax for qualified reverse-charge transactions.
