# Order Processing, Shipping, and Delivery Guide

This document details order workflows, shipping timelines, address modifications, and lost package investigations.

## Order Status and Tracking
- **Order Placement**: Once confirmed, orders receive a unique 10-digit Order ID (`ORD-XXXX-XXXX`).
- **Processing Stage (0-24 Hours)**: The fulfillment warehouse verifies stock allocation and packages items.
- **In-Transit Stage**: Tracking numbers (FedEx, UPS, DHL) are emailed to the customer once scanned at the carrier origin facility.
- **Estimated Delivery Windows**: Standard domestic shipping takes 3-5 business days; expedited shipping takes 1-2 business days; international delivery takes 7-14 business days depending on customs clearance.

## Order Modifications and Cancellations
- **Cancellation Window**: Orders can be self-canceled within 60 minutes of placement directly from the Order History page.
- **Warehouse Lockout**: Once an order status transitions to `Processing / Fulfillment Queue`, automated cancellation is locked to prevent inventory desynchronization. In this state, an agent must request warehouse hold.
- **Address Changes**: Shipping address changes are only permitted while the order is in `Pending Verification` status. Once labeled for shipping, rerouting must be requested directly through carrier delivery management tools (e.g., FedEx Delivery Manager).

## Damaged or Missing Shipments
- **Damaged Packages**: Customers should submit clear photos of damaged outer packaging and the internal defect within 7 days of delivery. A direct replacement unit will be dispatched under priority courier.
- **Lost in Transit**: A package is declared officially lost if tracking shows no scan updates for 7 consecutive business days for domestic shipments, or 14 days for international shipments. A replacement or full refund is automatically issued upon claim filing.
