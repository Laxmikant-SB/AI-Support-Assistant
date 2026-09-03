# Account Access and Login Troubleshooting Guide

This document provides official procedures and policy guidelines for user account authentication, password management, and account recovery.

## Password Reset and Recovery
Users experiencing login difficulties should first attempt standard self-service password recovery:
1. Navigate to the sign-in portal and select **Forgot Password**.
2. Enter the verified primary email address associated with the account.
3. Check your inbox for an automated reset link containing a cryptographically signed one-time token. Reset links expire after 15 minutes.
4. If the reset email is not received within 5 minutes, inspect spam/junk filters or verify if an enterprise email security gateway (such as Mimecast or Proofpoint) quarantined the message.

## Two-Factor Authentication (2FA) Reset
When a user loses access to their registered authenticator app or hardware security key:
- **Backup Recovery Codes**: Users can sign in using one of the 10 single-use recovery codes generated during initial 2FA setup.
- **Manual 2FA Reset Request**: If recovery codes are unavailable, an account security review is required. The customer must provide proof of domain ownership or the last 4 digits of the payment method on file.
- **Cooling-Off Period**: Once an administrator approves a manual 2FA reset, the account enters a 24-hour security hold during which sensitive actions (e.g., changing billing details, transferring workspace ownership) are temporarily restricted.

## Account Lockout Policy
Accounts are automatically locked after 5 consecutive failed authentication attempts within a rolling 10-minute window.
- **Temporary Lockout**: Automatically clears after 30 minutes without administrative intervention.
- **Permanent Lockout**: Triggered if brute-force patterns or suspicious geolocation velocity (e.g., logins from two distinct continents within 1 hour) are detected. Requires customer identity verification before reactivation.

## Workspace Membership and Role Delegation
- Workspace owners can invite team members and assign roles: Admin, Member, and Billing Only.
- To transfer primary workspace ownership, the current owner must initiate the transfer from **Settings > General > Workspace Ownership**. If the current owner has departed the company, an official written request from a company officer (C-level or IT Director) is required.
