# Security, Privacy, and Compliance Policies

This document details security incident handling, GDPR/CCPA data privacy compliance, and credential rotation protocols.

## GDPR / CCPA Data Rights Requests
- **Right of Access & Portability**: Users can request a full export of account data in JSON/CSV archive under **Settings > Privacy > Export Data**. The archive generation completes within 24 hours.
- **Right to Erasure (Account Deletion)**: Account deletion requests trigger a 7-day soft-delete grace period, followed by irreversible cryptographical scrubbing across primary and backup databases within 30 days.

## API Key and Secret Rotation
- Compromised or exposed API keys must be revoked immediately via the Developer Dashboard.
- **Zero-Downtime Key Migration**: Generate a secondary key, update your server environment variables, verify successful requests, and finally delete the compromised primary key.

## Reporting Security Vulnerabilities
- Report vulnerabilities to `security@example.com` or through our official Bug Bounty program.
- Please allow 48 hours for initial triage before public disclosure.
