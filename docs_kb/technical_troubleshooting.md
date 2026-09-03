# Technical Troubleshooting and API Integration Guide

This guide covers common error codes, API rate limiting, desktop app synchronization, and connectivity diagnostics.

## API Error Codes and Status Handling
- **HTTP 401 Unauthorized**: Missing or invalid Bearer API token. Check header format `Authorization: Bearer <TOKEN>`.
- **HTTP 403 Forbidden**: Token lacks necessary scoped permissions (e.g., attempting a write action with a read-only scoped key).
- **HTTP 429 Too Many Requests**: Rate limit exceeded. Standard tier permits 60 requests/minute; Enterprise permits 1,200 requests/minute. Always implement exponential backoff with jitter using the `Retry-After` header value.
- **HTTP 500 / 502 / 504 Internal Errors**: Temporary gateway or worker overload. Verify real-time status at `https://status.example.com` before reporting an outage.

## Desktop Client Synchronization Issues
If the desktop client fails to synchronize local workspace state:
1. Confirm client version is $\ge 2.4.0$ (**Help > About**). Outdated clients are rejected by the sync gateway.
2. Check local proxy settings. Corporate SSL inspection proxies may block WebSocket endpoints (`wss://sync.example.com`).
3. Clear the local cache directory:
   - Windows: `%APPDATA%\ExampleApp\cache`
   - macOS: `~/Library/Caches/ExampleApp`
4. Relaunch the client and perform a Force Full Sync via `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (macOS).

## Webhook Delivery Failures
- Webhooks time out after 5000ms if your receiving endpoint does not return a 2xx HTTP status.
- Failed webhooks are retried 5 times over an exponential schedule (1m, 5m, 15m, 1h, 6h).
- Ensure your endpoint verifies HMAC SHA256 payload signatures using the secret key configured in developer settings.
