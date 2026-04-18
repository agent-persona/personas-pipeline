# Discord Auth Notes

## Purpose

Keep a safe reference for how Discord auth looks in DevTools without storing live credentials.

## Redacted header example

```text
authorization: <discord-user-token-redacted>
content-type: application/json
origin: https://discord.com
referer: https://discord.com/channels/@me/<channel-id>
user-agent: Mozilla/5.0 ...
```

## Rules

- never commit a real Discord token
- never commit cookies copied from the browser
- if you need a local note with a live token, keep it outside the repo or in Doppler
- if a token lands in git by accident, rotate it first, then scrub the history before push
