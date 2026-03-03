# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Flimmer, please report it responsibly.

**Do NOT:**
- Open a public GitHub issue for security vulnerabilities
- Post details in Discussions or social media before coordinated disclosure

**Do:**
- Email us or use GitHub's private vulnerability reporting
- Include steps to reproduce if possible
- Allow up to 72 hours for initial response

We will acknowledge receipt within 72 hours and work with you on coordinated disclosure.

## Scope

This policy covers the Flimmer training toolkit and all files in this repository.

## Security Design

Flimmer is a video LoRA training toolkit. It:

- Processes local video files and images for model training
- Calls external APIs for captioning (Gemini, Replicate, OpenAI-compatible) — API keys are provided by the user
- Does not store or transmit user credentials beyond what's needed for configured API backends

The primary security concerns are:
- **Accidental secret exposure** when committing API keys or tokens. Flimmer's `.gitignore` and pre-commit hooks help prevent this.
- **Untrusted model weights** — only load checkpoints from trusted sources.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Pre-commit Secret Scanning

We strongly recommend all users install gitleaks as a pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

This catches secrets before they ever enter git history.
