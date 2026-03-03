# Contributing to Flimmer

Thank you for your interest in contributing.

## Security First

Before contributing, please install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This installs gitleaks secret scanning. **Every commit is scanned for accidentally included secrets, API keys, and tokens.** If the hook blocks your commit, it caught something — check the output and remove the sensitive data before committing.

**Never commit:**
- API keys, tokens, or passwords
- `.env` files (use `.env.example` with placeholder values)
- Private keys or certificates
- Cloud provider credentials
- Personal access tokens

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Install pre-commit hooks (`pre-commit install`)
4. Make your changes
5. Run the test suite (`pytest tests/`)
6. Commit with a descriptive message
7. Push to your fork and open a Pull Request

## What We're Looking For

- **Bug fixes** — Especially in video processing, caption generation, and dataset validation.
- **New caption backends** — Support for additional VLM APIs or local models.
- **Control signal implementations** — Depth, edge, pose, or other conditioning pipelines.
- **Documentation** — Examples, guides, tutorials for video LoRA training workflows.
- **Test coverage** — More edge cases, integration tests, fixture datasets.

## What We're Not Looking For

- Changes that break compatibility with existing dataset formats.
- Features that add heavy runtime dependencies without clear justification.
- Model weights or large binary files — use Git LFS or external hosting.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful. Be constructive.

## Questions?

Open an [issue](https://github.com/alvdansen/flimmer-trainer/issues) or reach out to the maintainers.
