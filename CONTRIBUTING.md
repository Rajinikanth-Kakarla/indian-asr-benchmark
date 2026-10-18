
# Contributing

Thank you for your interest in contributing to Indian ASR Benchmark!

## Getting started

1. Fork this repository on GitHub.
2. Clone your fork to your computer.
3. Create and activate a Python virtual environment.
4. Install the dependencies from `requirements.txt`.
5. Run the existing tests before making changes.

## Making changes

- Open an issue before starting a large feature.
- Create a separate branch for each change.
- Keep changes focused and document new behavior.
- Add or update tests when changing evaluation logic.
- Never replace reference transcripts with model predictions.
- Document text-normalization rules and evaluation settings.
- Do not upload private recordings, secrets, or datasets without redistribution permission.

## Testing

Run the test suite with:

    python -m pytest -v

## Pull requests

1. Explain the problem and the proposed solution.
2. Link the related issue, if one exists.
3. Describe the tests you ran.
4. Include relevant metric changes or example outputs.
5. Wait for review before merging.

## Scope

This project evaluates speech recognition systems. Contributions should aim to improve reproducibility, language coverage, evaluation quality, and usability.
