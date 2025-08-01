# CRUSH.md - Codebase Guidelines

## Build/Test/Lint Commands

- **Install Dependencies**: `pip install -r requirements.txt` (and for `backend/requirements.txt`)
- **Run All Tests**: `pytest` or `./scripts/run_tests.py`
- **Run Single Test**: `pytest <path_to_test_file>::<test_function_name>`
- **Format Code**: `./autoformat.sh` (uses Black and Ruff)
- **Lint/Typecheck**: `ruff check .`

## Code Style Guidelines

- **General Python**: Adhere to PEP 8 for naming and formatting.
- **Imports**: Alphabetical and grouped (e.g., `isort` style).
- **Formatting**: Automated via `./autoformat.sh` (Black, Ruff). Consistent indentation (4 spaces).
- **Types**: Use Python type hints for all functions and variables.
- **Naming**: Descriptive function, variable, and class names (e.g., `snake_case` for functions/variables, `PascalCase` for classes).
- **Error Handling**: Implement comprehensive `try-except` blocks for external API calls. Provide clear error messages; log errors.
- **Comments**: Sparse, focus on *why* complex logic exists.

## Project-Specific Rules (from .cursorrules & CLAUDE.md)

- **Vertex AI**: ALWAYS use Vertex AI (`gemini-2.0-flash-exp` model), never pass `api_key` to Agent (use ADC).
- **Cloud Run**: ALWAYS target Cloud Run for deployment.
- **Authentication**: MANDATORY to use Application Default Credentials (ADC) and service accounts. Never hardcode credentials.
- **No Mock Data**: NEVER use mock, fallback, or placeholder data. Return real errors.
