
## Local setup
1. Install uv `curl -LsSf https://astral.sh/uv/install.sh | sh` (linux / mac) or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (windows)
2. Run `uv sync --dev`
3. Install pre-commit hook `pre-commit install`
4. (optional) It is highly recommended to install ruff in your vscode for consistent formatting (set line length limit to 100). Pre-commit will make sure that all files you push to repo will follow correct formatting.