# 0) (One-time) install uv itself — pick ONE
# macOS (Homebrew)          :  brew install astral-sh/uv/uv
# Linux / macOS universal   :  curl -Ls https://astral.sh/uv/install.sh | sh
# With Rust toolchain       :  cargo install uv
# Windows (scoop)           :  scoop install uv


# 1) create an isolated Python env (defaults to .venv)
uv venv                 # ~2-3× faster than python -m venv  [oai_citation:0‡ubuntushell.com](https://ubuntushell.com/install-uv-python-package-manager/?utm_source=chatgpt.com)
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 2) install dependencies from lockfile for deterministic builds
# This ensures everyone gets the exact same versions
uv sync

# 3) sanity check
python - <<'PY'
from langchain_core.prompts import ChatPromptTemplate
print("LangChain core imported OK ✅")
PY

echo "Setup complete! Dependencies installed from lockfile for reproducible builds."
echo "To regenerate lockfile with latest compatible versions:"
echo "  uv pip install -e . && uv pip freeze > requirements.lock"