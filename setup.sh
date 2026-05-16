#!/bin/zsh
# One-time / rebuild: Python 3.11 venv + dependencies + spaCy model for Kokoro TTS
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init - zsh)" 2>/dev/null || true
  if [[ -f .python-version ]]; then
  pyenv install -s "$(cat .python-version)" 2>/dev/null || true
  fi
fi

PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "No python3 found. Install Python 3.11 (pyenv install 3.11) and retry."
  exit 1
fi

VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$VER" != "3.11" ]]; then
  echo "Warning: using Python $VER; this project targets 3.11 (see .python-version)."
fi

rm -rf .venv
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Installing spaCy English model for Kokoro (en_core_web_sm) ..."
.venv/bin/python -m spacy download en_core_web_sm

echo ""
echo "Setup complete. Run:  ./run.sh"
