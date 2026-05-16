#!/bin/zsh
# Run the voice agent using the project venv (Python 3.11 + MLX packages)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing .venv. Run once:  ./setup.sh"
  exit 1
fi

VER="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$VER" != "3.11" ]]; then
  echo "Warning: .venv is Python $VER; rebuild with ./setup.sh (project uses 3.11)."
fi

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi
exec "$VENV_PY" "$SCRIPT_DIR/voice_agent.py" "$@"
