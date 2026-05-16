#!/bin/zsh
# Run the voice agent using the project venv (Python 3.11 + MLX packages)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/voice_agent.py" "$@"
