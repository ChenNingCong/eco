#!/usr/bin/env bash
# Deploy Hearts to a Hugging Face Space.
#
# Usage:
#   ./deploy.sh              # space name defaults to "hearts"
#   ./deploy.sh my-hearts

set -euo pipefail

NAME="${1:-hearts}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Read username from logged-in credentials
USERNAME=$(hf auth whoami 2>/dev/null | head -1)
if [[ -z "$USERNAME" ]]; then
    echo "Error: not logged in. Run: hf login"
    exit 1
fi

SPACE="${USERNAME}/${NAME}"
echo "Deploying to ${SPACE} ..."

# ── Create space ──────────────────────────────────────────────────────────────
python3 -c "
from huggingface_hub import HfApi
HfApi().create_repo('${SPACE}', repo_type='space', space_sdk='docker', exist_ok=True)
"

# ── Upload source files ───────────────────────────────────────────────────────
for f in server.py app.py hearts_env.py obs_encoder.py ppo.py vec_env.py \
          requirements.txt README.md Dockerfile static/index.html; do
    if [[ -f "${DIR}/${f}" ]]; then
        echo "  uploading: ${f}"
        hf upload "${SPACE}" "${DIR}/${f}" "${f}" --repo-type space
    else
        echo "  skip (not found): ${f}"
    fi
done

# ── Newest model checkpoint ───────────────────────────────────────────────────
NEWEST=$(find "${DIR}/model" -name "*.pkt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -n "$NEWEST" ]]; then
    echo "  uploading model: $(basename "$NEWEST")"
    hf upload "${SPACE}" "$NEWEST" "model/$(basename "$NEWEST")" --repo-type space
else
    echo "  warning: no .pkt model files found, trained agent will not be available"
fi

echo ""
echo "Done! https://huggingface.co/spaces/${SPACE}"
