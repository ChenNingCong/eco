"""Entry point for Hugging Face Spaces (port 7860)."""
import os
from server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
