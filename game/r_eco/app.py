"""Entry point for Hugging Face Spaces (port 7860)."""
import os

# Download model from the Space repo before starting the server
def download_model():
    os.makedirs("model", exist_ok=True)
    target = "model/eco_latest.pkt"
    if os.path.exists(target):
        return
    try:
        from huggingface_hub import hf_hub_download
        space_id = os.environ.get("SPACE_ID", "NingcongChen/r-oko")
        path = hf_hub_download(
            repo_id=space_id,
            filename="model/eco_latest.pkt",
            repo_type="space",
            local_dir=".",
        )
        print(f"[app] Downloaded model to {path}")
    except Exception as e:
        print(f"[app] Could not download model: {e}")

download_model()

from server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
