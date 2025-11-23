#!/usr/bin/env python3
"""
NLP Model Downloader - Download required spacy models

SPECIFICATION:
- Download spacy en_core_web_sm model
- Verify installation
- Provide clear status messages

USAGE:
    python scripts/download_nlp_models.py

    # Or as module
    python -m scripts.download_nlp_models

WHY: Automated setup for NLP dependencies
"""
import sys
import subprocess
from pathlib import Path


def download_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """
    Download spacy model

    Args:
        model_name: Name of spacy model to download

    Returns:
        True if successful, False otherwise
    """
    print(f"🔽 Downloading spacy model: {model_name}")
    print("This may take a few minutes...")

    try:
        # Download model
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            capture_output=False
        )

        print(f"✅ Successfully downloaded {model_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {model_name}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return False


def verify_spacy_installation() -> bool:
    """Verify spacy is installed"""
    try:
        import spacy
        print(f"✅ spacy version: {spacy.__version__}")
        return True
    except ImportError:
        print("❌ spacy not installed. Run: pip install spacy", file=sys.stderr)
        return False


def verify_model_installation(model_name: str = "en_core_web_sm") -> bool:
    """
    Verify model is installed and loadable

    Args:
        model_name: Model to verify

    Returns:
        True if model loads successfully
    """
    try:
        import spacy
        nlp = spacy.load(model_name)

        # Test model
        doc = nlp("This is a test sentence.")

        print(f"✅ Model {model_name} loaded successfully")
        print(f"   Vocabulary size: {len(nlp.vocab)}")
        print(f"   Pipelines: {nlp.pipe_names}")

        return True

    except OSError:
        print(f"❌ Model {model_name} not found", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        return False


def main():
    """Main download and verification process"""
    print("=" * 60)
    print("NLP Model Download Script")
    print("=" * 60)
    print()

    # Step 1: Verify spacy installation
    print("Step 1: Verifying spacy installation...")
    if not verify_spacy_installation():
        print("\n⚠️  Please install spacy first: pip install spacy")
        return 1
    print()

    # Step 2: Download model
    print("Step 2: Downloading spacy model...")
    model_name = "en_core_web_sm"

    # Check if already installed
    if verify_model_installation(model_name):
        print(f"\n✨ Model {model_name} is already installed and working!")
        return 0

    # Download model
    if not download_spacy_model(model_name):
        print("\n❌ Failed to download model")
        return 1
    print()

    # Step 3: Verify installation
    print("Step 3: Verifying model installation...")
    if not verify_model_installation(model_name):
        print("\n❌ Model downloaded but failed to load")
        return 1

    print()
    print("=" * 60)
    print("✨ All NLP models installed successfully!")
    print("=" * 60)
    print()
    print("You can now use the AnalysisAgent with full NLP capabilities.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
