#!/usr/bin/env python3
"""
XERV CRAYON - TestPyPI Upload Script
=====================================

This script builds and uploads Crayon to TestPyPI for testing.

Usage:
    python upload_testpypi.py

Prerequisites:
    1. pip install build twine
    2. Create ~/.pypirc with TestPyPI credentials OR
    3. Set TWINE_USERNAME and TWINE_PASSWORD environment variables

TestPyPI Credentials:
    - Register at https://test.pypi.org/account/register/
    - Create API token at https://test.pypi.org/manage/account/token/
    - Use __token__ as username and the token as password

After Upload, Install With:
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ xerv-crayon
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def log(msg: str, level: str = "INFO") -> None:
    """Print status message."""
    emoji = {"INFO": "📦", "WARN": "⚠️", "ERROR": "❌", "OK": "✅", "RUN": "🔧"}.get(level, "")
    print(f"[UPLOAD] {emoji} {msg}")


def check_prerequisites() -> bool:
    """Check that required tools are installed."""
    log("Checking prerequisites...")
    
    # Check for build
    try:
        import build
        log("'build' package found", "OK")
    except ImportError:
        log("'build' package not found. Install with: pip install build", "ERROR")
        return False
    
    # Check for twine
    try:
        import twine
        log("'twine' package found", "OK")
    except ImportError:
        log("'twine' package not found. Install with: pip install twine", "ERROR")
        return False
    
    return True


def clean_build_artifacts() -> None:
    """Remove old build artifacts."""
    log("Cleaning old build artifacts...", "RUN")
    
    dirs_to_clean = ["dist", "build", "*.egg-info"]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                log(f"Removed: {path}")
            elif path.is_file():
                path.unlink()
                log(f"Removed: {path}")
    
    # Also clean src/*.egg-info
    for path in Path("src").glob("*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)
            log(f"Removed: {path}")


def build_package() -> bool:
    """Build source distribution and wheel."""
    log("Building package...", "RUN")
    
    # Build using python -m build
    cmd = [sys.executable, "-m", "build"]
    log(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        log("Build failed!", "ERROR")
        return False
    
    # Verify artifacts exist
    dist_files = list(Path("dist").glob("*"))
    if not dist_files:
        log("No build artifacts found in dist/", "ERROR")
        return False
    
    log(f"Build successful! Created {len(dist_files)} artifacts:", "OK")
    for f in dist_files:
        log(f"  - {f.name}")
    
    return True


def upload_to_testpypi() -> bool:
    """Upload to TestPyPI using twine."""
    log("Uploading to TestPyPI...", "RUN")
    
    # Check for credentials
    username = os.environ.get("TWINE_USERNAME", "__token__")
    password = os.environ.get("TWINE_PASSWORD")
    
    if not password:
        # Check for pypirc
        pypirc = Path.home() / ".pypirc"
        if not pypirc.exists():
            log("No TWINE_PASSWORD set and no ~/.pypirc found", "WARN")
            log("You will be prompted for credentials.", "INFO")
    
    cmd = [
        sys.executable, "-m", "twine", "upload",
        "--repository", "testpypi",
        "dist/*"
    ]
    
    log(f"Running: {' '.join(cmd)}")
    
    # Run twine (will prompt for password if not set)
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        log("Upload failed!", "ERROR")
        return False
    
    log("Upload successful!", "OK")
    return True


def print_install_instructions() -> None:
    """Print instructions for installing from TestPyPI."""
    print("\n" + "=" * 70)
    print("📦 INSTALLATION INSTRUCTIONS")
    print("=" * 70)
    print("""
To install from TestPyPI, run:

    pip install --index-url https://test.pypi.org/simple/ \\
                --extra-index-url https://pypi.org/simple/ \\
                xerv-crayon

For Google Colab:

    !pip install --index-url https://test.pypi.org/simple/ \\
                 --extra-index-url https://pypi.org/simple/ \\
                 xerv-crayon

Then test with:

    from crayon import CrayonVocab, check_backends
    print(check_backends())
    
    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite")
    tokens = vocab.tokenize("Hello, world!")
    print(tokens)
""")


def main() -> int:
    """Main upload process."""
    print("=" * 70)
    print("🖍️  XERV CRAYON - TestPyPI Upload")
    print("=" * 70)
    print()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    log(f"Working directory: {project_root}")
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Clean old artifacts
    clean_build_artifacts()
    
    # Build
    if not build_package():
        return 1
    
    # Upload
    if not upload_to_testpypi():
        return 1
    
    # Print instructions
    print_install_instructions()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
