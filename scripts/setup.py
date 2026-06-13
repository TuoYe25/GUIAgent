"""
Project setup script.

Usage:
    python scripts/setup.py
    python scripts/setup.py --force  # Force reinstall
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SYSTEM_PACKAGES = {
    "Windows": ["node", "npm"],
    "Darwin": ["node", "npm"],
    "Linux": ["node", "npm"],
}


def check_system_deps() -> bool:
    """Check if required system packages are installed."""
    import platform as pf

    system = pf.system()
    needed = REQUIRED_SYSTEM_PACKAGES.get(system, [])

    missing = []
    for dep in needed:
        try:
            subprocess.run([dep, "--version"], capture_output=True, check=True)
            print(f"  [OK] {dep}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(dep)
            print(f"  [MISSING] {dep}")

    return len(missing) == 0


def install_python_deps() -> None:
    """Install Python dependencies."""
    print("\nInstalling Python dependencies...")

    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print("  No requirements.txt found, generating...")
        # Generate basic requirements
        deps = [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "transformers>=4.40.0",
            "accelerate>=0.28.0",
            "vllm>=0.4.0",
            "Pillow>=10.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "loguru>=0.7.0",
            "websockets>=12.0",
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "groq>=0.5.0",
            "together>=1.0.0",
            "playwright>=1.40.0",
            "selenium>=4.15.0",
            "pyppeteer>=2.0.0",
            "beautifulsoup4>=4.12.0",
            "tqdm>=4.66.0",
        ]
        req_file.write_text("\n".join(deps), encoding="utf-8")

    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)

    # Install Playwright browsers
    print("\nInstalling Playwright browsers...")
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


def install_electron_deps() -> None:
    """Install Electron demo dependencies."""
    electron_dir = PROJECT_ROOT / "electron_demo"
    if not electron_dir.exists():
        print("  electron_demo/ not found, skipping")
        return

    print("\nInstalling Electron dependencies...")
    if (electron_dir / "package.json").exists():
        subprocess.run(["npm", "install"], cwd=str(electron_dir), check=True)
    else:
        # Generate package.json
        pkg = {
            "name": "edge-gui-agent-demo",
            "version": "0.1.0",
            "description": "Edge GUI Agent Electron Demo",
            "main": "main.js",
            "scripts": {
                "start": "electron .",
                "dev": "electron . --dev"
            },
            "devDependencies": {
                "electron": "^30.0.0"
            }
        }
        import json
        (electron_dir / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")
        subprocess.run(["npm", "install"], cwd=str(electron_dir), check=True)


def check_gpu() -> None:
    """Check GPU availability."""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  [OK] Apple Metal (MPS) available")
        else:
            print("  [WARN] No GPU detected. GUI model inference will be slow on CPU.")
    except ImportError:
        print("  [WARN] PyTorch not installed yet. Run setup to install.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge GUI Agent setup")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--skip-electron", action="store_true", help="Skip Electron deps")
    args = parser.parse_args()

    print("=" * 50)
    print("Edge GUI Agent — Setup")
    print("=" * 50)

    # 1. System deps
    print("\n[1/4] Checking system dependencies...")
    check_system_deps()

    # 2. GPU check
    check_gpu()

    # 3. Python deps
    print("\n[2/4] Installing Python dependencies...")
    install_python_deps()

    # 4. Electron deps
    if not args.skip_electron:
        print("\n[3/4] Installing Electron dependencies...")
        install_electron_deps()

    # 5. Create config
    print("\n[4/4] Creating local config...")
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        env_file.write_text(
            "# Edge GUI Agent Configuration\n"
            "GUI_AGENT_MODEL_PATH=auto\n"
            "GUI_AGENT_BACKEND=vllm\n"
            "GUI_AGENT_PORT=8765\n"
            "EDGE_VLM_PORT=8766\n"
            "LOG_LEVEL=INFO\n"
            "\n"
            "# Remote API keys (optional, for hybrid mode)\n"
            "# OPENAI_API_KEY=\n"
            "# ANTHROPIC_API_KEY=\n"
            "# GROQ_API_KEY=\n",
            encoding="utf-8",
        )
        print(f"  Created {env_file}")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("  1. Deploy GUI model:    python scripts/deploy_edge_vlm.py")
    print("  2. Run benchmarks:      python scripts/run_benchmarks.py --all")
    print("  3. Launch Electron demo: cd electron_demo && npm start")
    print("=" * 50)


if __name__ == "__main__":
    main()
