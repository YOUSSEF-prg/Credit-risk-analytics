"""Check the local environment: Python, packages, GPU, and the Ollama model.

Usage (from the project root, inside the conda environment):
    python scripts/check_environment.py [--model qwen2.5-coder:7b] [--ollama-url http://localhost:11434]

This script checks the setup only. It does not evaluate answer quality; that is
done by the RAG evaluation benchmark in Phase 6.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
import time

PACKAGES = [
    "pandas", "numpy", "scipy", "sklearn", "xgboost", "imblearn", "shap", "mlflow",
    "streamlit", "plotly", "matplotlib", "python_calamine", "yaml", "joblib", "requests",
]  # fmt: skip


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  [ok]   {name:16s} {getattr(module, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            ok = False
            print(f"  [FAIL] {name:16s} {type(exc).__name__}: {exc}")
    return ok


def check_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("  nvidia-smi not found: GPU unavailable, CPU fallback will be used.")
        return
    query = "--query-gpu=name,memory.total,memory.used,driver_version"
    result = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True
    )
    print("  " + (result.stdout.strip() or result.stderr.strip()))


def check_ollama(url: str, model: str) -> bool:
    import requests

    try:
        tags = requests.get(f"{url}/api/tags", timeout=5).json()
    except requests.RequestException as exc:
        print(f"  [FAIL] Ollama not reachable at {url}: {exc}")
        print("         Start it with `ollama serve` (or open the Ollama app).")
        return False
    names = [m["name"] for m in tags.get("models", [])]
    print(f"  Installed models: {names}")
    if model not in names:
        print(f"  [FAIL] {model} not installed. Run: ollama pull {model}")
        return False

    start = time.perf_counter()
    response = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "prompt": "Reply with exactly the word: ready",
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        },
        timeout=300,
    ).json()
    elapsed = time.perf_counter() - start
    print(f"  Response: {response.get('response', '').strip()!r}  ({elapsed:.1f}s incl. load)")
    if response.get("eval_count") and response.get("eval_duration"):
        rate = response["eval_count"] / (response["eval_duration"] / 1e9)
        print(f"  Generation speed: {rate:.1f} tokens/s")
    try:
        ps = requests.get(f"{url}/api/ps", timeout=5).json()
        for m in ps.get("models", []):
            vram, size = m.get("size_vram", 0), m.get("size", 0)
            share = vram / size if size else 0
            print(
                f"  Loaded {m['name']}: total {size / 1e9:.2f} GB, in VRAM {vram / 1e9:.2f} GB "
                f"({share:.0%} on GPU)"
            )
    except requests.RequestException:
        pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    print(f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        print("  [WARN] Project targets Python 3.12")
    print("Packages:")
    packages_ok = check_packages()
    print("GPU:")
    check_gpu()
    print("Ollama:")
    ollama_ok = check_ollama(args.ollama_url, args.model)
    print("\nSummary:", "packages OK" if packages_ok else "package problems",
          "|", "Ollama OK" if ollama_ok else "Ollama not ready")  # fmt: skip


if __name__ == "__main__":
    main()
