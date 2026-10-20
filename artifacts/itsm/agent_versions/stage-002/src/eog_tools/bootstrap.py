"""Install the authenticated SDK wheel currently served by EvoHarnessBench."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

DEFAULT_SERVICE_URL = "https://educator-marrow-cultural.ngrok-free.dev"
MINIMUM_SDK_VERSION = (0, 19, 2)


def require_sdk_version() -> str:
    """Reject an installed SDK that cannot select cohort and harness independently."""
    from importlib.metadata import version

    installed = version("simple-agentic-evals")
    try:
        parts = tuple(int(part) for part in installed.split(".")[:3])
    except ValueError as exc:
        raise RuntimeError(f"unrecognized simple-agentic-evals version: {installed}") from exc
    if len(parts) != 3 or parts < MINIMUM_SDK_VERSION:
        raise RuntimeError(f"simple-agentic-evals >=0.19.2 required; found {installed}")
    return installed


def bootstrap_sdk(project: Path) -> dict[str, str]:
    key = os.environ.get("EVAL_SERVICE_API_KEY")
    if not key:
        raise RuntimeError("EVAL_SERVICE_API_KEY is required in this terminal")
    base_url = os.environ.get("EVAL_SERVICE_URL", DEFAULT_SERVICE_URL).rstrip("/")
    headers = {
        "Authorization": f"Bearer {key}",
        "ngrok-skip-browser-warning": "true",
    }

    def read(url: str) -> bytes:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    manifest = json.loads(read(base_url + "/sdk"))
    filename = manifest["filename"]
    if Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError("service returned an invalid SDK wheel filename")
    served_version = filename.split("-")[1]
    try:
        served_parts = tuple(int(part) for part in served_version.split(".")[:3])
    except ValueError as exc:
        raise RuntimeError(f"unrecognized SDK wheel version: {served_version}") from exc
    if len(served_parts) != 3 or served_parts < MINIMUM_SDK_VERSION:
        raise RuntimeError(f"service SDK >=0.19.2 required; offered {served_version}")
    wheel = read(urljoin(base_url + "/", manifest["path"]))
    cache = project / ".cache" / "sdk"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / filename
    path.write_bytes(wheel)
    info = {
        "filename": filename,
        "sha256": hashlib.sha256(wheel).hexdigest(),
        "service_url": base_url,
    }
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", str(path)],
        check=True,
    )
    installed_version = require_sdk_version()
    if served_version != installed_version:
        raise RuntimeError("installed SDK version does not match the served wheel")
    info["version"] = installed_version
    (cache / "sdk-info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
