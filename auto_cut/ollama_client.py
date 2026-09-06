"""
Gets Ollama out of the way of the GPU, over stdlib urllib - no dependency.

This is all that remains of a larger client that drove a transcript-correction
pass. That feature is gone, but the VRAM handover is worth keeping on its own:
Ollama holds a model resident after you last used it, and on a 6GB card an 8B
model parked in VRAM leaves nothing for WhisperX large-v2, which then falls
back to the CPU and takes an order of magnitude longer. Unloading first is the
difference between a transcription that finishes and one you give up on.

Every call is best-effort. If Ollama is not installed or not running, that is
the normal case for most users and must never be an error.
"""

import json
import urllib.error
import urllib.request

BASE_URL = "http://localhost:11434"


def is_running(timeout=2.0):
    try:
        urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=timeout).read()
        return True
    except Exception:
        return False


def loaded_models(timeout=3.0):
    """
    Models Ollama currently has resident in VRAM.

    Worth checking before running WhisperX: on a 6GB card an 8B model parked in
    memory leaves nothing for large-v2, and the transcription simply fails or
    crawls on the CPU.
    """
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/ps", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    out = []
    for entry in payload.get("models", []):
        name = entry.get("name") or entry.get("model")
        if name:
            out.append({"name": name,
                        "size": entry.get("size", 0),
                        "size_vram": entry.get("size_vram", 0)})
    return out


def unload(model, timeout=30.0):
    """Asks Ollama to drop a model from VRAM now (keep_alive=0)."""
    body = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    request = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=timeout).read()
        return True
    except Exception:
        return False


def free_vram(log=None):
    """
    Unloads everything Ollama is holding. Returns the MB reclaimed.
    Called before transcription so the two don't fight over the same GPU.
    """
    resident = loaded_models()
    if not resident:
        return 0
    freed = 0
    for entry in resident:
        megabytes = entry["size_vram"] / (1024 * 1024)
        if log:
            log(f"Ollama is holding {entry['name']} ({megabytes:.0f} MB of VRAM) - "
                "unloading so WhisperX can use the GPU")
        if unload(entry["name"]):
            freed += megabytes
    return freed

