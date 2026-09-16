#!/usr/bin/env bash
# Launcher for macOS and Linux. The Windows equivalent is launch_autocut.bat.
set -e
cd "$(dirname "$0")/auto_cut"

# WhisperX prints transcript text to stdout; force UTF-8 so non-ASCII output
# cannot kill a run part-way through.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

exec python3 app.py
