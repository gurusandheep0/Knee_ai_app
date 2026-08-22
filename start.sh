#!/usr/bin/env sh
set -eu

for interpreter in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$interpreter" >/dev/null 2>&1; then
        exec "$interpreter" "$(dirname "$0")/start.py"
    fi
done

echo "Python 3.10 or newer is required." >&2
exit 2
