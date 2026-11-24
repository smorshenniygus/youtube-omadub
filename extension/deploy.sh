#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-}"
if [ -z "$DST" ]; then
  echo "Usage: $0 <target-chrome-extension-dir>" >&2
  exit 1
fi

echo "Running extension unit tests..."
(cd "$SRC" && node --test tests/*.test.cjs)

mkdir -p "$DST/icons" "$DST/fonts"

# Sync only runtime extension files; preserve target local-config.json (holds local auth token).
for f in "$SRC"/manifest.json "$SRC"/content.css "$SRC"/options.html "$SRC"/*.js; do
  [ -f "$f" ] || continue
  install -m 0644 "$f" "$DST/$(basename "$f")"
done

for f in "$SRC"/icons/*.png; do
  [ -f "$f" ] || continue
  install -m 0644 "$f" "$DST/icons/$(basename "$f")"
done

for f in "$SRC"/fonts/*; do
  [ -f "$f" ] || continue
  install -m 0644 "$f" "$DST/fonts/$(basename "$f")"
done

if [ ! -s "$DST/local-config.json" ] && [ -s "$SRC/local-config.json" ]; then
  install -m 0600 "$SRC/local-config.json" "$DST/local-config.json"
fi

# Remove non-runtime bloat from installed extension directory.
rm -rf "$DST/design" "$DST/tests" "$DST/README.md" "$DST/.gitignore" "$DST/icons/icon.svg"

echo "Deployed clean runtime extension to $DST ($(du -sh "$DST" | cut -f1))."
