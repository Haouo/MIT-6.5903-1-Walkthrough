#!/usr/bin/env bash
#
# extract_slides.sh — render selected pages of a lecture PDF to PNG figures.
#
# Renders chosen slide pages from a 6.5930/1 lecture PDF into
#   assets/<LECTURE>/<LECTURE>-p<NN>-<slug>.png
# at a fixed DPI, so the bilingual walkthroughs can embed them inline.
#
# Usage:
#   scripts/extract_slides.sh <pdf-path> <lecture-id> <page:slug> [<page:slug> ...]
#
# Example:
#   scripts/extract_slides.sh "Lecture/L01-Intro_and_Applications.pdf" L01 \
#     2:ai-ingredients 22:moore-dennard-slowdown 28:teaal-pyramid
#
# Produces:
#   assets/L01/L01-p02-ai-ingredients.png
#   assets/L01/L01-p22-moore-dennard-slowdown.png
#   assets/L01/L01-p28-teaal-pyramid.png
#
# Requirements: pdftoppm (poppler-utils).

set -euo pipefail

DPI="${DPI:-150}"   # override with: DPI=200 scripts/extract_slides.sh ...

if ! command -v pdftoppm >/dev/null 2>&1; then
  echo "error: pdftoppm not found (install poppler-utils)" >&2
  exit 1
fi

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <pdf-path> <lecture-id> <page:slug> [<page:slug> ...]" >&2
  exit 2
fi

PDF="$1"
LECTURE="$2"
shift 2

if [ ! -f "$PDF" ]; then
  echo "error: PDF not found: $PDF" >&2
  exit 1
fi

# Resolve repo root as the directory containing this script's parent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$REPO_ROOT/assets/$LECTURE"
mkdir -p "$OUT_DIR"

for spec in "$@"; do
  page="${spec%%:*}"
  slug="${spec#*:}"
  if [ "$page" = "$spec" ] || [ -z "$slug" ]; then
    echo "warn: skipping malformed spec '$spec' (expected page:slug)" >&2
    continue
  fi

  # Zero-pad page to 2 digits for stable filename sorting.
  printf -v padded "%02d" "$page"
  base="$OUT_DIR/${LECTURE}-p${padded}-${slug}"

  # pdftoppm appends -<page> when -f/-l span differs; for a single page with
  # matching -f/-l it appends nothing only when -singlefile is used.
  pdftoppm -png -r "$DPI" -f "$page" -l "$page" -singlefile "$PDF" "$base"
  echo "wrote ${base}.png"
done
