#!/usr/bin/env bash
#
# check_walkthroughs.sh — validate the bilingual lecture walkthroughs.
#
# Runs three structural checks (no PDF tooling needed, so it works in CI):
#   1. EN/ZH filename parity — every English walkthrough has a Chinese twin.
#   2. Image links resolve — every embedded ../../assets/... path exists.
#   3. Template sections — each file contains the required 8 sections
#      (language-appropriate headings for en/ vs zh/).
#
# Exits non-zero on the first category that fails, after listing all problems.
# Usage: scripts/check_walkthroughs.sh   (run from repo root)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

EN_DIR="walkthroughs/en"
ZH_DIR="walkthroughs/zh"
fail=0

# Required section markers per language (substrings, matched anywhere in file).
EN_SECTIONS=("TL;DR" "Learning Objectives" "Key Terms" "Takeaways" "Connections" "Appendix")
ZH_SECTIONS=("TL;DR" "學習目標" "關鍵詞彙" "重點回顧" "連結" "附錄")

echo "==> 1. EN/ZH filename parity"
en_list="$(find "$EN_DIR" -maxdepth 1 -name '*.md' -printf '%f\n' | sort)"
zh_list="$(find "$ZH_DIR" -maxdepth 1 -name '*.md' -printf '%f\n' | sort)"
if [ "$en_list" != "$zh_list" ]; then
  echo "   FAIL: en/ and zh/ filenames differ:"
  diff <(printf '%s\n' "$en_list") <(printf '%s\n' "$zh_list") | sed 's/^/     /' || true
  fail=1
else
  echo "   OK: $(printf '%s\n' "$en_list" | grep -c .) lectures present in both languages"
fi

echo "==> 2. Embedded image links resolve"
img_total=0
img_broken=0
while IFS= read -r f; do
  dir="$(dirname "$f")"
  while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    img_total=$((img_total + 1))
    if [ ! -f "$dir/$rel" ]; then
      echo "   BROKEN: $f -> $rel"
      img_broken=$((img_broken + 1))
    fi
  done < <(grep -oE '!\[[^]]*\]\([^)]+\)' "$f" | sed -E 's/.*\(//; s/\)$//')
done < <(find "$EN_DIR" "$ZH_DIR" -name '*.md' | sort)
if [ "$img_broken" -ne 0 ]; then
  echo "   FAIL: $img_broken of $img_total image links are broken"
  fail=1
else
  echo "   OK: all $img_total image links resolve"
fi

echo "==> 3. Template sections present"
check_sections() {
  local f="$1"; shift
  local missing=()
  local sec
  for sec in "$@"; do
    grep -qF "$sec" "$f" || missing+=("$sec")
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    echo "   FAIL: $f missing: ${missing[*]}"
    return 1
  fi
  return 0
}
section_fail=0
while IFS= read -r f; do
  check_sections "$f" "${EN_SECTIONS[@]}" || section_fail=1
done < <(find "$EN_DIR" -name '*.md' | sort)
while IFS= read -r f; do
  check_sections "$f" "${ZH_SECTIONS[@]}" || section_fail=1
done < <(find "$ZH_DIR" -name '*.md' | sort)
if [ "$section_fail" -ne 0 ]; then
  fail=1
else
  echo "   OK: every walkthrough has all required sections"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAILED"
  exit 1
fi
echo "RESULT: PASSED"
