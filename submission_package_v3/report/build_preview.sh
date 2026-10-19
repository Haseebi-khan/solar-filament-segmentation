#!/usr/bin/env bash
# Local preview of the report: compiles main.tex with the stand-ins in local_standins/ (the organisers' Overleaf
# project provides the real preamble.tex and main.bib) and writes main_preview.pdf next to this script.
# The PDF to hand in is compiled in the organisers' Overleaf project (see ../TASKS.md, step 6).
#   submission_package_v3/report/build_preview.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(mktemp -d)
cp -r "$HERE/main.tex" "$HERE/ours.bib" "$HERE/figures" "$HERE"/local_standins/* "$B/"
( cd "$B" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null ) \
  || { echo "LaTeX failed, see $B/main.log"; exit 1; }
cp "$B/main.pdf" "$HERE/main_preview.pdf"
echo "overfull boxes: $(grep -c "Overfull" "$B/main.log" || true)"
rm -rf "$B"
echo "wrote $HERE/main_preview.pdf"
