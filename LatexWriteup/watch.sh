#!/bin/bash
# LaTeX Auto-Compilation Script (Watch Mode)

# Add MacTeX binaries to PATH
export PATH="/Library/TeX/texbin:$PATH"

cd "$(dirname "$0")"

echo "Starting LaTeX watch mode..."
echo "The document will recompile automatically when you save changes."
echo "Press Ctrl+C to stop."
echo ""

latexmk -pdf -pvc -interaction=nonstopmode main.tex
