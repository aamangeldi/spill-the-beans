#!/bin/bash
# LaTeX Compilation Script

# Add MacTeX binaries to PATH
export PATH="/Library/TeX/texbin:$PATH"

cd "$(dirname "$0")"

echo "Compiling LaTeX document..."

# Run pdflatex first pass
pdflatex -interaction=nonstopmode main.tex

# Run bibtex for bibliography
bibtex main

# Run pdflatex second pass (incorporate bibliography)
pdflatex -interaction=nonstopmode main.tex

# Run pdflatex third pass (resolve cross-references)
pdflatex -interaction=nonstopmode main.tex

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful! Output: main.pdf"
    echo ""
    echo "Cleaning auxiliary files..."
    rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
    echo "✓ Cleanup complete!"
else
    echo ""
    echo "✗ Compilation failed. Check the errors above."
    exit 1
fi
