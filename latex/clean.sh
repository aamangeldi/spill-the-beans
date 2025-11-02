#!/bin/bash
# Clean LaTeX auxiliary files

cd "$(dirname "$0")"

echo "Cleaning LaTeX auxiliary files..."

rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Alternative: use latexmk to clean
# latexmk -c

echo "✓ Cleanup complete!"
