# LaTeX Setup

## Install MacTeX

```bash
# Install MacTeX via Homebrew
brew install --cask mactex

# Run the installer (will open GUI)
open /opt/homebrew/Caskroom/mactex/2025.0308/mactex-*.pkg
```

Follow the installer prompts and enter your password when asked.

## Test Compilation

After installation completes, restart your terminal and test:

```bash
cd LatexTemplate
./compile.sh
```

The PDF should be generated successfully.
