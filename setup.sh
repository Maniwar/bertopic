#!/bin/bash
# Multi-Bot Forum Application - Setup Script
# ===========================================

set -e  # Exit on error

echo "=========================================="
echo "🤖 Multi-Bot Setup Script"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Found Python $PYTHON_VERSION${NC}"
echo ""

# Check pip
echo "📦 Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip not found!${NC}"
    echo "   Installing pip..."
    python3 -m ensurepip --upgrade
fi
echo -e "${GREEN}✅ pip is available${NC}"
echo ""

# Install Python dependencies
echo "📥 Installing Python dependencies..."
echo "   (This may take a few minutes)"
pip3 install -r requirements_multibot.txt --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python packages installed${NC}"
else
    echo -e "${RED}❌ Failed to install Python packages${NC}"
    exit 1
fi
echo ""

# Check for Tesseract OCR
echo "👁️  Checking Tesseract OCR (for vision features)..."
if command -v tesseract &> /dev/null; then
    TESSERACT_VERSION=$(tesseract --version | head -n1)
    echo -e "${GREEN}✅ $TESSERACT_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Tesseract not found${NC}"
    echo "   Vision features will not work without Tesseract"
    echo ""
    echo "   To install:"
    echo "   - Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "   - macOS: brew install tesseract"
    echo "   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi
echo ""

# Set up .env file
echo "🔐 Setting up environment file..."
if [ ! -f .env ]; then
    echo "   Copying .env.example to .env"
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file${NC}"
    echo -e "${YELLOW}   ⚠️  Please edit .env and add your Reddit credentials!${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi
echo ""

# Check for Chrome/Chromium
echo "🌐 Checking for Chrome/Chromium (for browser automation)..."
CHROME_FOUND=0
if command -v google-chrome &> /dev/null; then
    echo -e "${GREEN}✅ Google Chrome found${NC}"
    CHROME_FOUND=1
elif command -v chromium &> /dev/null; then
    echo -e "${GREEN}✅ Chromium found${NC}"
    CHROME_FOUND=1
elif command -v chromium-browser &> /dev/null; then
    echo -e "${GREEN}✅ Chromium Browser found${NC}"
    CHROME_FOUND=1
else
    echo -e "${YELLOW}⚠️  Chrome/Chromium not found${NC}"
    echo "   Browser automation will not work"
    echo ""
    echo "   To install:"
    echo "   - Ubuntu/Debian: sudo apt-get install chromium-browser"
    echo "   - macOS: brew install --cask google-chrome"
    echo "   - Windows: Download from https://www.google.com/chrome/"
fi
echo ""

# Database initialization
echo "💾 Initializing database..."
python3 -c "from multi_bot_forum_app import Database; Database()" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Database initialization skipped${NC}"
fi
echo ""

# Test imports
echo "🧪 Testing imports..."
python3 -c "import multi_bot_forum_app; import reddit_browser_integration" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Core modules load successfully${NC}"
else
    echo -e "${RED}❌ Error importing modules${NC}"
    echo "   Check requirements installation"
fi
echo ""

# Summary
echo "=========================================="
echo "📋 Setup Summary"
echo "=========================================="
echo ""

if [ -f .env ]; then
    # Check if credentials are set
    if grep -q "your_client_id_here" .env; then
        echo -e "${YELLOW}⚠️  Action Required:${NC}"
        echo "   1. Edit .env file and add your Reddit credentials"
        echo "   2. Get credentials from: https://www.reddit.com/prefs/apps"
        echo ""
    else
        echo -e "${GREEN}✅ Environment configured${NC}"
    fi
fi

echo "🎯 Ready to run:"
echo ""
echo "   Demo Mode (no credentials needed):"
echo "   $ python3 multi_bot_forum_app.py"
echo ""
echo "   Reddit API Mode:"
echo "   $ python3 run_api_bot.py"
echo ""
echo "   Vision Mode (experimental):"
echo "   $ python3 run_vision_bot.py"
echo ""
echo "   Full Bot Manager:"
echo "   $ python3 run_full_bot_manager.py"
echo ""

echo "📚 Documentation:"
echo "   • QUICKSTART.md - Getting started"
echo "   • README_MULTIBOT.md - Full docs"
echo "   • VISION_AUTOMATION_GUIDE.md - Vision features"
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
