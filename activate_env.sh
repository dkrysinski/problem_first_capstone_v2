#!/bin/bash

# Activation script for RegTech AI virtual environment
echo "🚀 Activating RegTech AI virtual environment..."

# Activate the virtual environment
source regtech_env/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python version: $(python --version)"
echo "📍 Virtual env location: $(which python)"
echo ""
echo "Available commands:"
echo "  streamlit run src/streamlit_app/app.py  - Start the web application"
echo "  python -c \"from src.langchain_modules.agent import AIAgent; agent = AIAgent()\"  - Test agent initialization"
echo ""
echo "To deactivate: deactivate"