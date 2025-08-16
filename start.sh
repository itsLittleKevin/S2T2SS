#!/bin/bash
echo "🚀 S2T2SS - Activating Virtual Environment"
echo "========================================="

# Check if .venv exists
if [ -f ".venv/bin/activate" ]; then
    echo "✅ Found virtual environment at .venv"
    echo "🔄 Activating virtual environment..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
    echo "🚀 Starting S2T2SS Launcher..."
    python launcher.py
elif [ -f "venv/bin/activate" ]; then
    echo "✅ Found virtual environment at venv"
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
    echo "🚀 Starting S2T2SS Launcher..."
    python launcher.py
else
    echo "❌ No virtual environment found"
    echo ""
    echo "💡 To create a virtual environment:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "🔄 Starting launcher with system Python..."
    python launcher.py
fi
