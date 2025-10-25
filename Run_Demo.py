#!/bin/bash
# Quick start script for R-CNN Suite

echo "🚀 Starting R-CNN Suite Setup..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run glyph loader to verify installations
echo "🔧 Verifying environment..."
python glyph_loader.py

# Create sample image and run demo
echo "🎯 Running demo..."
python demo.py

echo "✅ Setup complete!"
echo "💡 Usage examples:"
echo "   python rcnn_suite.py --mode faster --image your_image.jpg --out result.jpg"
echo "   python rcnn_suite.py --mode rcnn --image your_image.jpg --out rcnn_result.jpg"
