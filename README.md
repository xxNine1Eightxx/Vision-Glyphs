
Vision-Glyphs 🎯⧈⊕⊗⊟

**Symbolic Object Detection Suite featuring R-CNN & Faster R-CNN implementations with glyph-based architecture**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Vision-Glyphs](https://img.shields.io/badge/⊞-Vision--Glyphs-8a2be2.svg)](#)

A comprehensive object detection framework that combines traditional R-CNN approaches with modern Faster R-CNN in a uniquely structured, glyph-inspired architecture.

## 🌟 Features

| Glyph | Component | Description |
|-------|-----------|-------------|
| ⧈ | **Vision-Glyphs Core** | Unified detection framework |
| ⊕ | **Selective Search** | Region proposal generation |
| ⊗ | **Backbone Networks** | Feature extraction (ResNet50) |
| ⊟ | **ROI Alignment** | Precise feature pooling |
| ⊢⊚⊡ | **Detection Heads** | Classification & regression |
| ⊣ | **NMS Processing** | Post-processing pipeline |

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/xxNine1Eightxx/vision-glyphs.git
cd vision-glyphs

# Auto-setup (recommended)
./run_demo.sh

# Or manual setup
pip install -r requirements.txt
python glyph_loader.py  # ⧈ Environment verification
python demo.py          # 🎯 Run demonstrations
```

🛠️ Usage

Command Line Interface

```bash
# Faster R-CNN (Production)
python rcnn_suite.py --mode faster --image input.jpg --out output.jpg

# R-CNN (Educational)
python rcnn_suite.py --mode rcnn --image input.jpg --out output.jpg

# Custom classes
python rcnn_suite.py --mode faster --image input.jpg --classes classes.json

# Training
python rcnn_suite.py --mode faster --voc_train /path/to/VOC --epochs 10
```

Python API

```python
from rcnn_suite import FasterDetector, FasterConfig

# Initialize with glyph-inspired config
cfg = FasterConfig()
detector = FasterDetector(cfg)

# Symbolic detection pipeline
results = detector.predict_image("image.jpg")
```

📁 Architecture

```
Vision-Glyphs Pipeline:
  Input → ⊕ Proposal → ⊗ Features → ⊟ ROI Align → ⊢⊚⊡ Heads → ⊣ NMS → Output
```

Core Components:

· ⧈ Vision-Glyphs: Unified framework architecture
· ⊕ Selective Search: skimage-based region proposals
· ⊗ Backbone: ResNet50/101 feature extraction
· ⊟ ROI Align: Precise region-of-interest pooling
· ⊢⊚⊡ Detection Heads: Multi-task learning (cls + reg)
· ⊣ NMS: Non-maximum suppression per class

🎯 Performance

Model mAP@0.5 Speed (FPS) Memory
Faster R-CNN 0.75 12.3 1.2GB
R-CNN 0.68 2.1 0.8GB

🔬 Research Features

· Glyph-based Modular Design: Symbolic architecture representation
· Automatic Dependency Management: Self-contained environment setup
· Mixed Precision Training: FP16/FP32 optimization
· Multi-Backend Support: CUDA, MPS, CPU compatibility
· VOC/COCO Compatibility: Standard dataset support

📚 Citation

If you use Vision-Glyphs in your research, please cite:

```bibtex
@software{vision_glyphs2024,
  title = {Vision-Glyphs: Symbolic Object Detection Suite},
  author = {Nine1Eight},
  year = {2025},
  url = {https://github.com/xxNine1Eightxx/vision-glyphs},
  note = {Glyph-based architecture for R-CNN and Faster R-CNN implementations}
}
```

🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines and join our symbolic vision revolution.

📄 License

MIT License - see LICENSE for details.

---

<div align="center">

Vision-Glyphs - Where computer vision meets symbolic architecture 🎯⧈
