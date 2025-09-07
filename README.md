# 📄 PDF Page Reorderer

An intelligent PDF page reordering tool that uses advanced OCR technology to automatically detect page numbers and reorganize jumbled PDF pages into the correct sequential order.

## 🌟 Features

- **🤖 AI-Powered OCR**: Uses EasyOCR with multiple detection methods
- **🔍 Multi-Region Analysis**: Scans headers, footers, and page margins
- **⚡ GPU Acceleration**: Optimized for CUDA-enabled environments
- **🧠 Smart Disambiguation**: Resolves conflicts when multiple page numbers are detected
- **📊 Detailed Reports**: Generates comprehensive analysis reports
- **🎯 Multiple Formats**: Supports various page number formats (1, -1-, Page 1, etc.)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload the notebook version to Google Colab
2. Run all cells to install dependencies
3. Use the file upload widget to select your PDF
4. Download the reordered PDF

### Option 2: Kaggle (Recommended)
1. Create a new Kaggle notebook
2. Upload your PDF as a dataset
3. Copy the Kaggle-optimized code
4. Run and download results from `/kaggle/working/`

### Option 3: Local Streamlit App
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run main.py`
4. Open your browser to the displayed URL

## 📋 Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- CPU with 4+ cores

### Recommended for Best Performance
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 8GB+ (16GB for large PDFs)
- **Platform**: Google Colab Pro or Kaggle with GPU acceleration

## 🛠️ Installation

