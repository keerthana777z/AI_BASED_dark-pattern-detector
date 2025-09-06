# 🕵️ Dark Pattern Detector

An AI-powered tool that uses BERT (Bidirectional Encoder Representations from Transformers) to detect and classify dark patterns in text. Dark patterns are manipulative design techniques used in websites and apps to trick users into unintended actions.
hi this is keertha

## 🎯 Features

- **Multi-class Classification**: Detects various types of dark patterns including :
  - **Scarcity**: "Only 2 items left in stock!"
  - **Urgency**: "This offer expires in 5 minutes!"
  - **Social Proof**: "500 people bought this today!"
  - **Misdirection**: "No, I don't want to save money"


- **High Accuracy**: Achieves 99%+ accuracy on validation data
- **CPU Optimized**: Runs entirely on CPU without GPU requirements
- **Interactive Web App**: User-friendly Streamlit interface
- **Real-time Analysis**: Instant dark pattern detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/keerthana777z/dark-pattern-detector.git
cd dark-pattern-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

#### Option 2: Train Your Own Model
```bash
python dark_pattern_detector.py
```

## 📊 Model Performance

- **Accuracy**: 99.07%
- **F1 Score**: 99.04%
- **Precision**: 99.13%
- **Recall**: 99.07%

## 🗂️ Project Structure

```
dark-pattern-detector/
├── app.py                          # Streamlit web application
├── dark_pattern_detector.py        # Model training script
├── final_dark_pattern_model/       # Trained BERT model
├── combined_dark_patterns_FULL.csv # Training dataset
├── requirements.txt                # Python dependencies
├── visualize.py                    # Data visualization tools
├── prepare_data.py                 # Data preprocessing utilities
└── README.md                       # This file
```

## 🔧 Technical Details

- **Model**: BERT-base-uncased fine-tuned for sequence classification
- **Framework**: PyTorch + Transformers (Hugging Face)
- **Training**: 3 epochs with CPU optimization
- **Text Processing**: Lowercase, HTML tag removal, punctuation cleaning
- **Interface**: Streamlit for web deployment

## 📈 Dataset

The model is trained on a comprehensive dataset of dark pattern examples covering multiple categories. The dataset includes real-world examples from various websites and applications.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Hugging Face Transformers library
- Streamlit for the web interface
- The research community working on dark pattern detection
