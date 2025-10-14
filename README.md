
# 🕵️‍♂️ AI-Based Dark Pattern Detector for Web Content

This project is a complete, end-to-end system that uses a fine-tuned **BERT** model to automatically detect and classify manipulative language (“dark patterns”) on websites.  

The final application is an interactive web analyzer built with **Streamlit**, capable of scraping any URL, analyzing its text content in real time, and generating a comprehensive report.

---

## 🚀 Key Features

| Feature                        | Description                                                                                             |
|----------------------------------|---------------------------------------------------------------------------------------------------------|
| 🧠 **End-to-End Analysis**       | Provide a URL and the system handles everything — from scraping to reporting.                           |
| 🤖 **High-Accuracy AI Model**   | Fine-tuned BERT model achieving **97.55%** accuracy.                                                    |
| 🌐 **Real-Time Web Scraping**  | Uses Selenium to extract text content from live web pages.                                              |
| 📊 **Interactive Dashboard**   | Streamlit interface that visualizes the analysis clearly.                                               |
| 📝 **Detailed Reporting**      | Includes a *Manipulation Score*, category-wise bar chart, and table of manipulative phrases.             |

---

## 🏛️ Project Architecture

The project follows a structured pipeline:

1. **Offline Phase**: Data preparation and BERT model fine-tuning.  
2. **Online Phase**: Real-time text extraction, classification, and visualization.



```markdown
![Project Architecture](https://github.com/keerthana777z/AI_BASED_dark-pattern-detector/blob/main/R.drawio.png)
```

---

## 📈 Model Performance

The fine-tuned BERT model was evaluated on a test set of **774 unseen samples** and achieved exceptional performance.

**Overall Accuracy:** 97.55%  
**Weighted F1-Score:** 97.00%

| Category            | Precision | Recall | F1-score | Support |
|----------------------|-----------|--------|----------|---------|
| Forced Action        | 0.00      | 0.00   | 0.00     | 2       |
| Misdirection         | 0.98      | 0.93   | 0.95     | 86      |
| Not Dark Pattern     | 0.96      | 0.98   | 0.97     | 236     |
| Obstruction          | 1.00      | 1.00   | 1.00     | 11      |
| Scarcity             | 0.99      | 1.00   | 0.99     | 219     |
| Sneaking             | 0.50      | 0.20   | 0.29     | 5       |
| Social Proof         | 1.00      | 1.00   | 1.00     | 125     |
| Urgency              | 0.97      | 0.98   | 0.97     | 90      |
| **Accuracy**         |           |        | **0.98** | 774     |
| **Macro avg**        | **0.80**  | **0.76** | **0.77** | 774   |
| **Weighted avg**     | **0.97**  | **0.98** | **0.97** | 774   |

> ⚠️ Lower scores for Forced Action and Sneaking are due to limited training samples in those categories.

---

## 🛠️ How to Run This Project

### 1️⃣ Setup the Environment

Create and activate a Python virtual environment:

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Prepare the Dataset

Combine raw data files into the final dataset:

```bash
python create_full_dataset.py
```

This will generate:

```
combined_dark_patterns_FULL.csv
```

---

### 3️⃣ Train the AI Model

Fine-tune the BERT model:

```bash
python dark_pattern_detector.py
```

This will create the folder:

```
final_dark_pattern_model/
```

containing the trained model.

---

### 4️⃣ Launch the Web Application

Run the Streamlit app:

```bash
streamlit run app.py
```

This will open the **interactive analyzer** in your browser.  
Enter any URL to analyze manipulative language patterns in real-time.

---

## 📂 Project Structure

```
AI_BASED_dark-pattern-detector/
│
├── app.py                            # Streamlit web interface
├── create_full_dataset.py            # Dataset preparation script
├── dark_pattern_detector.py          # Model training script
├── requirements.txt                  # Required Python libraries
├── final_dark_pattern_model/         # Trained BERT model files
├── combined_dark_patterns_FULL.csv   # Final dataset
├── README.md                         # Project documentation
└── ...
```

---

## 🧠 Tech Stack

- 🐍 **Python**  
- 🤖 **BERT** (Hugging Face Transformers)  
- 🕸️ **Selenium** for web scraping  
- 📊 **Streamlit** for the dashboard  
- 🧪 **Pandas**, **Scikit-learn**, **Matplotlib**

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit pull requests.

---

## 🪪 License

This project is licensed under the **MIT License**.

---

## ✨ Acknowledgements

- [Hugging Face](https://huggingface.co/) — for the Transformer models  
- [Streamlit](https://streamlit.io/) — for the easy-to-use UI framework  
- [Selenium](https://www.selenium.dev/) — for reliable web scraping

---
##by
https://github.com/keerthana777z
