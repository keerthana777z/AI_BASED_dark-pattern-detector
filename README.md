# 🕵️‍♂️ AI-Based Dark Pattern Detector for Web Content

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

This project is a complete, end-to-end AI-powered web analyzer that uses a fine-tuned **BERT** model to automatically detect and classify manipulative language — commonly known as **dark patterns** — on websites.

The final application is an interactive web tool built with **Streamlit**, capable of scraping any URL, analyzing its text content in real time, and generating a comprehensive manipulation report.

---

## 😈 What Are Dark Patterns?

A **dark pattern** is a trick embedded in a website or app’s design to influence users to do something they didn’t intend — like buying a product, sharing personal information, or staying subscribed.

### Common Examples:

* **Hidden Costs:** A product or room looks cheap — but extra fees appear only at the final checkout step.
* **Tricky Questions:** “Untick this box if you *do not* want to receive marketing emails.”
* **Fake Urgency:** “Deal ends in 5 minutes!” ⏳ — even if the timer resets.
* **Fake Scarcity:** “Only 2 left in stock!” — though plenty might remain.
* **Hard to Cancel:** Easy to sign up, but canceling involves endless forms, calls, or confusing menus.

---

## 🦸 Our Solution

Detecting such patterns manually is time-consuming and often ineffective. This system automates the entire process using AI and real-time web scraping.

### How It Works (Step-by-Step)

1.  **🌐 Input Website:** The user provides a website URL.
2.  **🤖 Robot Reader (Selenium):** A crawler visits the website and extracts all visible text — buttons, labels, popups, small print, etc.
3.  **🧠 AI Detective (BERT):** The fine-tuned BERT model analyzes the text and classifies it into categories like: `Urgency`, `Scarcity`, `Obstruction`, `Misdirection`, `Social Proof`, `Sneaking`, `Forced Action`, or `Not Dark Pattern`.
4.  **📊 Report Generation (Streamlit):** The system produces a report including:
    * A **Manipulation Score (%)**
    * A category-wise **bar chart**
    * A **table of flagged sentences** showing suspicious wording.

---

## 🚀 Key Features

| Feature                   | Description                                                                                       |
| :------------------------ | :------------------------------------------------------------------------------------------------ |
| 🧠 **End-to-End Analysis** | Provide a URL, and the system handles everything — from scraping to reporting.                      |
| 🤖 **High-Accuracy AI Model** | Fine-tuned BERT model achieving **97.55%** accuracy on the validation set.                        |
| 🌐 **Real-Time Web Scraping** | Uses Selenium to automatically extract text content from live web pages.                            |
| 📊 **Interactive Dashboard** | Streamlit interface that visualizes the analysis clearly.                                         |
| 📝 **Detailed Reporting** | Includes a *Manipulation Score*, category-wise bar chart, and table of detected manipulative phrases. |

---

## 🏛️ Project Architecture

The project follows a structured pipeline:

1.  **Offline Phase**: Data preparation (combining sources, cleaning) and BERT model fine-tuning.
2.  **Online Phase**: Real-time text extraction via Selenium, classification using the trained model, and visualization via Streamlit.

![Architecture Diagram](https://github.com/keerthana777z/AI_BASED_dark-pattern-detector/raw/main/R.drawio.png)

---

## 📈 Model Performance

The fine-tuned BERT model was evaluated on a test set of **774 unseen samples** and achieved exceptional performance:

-   **Overall Accuracy:** `97.55%`
-   **Weighted F1-Score:** `97.00%`

| Category           | Precision | Recall | F1-score | Support |
| :----------------- | :-------- | :----- | :------- | :------ |
| Forced Action      | 0.00      | 0.00   | 0.00     | 2       |
| Misdirection       | 0.98      | 0.93   | 0.95     | 86      |
| Not Dark Pattern   | 0.96      | 0.98   | 0.97     | 236     |
| Obstruction        | 1.00      | 1.00   | 1.00     | 11      |
| Scarcity           | 0.99      | 1.00   | 0.99     | 219     |
| Sneaking           | 0.50      | 0.20   | 0.29     | 5       |
| Social Proof       | 1.00      | 1.00   | 1.00     | 125     |
| Urgency            | 0.97      | 0.98   | 0.97     | 90      |
|                    |           |        |          |         |
| **Weighted Avg** | **0.97** | **0.98** | **0.97** | **774** |

> ⚠️ **Note:** Lower scores for `Forced Action` and `Sneaking` are due to the extremely limited number of training samples (2 and 5, respectively) in the source datasets.

---

## 🛠️ How to Run This Project

### 1️⃣ Setup the Environment

First, create and activate a Python virtual environment:

```bash
# Create the virtual environment
python -m venv venv

# Activate it (on Mac/Linux)
source venv/bin/activate
# On Windows, use: venv\Scripts\activate

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


## 🪪 License

This project is licensed under the **MIT License**.

---

## ✨ Acknowledgements

- [Hugging Face](https://huggingface.co/) — for the Transformer models  
- [Streamlit](https://streamlit.io/) — for the easy-to-use UI framework  
- [Selenium](https://www.selenium.dev/) — for reliable web scraping

---
## Done by
https://github.com/keerthana777z

https://github.com/Abishek7952
