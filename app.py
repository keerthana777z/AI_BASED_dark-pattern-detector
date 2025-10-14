import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import re
import os

# ==============================================================================
# Page Configuration 
# ==============================================================================
st.set_page_config(page_title="Dark Pattern Web Analyzer", page_icon="🕵️", layout="wide")

# ==============================================================================
# Environment Setup for Reproducibility
# ==============================================================================
# Force CPU usage as the model was trained on CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
    torch.backends.mps.is_available = lambda: False
device = torch.device("cpu")

# ==============================================================================
# Cached Functions for Performance
# ==============================================================================

# Cache the Selenium WebDriver setup to avoid reinstalling it on every run
@st.cache_resource
def get_driver():
    print("Setting up Selenium WebDriver...")
    options = Options()
    options.add_argument("--headless")  # Run browser in the background
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")
    options.add_argument("--disable-javascript")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    try:
        # Try to get a fresh ChromeDriver installation
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print("WebDriver setup complete.")
        return driver
    except Exception as e:
        print(f"Failed to initialize Selenium WebDriver: {e}")
        st.warning("⚠️ Web scraping feature is disabled due to ChromeDriver issues.")
        st.info("💡 You can still use the text input feature to analyze dark patterns!")
        return None

# Cache the trained model and tokenizer to avoid reloading them
@st.cache_resource
def load_model():
    print("Loading fine-tuned BERT model...")
    model_path = "./final_dark_pattern_model"
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model, tokenizer, None
    except OSError:
        error_msg = f"Error: Model not found at '{model_path}'. Please ensure the model has been trained and the folder exists."
        return None, None, error_msg

# ==============================================================================
# Core Functions: Scraper, Cleaning, and Prediction
# ==============================================================================

def scrape_page_text(url, driver):
    """Uses Selenium to scrape all relevant text snippets from a given URL."""
    if not driver:
        return []
    
    print(f"Scraping URL: {url}")
    try:
        driver.get(url)
        # Find all common text-containing elements
        tags_to_extract = ['p', 'h1', 'h2', 'h3', 'button', 'a', 'span', 'li']
        snippets = []
        for tag in tags_to_extract:
            elements = driver.find_elements(By.TAG_NAME, tag)
            for el in elements:
                text = el.text
                if text and len(text.strip()) > 3:  # Only add non-empty, meaningful snippets
                    snippets.append(text.strip())
        print(f"Scraped {len(snippets)} text snippets.")
        return list(set(snippets)) # Return unique snippets
    except Exception as e:
        st.error(f"An error occurred while scraping the URL: {e}")
        return []

def clean_text(text):
    """Cleans text for BERT model prediction."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

def predict_dark_patterns(snippets, model, tokenizer):
    """Runs predictions on a list of text snippets."""
    results = []
    for text in snippets:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue

        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        prediction_id = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()
        
        predicted_label = model.config.id2label[prediction_id]
        
        results.append({
            "text": text,
            "prediction": predicted_label,
            "confidence": f"{confidence:.2%}"
        })
    return results

# ==============================================================================
# Streamlit App UI
# ==============================================================================

st.title("🕵️ Dark Pattern Web Analyzer")
st.markdown("Enter a website URL to automatically scrape its text and analyze it for manipulative language using a fine-tuned BERT model.")

# Load resources and handle potential errors
driver = get_driver()
model, tokenizer, error_msg = load_model()

if error_msg:
    st.error(error_msg)
    st.stop()

# Continue even if driver is None - we'll handle it in the UI

# URL input from the user
url_input = st.text_input("Enter website URL to analyze:", placeholder="https://www.example.com")

if st.button("Analyze Website", type="primary"):
    if not url_input:
        st.warning("Please enter a URL.")
    elif not driver:
        st.error("⚠️ Web scraping is currently unavailable due to ChromeDriver issues.")
        st.info("💡 Try using the text input feature below instead!")
    else:
        with st.spinner(f"Scraping and analyzing {url_input}... This might take a minute."):
            # 1. Scrape
            scraped_snippets = scrape_page_text(url_input, driver)
            
            if scraped_snippets:
                # 2. Predict
                predictions = predict_dark_patterns(scraped_snippets, model, tokenizer)
                df = pd.DataFrame(predictions)
                
                # 3. Analyze and Summarize Results
                dark_patterns_df = df[df['prediction'] != 'Not Dark Pattern']
                manipulative_count = len(dark_patterns_df)
                total_snippets = len(df)
                manipulation_score = (manipulative_count / total_snippets * 100) if total_snippets > 0 else 0
                
                # 4. Display Report
                st.success("Analysis complete!")
                
                st.subheader("📊 Summary Report")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Manipulation Score", f"{manipulation_score:.1f}%")
                    st.write("This score represents the percentage of detected text snippets that were classified as a dark pattern.")
                
                with col2:
                    st.metric("Detected Manipulative Snippets", f"{manipulative_count}")
                    st.write(f"Out of {total_snippets} unique text snippets analyzed on the page.")

                # Display category distribution
                if not dark_patterns_df.empty:
                    st.subheader("Detected Dark Pattern Categories")
                    category_counts = dark_patterns_df['prediction'].value_counts()
                    st.bar_chart(category_counts)
                    
                    st.subheader("📜 Detected Manipulative Text")
                    st.dataframe(dark_patterns_df[['text', 'prediction', 'confidence']])
                else:
                    st.info("No dark patterns were detected on this page.")
            else:
                st.warning("Could not extract any text from the provided URL. The page might be empty, protected, or a Single Page Application that is difficult to scrape.")

# ==============================================================================
# Alternative Text Input Section
# ==============================================================================
st.markdown("---")
st.subheader("📝 Alternative: Direct Text Analysis")
st.markdown("If web scraping isn't working, you can paste text directly below for analysis:")

text_input = st.text_area(
    "Enter text to analyze for dark patterns:",
    height=150,
    placeholder="e.g., 'Only 2 items left in stock! Hurry, this offer expires in 5 minutes!'"
)

if st.button("Analyze Text", type="secondary"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Split text into sentences for analysis
            sentences = [s.strip() for s in text_input.split('.') if s.strip()]
            if not sentences:
                sentences = [text_input]

            predictions = predict_dark_patterns(sentences, model, tokenizer)
            df = pd.DataFrame(predictions)

            # Analyze results
            dark_patterns_df = df[df['prediction'] != 'Not Dark Pattern']
            manipulative_count = len(dark_patterns_df)
            total_snippets = len(df)
            manipulation_score = (manipulative_count / total_snippets * 100) if total_snippets > 0 else 0

            # Display results
            st.success("Analysis complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Manipulation Score", f"{manipulation_score:.1f}%")
            with col2:
                st.metric("Dark Patterns Found", f"{manipulative_count}")

            if not dark_patterns_df.empty:
                st.subheader("🚨 Detected Dark Patterns")
                for _, row in dark_patterns_df.iterrows():
                    st.warning(f"**{row['prediction']}** ({row['confidence']}): {row['text']}")
            else:
                st.success("✅ No dark patterns detected in this text!")

