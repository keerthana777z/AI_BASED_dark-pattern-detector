import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os

# ==============================================================================
# MUST BE FIRST: Configure Streamlit page
# ==============================================================================
st.set_page_config(page_title="Dark Pattern Detector", page_icon="🕵️", layout="wide")

# Force CPU usage - disable MPS and CUDA
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure CPU usage
if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
    torch.backends.mps.is_available = lambda: False

# ==============================================================================
#  Load the fine-tuned model and tokenizer
# ==============================================================================
# Use st.cache_resource to load the model only once and save memory
@st.cache_resource
def load_model():
    # This path should point to the folder where your final model was saved
    model_path = "./final_dark_pattern_model"
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer, None
    except OSError as e:
        # If the model isn't found, return error message instead of displaying it here
        error_msg = f"Error: Model not found at path: {model_path}. Please make sure you have trained the model by running the training script and that the 'final_dark_pattern_model' folder exists in the same directory as this app."
        return None, None, error_msg

# Load the model and tokenizer
model, tokenizer, error_msg = load_model()

# Set up the device (will use CPU as configured in training)
device = torch.device("cpu")
if model:
    model.to(device)

# ==============================================================================
#  Text cleaning and prediction functions
# ==============================================================================
# This text cleaning function MUST be identical to the one used for training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)       # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)    # Remove punctuation and numbers
    tokens = text.split()                   # Split text into words
    return ' '.join(tokens)

def predict_dark_pattern(text):
    # If the model failed to load, return an error message
    if not model or not tokenizer:
        return "Model is not available."

    # Set the model to evaluation mode
    model.eval()

    # Clean the input text
    cleaned_text = clean_text(text)

    # Tokenize the text for BERT
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class ID and the human-readable label
    prediction_id = torch.argmax(outputs.logits, dim=-1).item()
    predicted_label = model.config.id2label[prediction_id]

    return predicted_label

# ==============================================================================
#  Streamlit App User Interface
# ==============================================================================

st.title("🕵️ Dark Pattern Detector")
st.markdown("This tool uses a BERT-based AI model to analyze text and identify manipulative design patterns (dark patterns) commonly found on websites and apps.")

# Display error message if model failed to load
if error_msg:
    st.error(error_msg)
    st.stop()

st.write("") # Add a little space

# Create two columns for a cleaner layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Enter text from a website or app to analyze:",
        height=150,
        placeholder="e.g., 'Only 2 items left in stock, order now!' or 'This offer expires in 5 minutes!'"
    )

    if st.button("Analyze Text", type="primary"):
        if text_input and model:
            with st.spinner('AI is thinking...'):
                result = predict_dark_pattern(text_input)
                st.success(f"**Detected Pattern:** `{result}`")
        elif not model:
            st.error("Cannot analyze text because the model is not loaded.")
        else:
            st.warning("Please enter some text to analyze.")

with col2:
    st.subheader("What are Dark Patterns?")
    st.markdown("""
    Dark patterns are tricks used in user interfaces to make you do things you didn't mean to, like buying something or signing up for a service. This tool helps identify them based on the language used.

    **Common Types:**
    - **Scarcity:** Suggesting an item is in limited supply.
    - **Urgency:** Using time-based pressure (e.g., countdowns).
    - **Social Proof:** Implying high demand from others.
    - **Misdirection:** Guiding you toward an unintended choice.
    """)

# Add some example texts for users to try
st.markdown("---")
st.subheader("Try These Examples:")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("🔥 Urgency Example"):
        st.session_state.example_text = "Hurry! This offer expires in 5 minutes!"

with example_col2:
    if st.button("📦 Scarcity Example"):
        st.session_state.example_text = "Only 2 items left in stock!"

with example_col3:
    if st.button("👥 Social Proof Example"):
        st.session_state.example_text = "500 people bought this today!"

# Display example text if button was clicked
if hasattr(st.session_state, 'example_text'):
    st.text_area("Example text:", value=st.session_state.example_text, height=50, key="example_display")
    if st.button("Analyze Example", type="secondary"):
        if model:
            with st.spinner('Analyzing example...'):
                result = predict_dark_pattern(st.session_state.example_text)
                st.success(f"**Example Result:** `{result}`")
