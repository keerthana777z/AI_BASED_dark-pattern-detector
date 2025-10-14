AI-Based Dark Pattern Detector for Web Content
This project is a complete, end-to-end system that uses a fine-tuned BERT model to automatically detect and classify manipulative language ("dark patterns") on websites. The final application is an interactive web analyzer built with Streamlit that can scrape any given URL, analyze its text content in real-time, and generate a comprehensive report.

🚀 Key Features
End-to-End Analysis: Simply provide a URL, and the system handles the rest—from scraping to reporting.

High-Accuracy AI Model: Utilizes a fine-tuned BERT model that achieves 97.55% accuracy in classifying dark patterns.

Real-Time Web Scraping: Uses Selenium to automatically extract all relevant text from a live webpage.

Interactive Dashboard: A user-friendly interface built with Streamlit that visualizes the analysis results.

Detailed Reporting: Generates a "Manipulation Score," a bar chart of detected categories, and a detailed table of the manipulative phrases found.

🏛️ Project Architecture
The project follows a complete pipeline from data preparation to live prediction. The offline phases involve preparing the dataset and training the model, while the online phase involves the live analysis in the Streamlit application.

(Note: You will need to export your draw.io diagram as a PNG/JPG, upload it to a site like Imgur, and paste the link here.)

📈 Model Performance
The final BERT model was evaluated on a test set of 774 samples that it had never seen during training. The results demonstrate high performance and reliability.

Overall Accuracy: 97.55%

Weighted F1-Score: 97.00%

Detailed Classification Report
The model performs exceptionally well on the most common categories. The lower scores for Forced Action and Sneaking are due to the very small number of examples (2 and 5, respectively) for those categories in the source dataset.

                 precision    recall  f1-score   support

    Forced Action       0.00      0.00      0.00         2
     Misdirection       0.98      0.93      0.95        86
 Not Dark Pattern       0.96      0.98      0.97       236
      Obstruction       1.00      1.00      1.00        11
         Scarcity       0.99      1.00      0.99       219
         Sneaking       0.50      0.20      0.29         5
     Social Proof       1.00      1.00      1.00       125
          Urgency       0.97      0.98      0.97        90

        accuracy                           0.98       774
       macro avg       0.80      0.76      0.77       774
    weighted avg       0.97      0.98      0.97       774

🛠️ How to Run This Project
Follow these steps to set up and run the project on your local machine.

1. Setup the Environment
First, create and activate a Python virtual environment.

# Create the virtual environment
python -m venv venv

# Activate it (on Mac/Linux)
source venv/bin/activate

Next, install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

2. Prepare the Dataset
Run the script to combine the raw data files into the final dataset used for training.

python create_full_dataset.py

This will generate the combined_dark_patterns_FULL.csv file.

3. Train the AI Model
Run the main training script to fine-tune the BERT model. This step will take 15-45 minutes.

python dark_pattern_detector.py

This will create the final_dark_pattern_model folder containing the trained AI.

4. Launch the Web Application
Once the model is trained, you can start the interactive web analyzer.

streamlit run app.py
