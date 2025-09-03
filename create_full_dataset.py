import pandas as pd

print("Loading original datasets...")
# Load the first dataset from the CSV file
df_mathur = pd.read_csv('dark-patterns.csv')
df_mathur_cleaned = df_mathur[['Pattern String', 'Pattern Category']].rename(
    columns={'Pattern String': 'text', 'Pattern Category': 'category'}
)

# Load the second dataset from the TSV file
df_yamana = pd.read_csv('dataset.tsv', sep='\t')
df_yamana_cleaned = df_yamana[['text', 'Pattern Category']].rename(
    columns={'Pattern Category': 'category'}
)

# Combine the two dataframes
df_combined = pd.concat([df_mathur_cleaned, df_yamana_cleaned], ignore_index=True)

# Remove any rows where the text or category is missing
df_combined.dropna(subset=['text', 'category'], inplace=True)

# CRITICAL CHANGE: We are KEEPING the "Not Dark Pattern" category this time.

# Save the new, complete dataset to a new file to avoid confusion
df_combined.to_csv('combined_dark_patterns_FULL.csv', index=False)

print("\n✅ New dataset 'combined_dark_patterns_FULL.csv' created successfully!")
print("It now includes the 'Not Dark Pattern' category.")
print("\nNew Category Distribution:")
print(df_combined['category'].value_counts())
