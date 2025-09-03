import pandas as pd

# Load the first dataset from the CSV file
df_mathur = pd.read_csv('dark-patterns.csv')

# We only need the 'Pattern String' and 'Pattern Category' columns.
# Let's rename them for consistency.
df_mathur_cleaned = df_mathur[['Pattern String', 'Pattern Category']].rename(
    columns={'Pattern String': 'text', 'Pattern Category': 'category'}
)

# Load the second dataset from the TSV file (tab-separated)
df_yamana = pd.read_csv('dataset.tsv', sep='\t')

# We only need the 'text' and 'Pattern Category' columns.
# Let's rename the 'Pattern Category' column for consistency.
df_yamana_cleaned = df_yamana[['text', 'Pattern Category']].rename(
    columns={'Pattern Category': 'category'}
)

# Combine the two cleaned dataframes into a single one
df_combined = pd.concat([df_mathur_cleaned, df_yamana_cleaned], ignore_index=True)

# Remove any rows where the 'text' column might be empty or invalid
df_combined.dropna(subset=['text'], inplace=True)

# Remove rows where the category is 'Not Dark Pattern' to focus the model on detecting actual dark patterns
df_combined = df_combined[df_combined['category'] != 'Not Dark Pattern']

# Save the final combined data to a new CSV file
df_combined.to_csv('combined_dark_patterns.csv', index=False)

# --- Output ---
print("✅ Datasets loaded, cleaned, and combined successfully!")
print(f"Total samples in the final dataset: {len(df_combined)}")

print("\nHere's a preview of the combined data:")
print(df_combined.head())

print("\nDistribution of Dark Pattern Categories:")
print(df_combined['category'].value_counts())