import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined dataset we created in the last step
df = pd.read_csv('combined_dark_patterns.csv')

# Set a professional style for the plot
sns.set(style="whitegrid", font_scale=1.1)

# Create the plot figure
plt.figure(figsize=(12, 8))

# Create the bar chart using seaborn
# We use 'y='category' to make it a horizontal bar chart, which is easier to read with long labels
ax = sns.countplot(
    y='category',
    data=df,
    order=df['category'].value_counts().index, # Order bars from most to least frequent
    palette='viridis' # A nice color scheme
)

# Add a title and labels for clarity
ax.set_title('Distribution of Dark Pattern Categories', fontsize=18, pad=20)
ax.set_xlabel('Number of Samples', fontsize=14)
ax.set_ylabel('Pattern Category', fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Add the exact count at the end of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(
        width + 10,
        p.get_y() + p.get_height() / 2,
        f'{int(width)}',
        va='center'
    )

# Adjust layout and save the plot as an image file
plt.tight_layout()
plt.savefig('category_distribution.png')

print("✅ Bar chart saved as 'category_distribution.png'")