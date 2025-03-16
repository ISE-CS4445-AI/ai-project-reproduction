import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

# Set the style for the plots
sns.set(style="whitegrid")
plt.style.use('ggplot')

# Load the data
df = pd.read_csv('demographic.csv')

# Display basic info about the dataset
print(df.info())
print("\nSample data:")
print(df.head())

# Clean the gender column - extract the woman probability


def extract_woman_probability(gender_str):
    try:
        # Convert the string representation of dict to actual dictionary
        gender_str = gender_str.replace("'", '"')
        import json
        gender_dict = json.loads(gender_str)
        return gender_dict.get('Woman', 0)
    except:
        return None


# Apply the function to create a new column
df['woman_probability'] = df['gender'].apply(extract_woman_probability)

# Create a figure for all visualizations
plt.figure(figsize=(20, 25))

# 1. Pie chart of dominant race distribution
plt.subplot(3, 2, 1)
race_counts = df['dominant_race'].value_counts()
plt.pie(race_counts, labels=race_counts.index,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title('Distribution of Dominant Race', fontsize=16)

# 2. Count plot of dominant race
plt.subplot(3, 2, 2)
sns.countplot(y='dominant_race', data=df, palette='viridis')
plt.title('Count of Individuals by Dominant Race', fontsize=16)
plt.xlabel('Count')
plt.ylabel('Dominant Race')

# 3. Histogram of age distribution
plt.subplot(3, 2, 3)
sns.histplot(df['age'], bins=10, kde=True)
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Count')

# 4. Box plot of age by dominant race
plt.subplot(3, 2, 4)
sns.boxplot(x='dominant_race', y='age', data=df, palette='Set3')
plt.title('Age Distribution by Dominant Race', fontsize=16)
plt.xlabel('Dominant Race')
plt.ylabel('Age')
plt.xticks(rotation=45)

# 5. Violin plot of age by dominant race
plt.subplot(3, 2, 5)
sns.violinplot(x='dominant_race', y='age', data=df, palette='pastel')
plt.title('Age Distribution Density by Dominant Race', fontsize=16)
plt.xlabel('Dominant Race')
plt.ylabel('Age')
plt.xticks(rotation=45)

# 6. Scatter plot of age vs. woman probability with race as hue
plt.subplot(3, 2, 6)
sns.scatterplot(x='age', y='woman_probability',
                hue='dominant_race', data=df, palette='Dark2', s=100)
plt.title('Age vs. Woman Probability by Race', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Woman Probability')

plt.tight_layout(pad=3.0)
plt.savefig('demographic_visuals_1.png', bbox_inches='tight')
plt.close()

# Create a second figure for additional visualizations
plt.figure(figsize=(20, 20))

# 7. Age distribution KDE by dominant race
plt.subplot(2, 2, 1)
for race in df['dominant_race'].unique():
    sns.kdeplot(df[df['dominant_race'] == race]['age'], label=race, shade=True)
plt.title('Age Distribution Density by Dominant Race', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()

# 8. Heatmap of age vs. emotion, with race as a separate plot
races = df['dominant_race'].unique()
emotions = df['emotion'].unique()
plt.subplot(2, 2, 2)

# Create a crosstab
cross_tab = pd.crosstab(df['dominant_race'], df['emotion'])
sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Race vs. Emotion Counts', fontsize=16)
plt.xlabel('Emotion')
plt.ylabel('Dominant Race')

# 9. Stacked bar chart of race proportions by age group
plt.subplot(2, 2, 3)
# Create age groups
df['age_group'] = pd.cut(df['age'], bins=[20, 25, 30, 35, 40], labels=[
                         '20-25', '26-30', '31-35', '36-40'])
# Create a normalized crosstab
age_race_ct = pd.crosstab(
    df['age_group'], df['dominant_race'], normalize='index')
age_race_ct.plot(kind='bar', stacked=True, colormap='tab10')
plt.title('Race Proportions by Age Group', fontsize=16)
plt.xlabel('Age Group')
plt.ylabel('Proportion')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')

# 10. Bar chart of average age by dominant race
plt.subplot(2, 2, 4)
avg_age = df.groupby('dominant_race')[
    'age'].mean().sort_values(ascending=False)
avg_age.plot(kind='bar', color=sns.color_palette('husl', len(avg_age)))
plt.title('Average Age by Dominant Race', fontsize=16)
plt.xlabel('Dominant Race')
plt.ylabel('Average Age')
plt.xticks(rotation=45)

for i, v in enumerate(avg_age):
    plt.text(i, v + 0.1, f'{v:.1f}', ha='center')

plt.tight_layout(pad=3.0)
plt.savefig('demographic_visuals_2.png', bbox_inches='tight')

# Create a Google Colab-specific display section
print("\nTo display these visualizations in Google Colab, use the following code:")
print('''
# Display the saved figures
from IPython.display import Image, display
display(Image('demographic_visuals_1.png'))
display(Image('demographic_visuals_2.png'))

# To display plots inline in Colab without saving
# Just add this at the beginning of your notebook:
%matplotlib inline
''')

print("\nData Summary Statistics:")
print("\nAge statistics by dominant race:")
print(df.groupby('dominant_race')['age'].describe())

print("\nDominant race distribution:")
print(df['dominant_race'].value_counts(normalize=True) * 100)
