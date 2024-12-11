import pandas as pd
import re

# Load the dataset (assuming you saved it as 'dataset-spam.csv')
df = pd.read_csv("dataset_1.csv", encoding="latin1")

# Define a pattern for punctuation
punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'

# Add a column for the word count
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# Add a column for the number count (count of numeric sequences in the message)
df['number_count'] = df['message'].apply(lambda x: len(re.findall(r'\d+', x)))

df['standalone_number_count'] = df['message'].apply(lambda x: len([char for char in x if char.isdigit()]))

# Add a column for the average word length
df['average_word_length'] = df['message'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)

# Add a column for the ratio of words to punctuation
df['ratio_words_punctuation'] = df['word_count'] / df['punct_count'].replace(0, 1)

# Save the updated DataFrame to a new CSV
df.to_csv("updated-dataset-2.csv", index=False)

# Display the first few rows to confirm changes
print(df.head())