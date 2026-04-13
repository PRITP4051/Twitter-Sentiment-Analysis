import pandas as pd
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("clean_dataset_8000.csv")

# 🧹 Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# ❌ Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

df['clean_text'] = df['clean_text'].apply(remove_stopwords)

# 🏷️ Auto labeling
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    
    if score > 0:
        return 1   # Positive
    elif score < 0:
        return 0   # Negative
    else:
        return 2   # Neutral

df['label'] = df['clean_text'].apply(get_sentiment)

# Save
df.to_csv("labeled_data.csv", index=False)

print("✅ Preprocessing + labeling done!")
print(df['label'].value_counts())