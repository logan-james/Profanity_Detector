import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report  # Add this line
import pickle

# Load the dataset
df = pd.read_csv('train.csv')

# Create a 'profane' column: 1 if the tweet has hate speech or offensive language annotations, else 0
df['profane'] = df.apply(lambda row: 1 if (row['hate_speech_count'] > 0 or row['offensive_language_count'] > 0) else 0, axis=1)

# Preprocess the tweet text (assuming the column name is 'tweet')
df['tweet'] = df['tweet'].fillna('')  # Handling missing values

# Define X and y
X = df['tweet']  # The tweet text
y = df['profane']  # The target column indicating whether it's profane or not

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the tweets
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # You can add more preprocessing options here
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train the model using Logistic Regression
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

# Save the trained model, vectorizer, and test data
with open('profanity_logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
