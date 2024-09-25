import pickle

# Function to load the saved model
def load_model():
    with open("profanity_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Function to load the saved vectorizer
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Function to predict comment
def predict_comment(model, vectorizer, comment):
    # Vectorize and preprocess the comment
    comment_vectorized = vectorizer.transform([comment])
    result = model.predict(comment_vectorized)
    return result[0]
