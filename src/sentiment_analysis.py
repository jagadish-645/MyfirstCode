import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/rotten_tomatoes_reviews.csv')

# The dataset has 'reviews' and 'labels' (1 for positive, 0 for negative)
# Let's rename columns for clarity, although it's not strictly necessary as the names are already good.
df.rename(columns={'reviews': 'review', 'labels': 'sentiment'}, inplace=True)

# Drop any rows with missing values
df.dropna(subset=['review', 'sentiment'], inplace=True)

# Separate features (X) and target (y)
X = df['review']
y = df['sentiment']

# 2. Split Data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create a pipeline
# The pipeline will first transform the text data using TF-IDF and then train a logistic regression model.
print("Creating and training the model pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', LogisticRegression(solver='liblinear'))
])

# 4. Train the model
pipeline.fit(X_train, y_train)

# 5. Evaluate the model
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.4f}")

# 6. Save the trained model
print("Saving the trained model to 'sentiment_model.joblib'...")
joblib.dump(pipeline, 'sentiment_model.joblib')

print("Model training and saving complete.")