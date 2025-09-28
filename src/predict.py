import joblib
import sys

# Load the trained model
try:
    model = joblib.load('sentiment_model.joblib')
except FileNotFoundError:
    print("Error: The trained model 'sentiment_model.joblib' was not found.")
    print("Please run 'src/sentiment_analysis.py' first to train and save the model.")
    sys.exit(1)

def predict_sentiment(text):
    """
    Predicts the sentiment of a given text string.
    Returns 'Positive' or 'Negative'.
    """
    # The model expects a list of texts, so we wrap the input in a list
    prediction = model.predict([text])

    # The output is a numpy array (e.g., [1]), so we get the first element
    sentiment = prediction[0]

    return "Positive" if sentiment == 1 else "Negative"

if __name__ == "__main__":
    # Check if a sentence was provided as a command-line argument
    if len(sys.argv) > 1:
        # Join all arguments to form the sentence
        input_sentence = " ".join(sys.argv[1:])
        print(f"Analyzing: '{input_sentence}'")
        result = predict_sentiment(input_sentence)
        print(f"Predicted Sentiment: {result}")
    else:
        # Interactive mode if no command-line argument is given
        print("Sentiment Analysis Predictor")
        print("Enter a sentence to analyze its sentiment, or type 'exit' to quit.")

        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                break

            result = predict_sentiment(user_input)
            print(f"  -> Predicted Sentiment: {result}\n")