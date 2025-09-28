# My First AI Project: Sentiment Analysis

This project is a simple sentiment analysis model that predicts whether a piece of text (like a movie review) has a positive or negative sentiment. It's a great first project for anyone looking to get started with AI and Natural Language Processing (NLP).

The model is trained on the "Rotten Tomatoes Reviews Dataset" and uses a `scikit-learn` pipeline with TF-IDF vectorization and a Logistic Regression classifier.

## Project Structure

- `data/`: Contains the dataset used for training.
- `src/`: Contains the Python source code.
  - `sentiment_analysis.py`: The script to train the model.
  - `predict.py`: The script to make predictions on new text.
- `requirements.txt`: A list of the Python dependencies required for this project.
- `sentiment_model.joblib`: The pre-trained model file, generated after running the training script.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

There are two main steps to using this project: training the model and making predictions.

### 1. Train the Model

To train the sentiment analysis model, run the `sentiment_analysis.py` script from the root directory of the project:

```bash
python3 src/sentiment_analysis.py
```

This script will load the dataset, train the model, and save the trained pipeline to a file named `sentiment_model.joblib`. You only need to run this once.

### 2. Make Predictions

Once the model is trained, you can use the `predict.py` script to analyze the sentiment of any sentence.

You can run it in two ways:

**A) Pass a sentence as a command-line argument:**

```bash
python3 src/predict.py "This was the best movie I have ever seen!"
```
**Output:**
```
Analyzing: 'This was the best movie I have ever seen!'
Predicted Sentiment: Positive
```

---

```bash
python3 src/predict.py "The acting was terrible and the plot was boring."
```
**Output:**
```
Analyzing: 'The acting was terrible and the plot was boring.'
Predicted Sentiment: Negative
```

**B) Run in interactive mode:**

If you run the script without any arguments, it will enter an interactive mode where you can type sentences one by one.

```bash
python3 src/predict.py
```
**Output:**
```
Sentiment Analysis Predictor
Enter a sentence to analyze its sentiment, or type 'exit' to quit.
> I loved the experience, it was truly magical.
  -> Predicted Sentiment: Positive

> I would not recommend this to my worst enemy.
  -> Predicted Sentiment: Negative

> exit
```