from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import requests

app = Flask(__name__)

# Function to read text file from GitHub
def read_text_file_from_github(file_url):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            return response.text.splitlines()  # Split text into lines
        else:
            print(f"Failed to fetch file from {file_url}. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching file from {file_url}: {e}")
        return None

# Load data from GitHub text file
github_file_url = "https://raw.githubusercontent.com/Ridamgupta/Proj_mlops/main/Emocontext.txt"
lines = read_text_file_from_github(github_file_url)
data = []
for line in lines[1:]:
    parts = line.strip().split('\t')
    text = ' '.join(parts[1:4])
    label = parts[-1]
    data.append((text, label))

X = [sample[0] for sample in data]
y = [sample[1] for sample in data]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Define routes for Flask app
@app.route('/')
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Prediction</title>
</head>
<body>
    <h1>Emotion Prediction</h1>
    <form action="/predict" method="post">
        <label for="text">Enter text:</label><br>
        <textarea id="text" name="text" rows="4" cols="50"></textarea><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
"""

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_emotion(text)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Prediction Result</title>
</head>
<body>
    <h1>Emotion Prediction Result</h1>
    <p>Predicted emotion: {prediction}</p>
</body>
</html>
"""

def predict_emotion(text):
    prediction = pipeline.predict([text])
    return prediction[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
