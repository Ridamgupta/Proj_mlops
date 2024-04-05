from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import requests
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])


pipeline.fit(X_train, y_train)

def predict_emotion(text):
    prediction = pipeline.predict([text])
    return prediction[0]

while True:
    try:
        input_text = input("Enter text (press 'q' to quit): ")
        if input_text.lower() == 'q':
            break
        else:
            prediction = predict_emotion(input_text)
            print("Predicted emotion:", prediction)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
