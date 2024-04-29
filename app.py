# app.py

from flask import Flask, request, render_template
import joblib
from preprocess import preprocess_text
import nltk

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')

# Load your trained model
model = joblib.load('modal.pkl')

# Load the TfidfVectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict(text):
    preprocessed_text = preprocess_text(text)
    
    # Use the loaded vectorizer to transform the text
    text_vector = vectorizer.transform([preprocessed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_result():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict(text)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
