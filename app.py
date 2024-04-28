from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_fake_news(title):
    # Preprocess input data
    input_data = vectorizer.transform([title])

    # Make prediction
    prediction = model.predict(input_data)

    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
        prediction = predict_fake_news(title)
        # Redirect to the predict page with the prediction as a query parameter
        return redirect(url_for('predict', prediction=prediction))
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Get the prediction from the query parameters
    prediction = request.args.get('prediction')
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
