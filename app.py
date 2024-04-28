# Flask: Used for creating the web application.
# render_template: Used for rendering HTML templates.
# request: Used for accessing request data, such as form inputs.
# redirect and url_for: Used for redirecting to other routes within the application.
from flask import Flask, render_template, request, redirect, url_for

# TfidfVectorizer and LogisticRegression: Used for text vectorization and classification, respectively, from scikit-learn.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# pickle: Used for loading pre - trained models and vectorizers saved as pickle files.
import pickle
# app = Flask(__name__): Creates a Flask application instance.
app = Flask(__name__)

# Load the model
# with open('model.pkl', 'rb') as f:: Opens the saved model file in read-binary mode.
with open('model.pkl', 'rb') as f:
    # Loads the pre - trained logistic regression model.
    model = pickle.load(f)

# Load the vectorizer
    # Opens the saved vectorizer file in read - binary mode.
with open('vectorizer.pkl', 'rb') as f:
    # Loads the pre - trained TF - IDF vectorizer.
    vectorizer = pickle.load(f)

# Takes a title as input, preprocesses it using the loaded vectorizer,
# makes a prediction using the loaded model, and returns the predicted label.
def predict_fake_news(title):
    # Preprocess input data
    input_data = vectorizer.transform([title])

    # Make prediction
    prediction = model.predict(input_data)

    return prediction[0]

    # Handles both GET and POST requests. If it's a POST request:
    # Extracts the title from the submitted form.
    # Calls predict_fake_news to get the prediction for the title.
    # Redirects to the '/predict' route with the prediction as a query parameter.
    # If it's a GET request: Renders the 'index.html' template, which contains the form to input the title.
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form['title']
        prediction = predict_fake_news(title)
        # Redirect to the predict page with the prediction as a query parameter
        return redirect(url_for('predict', prediction=prediction))
    return render_template('index.html')

    # Renders the 'predict.html' template.
    # Retrieves the prediction from the query parameters passed in the URL.
    # Passes the prediction to the template for display.
@app.route('/predict')
def predict():
    # Get the prediction from the query parameters
    prediction = request.args.get('prediction')
    return render_template('predict.html', prediction=prediction)

    # if __name__ == '__main__':: Ensures that the app is only run if the script is executed directly, not imported as a module.
    # app.run(debug=True): Starts the Flask application in debug mode, allowing for easier debugging and development.
if __name__ == '__main__':
    app.run(debug=True)
