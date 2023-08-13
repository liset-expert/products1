from flask import Flask, request, jsonify, render_template
import redis
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

try:
    # Load the pre-trained model and data
    with open('pickle/model_tfidf.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('pickle/mlb_tfidf.pkl', 'rb') as model_file:
        mlb = pickle.load(model_file)
    with open('pickle/vectorizer_tfidf.pkl', 'rb') as model_file:
        vectorizer = pickle.load(model_file)
    with open('pickle/X_train_vectorized_tfidf.pkl', 'rb') as f:
        X_train_vectorized = pickle.load(f)

except Exception as e:
    print("Error loading model or data:", str(e))

def predict_categories(description):
    """
    Predict categories for a given description.

    Args:
        description (str): The description for which to predict categories.

    Returns:
        list: List of predicted categories.
    """
    new_description_vectorized = vectorizer.transform([description])

    new_description_pred_bin = model.predict(new_description_vectorized)
    new_description_categories = mlb.inverse_transform(new_description_pred_bin)

    print(f'new_description_categories: {new_description_categories}') 

    if not new_description_categories[0]:
        # Get the probabilities predicted by the model
        predicted_probs = model.predict_proba(new_description_vectorized)

        # Convert predicted_probs to a NumPy array
        predicted_probs_array = np.array(predicted_probs)

        # Select the probabilities of the positive class for each category
        positive_probs = predicted_probs_array[:, :, 1]

        # Find the category with the highest probability
        most_probable_label_idx = np.argmax(positive_probs)
        most_probable_label = mlb.classes_[most_probable_label_idx]

        new_description_categories = [tuple([most_probable_label])]


    categories = new_description_categories

    return categories

def get_cache_key(description):
    """
    Generate a cache key for the given description.

    Args:
        description (str): The description for which to generate a cache key.

    Returns:
        str: Cache key.
    """
    return hashlib.md5(description.encode('utf-8')).hexdigest()

@app.route('/')
def index():
    """
    Render the index HTML template.

    Returns:
        str: HTML content of the index page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict categories for a given description using the pre-trained model.

    Returns:
        dict: A JSON response containing predicted categories.
    """
    data = request.get_json()

    cache_key = get_cache_key(data['description'])
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return jsonify({'categories': cached_result.decode('utf-8').split(',')})

    categories_tuple = predict_categories(data['description'])

    # Convert the tuple to a list of strings and format without quotes and parentheses
    categories = [str(cat).strip("(')") for cat in categories_tuple]

    redis_client.set(cache_key, ','.join(categories))

    response = jsonify({'categories': categories})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Content-Type', 'application/json')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')