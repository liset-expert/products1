import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def train_and_predict_model(X_train, y_train, X_test, y_test):

    """
    Trains a multi-label classification model using Logistic Regression with TF-IDF vectorization,
    performs grid search for hyperparameter tuning, predicts labels for the test set, and calculates
    Average Precision and F1 Score per class.

    Args:
        X_train (list): List of training texts.
        y_train (list of lists): List of lists containing multi-label training labels.
        X_test (list): List of test texts.
        y_test (list of lists): List of lists containing multi-label test labels.

    Returns:
        tuple: A tuple containing Average Precision, F1 Score per class, and the classification report.
    """

    # Create an instance of TfidfVectorizer with the desired parameters
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1, 1), sublinear_tf=True)

    # Convert text to numeric features for the training set
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Convert the labels to a binary array for the training set
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)

    # Create a multi-label classification model using Logistic Regression
    model = MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=2000))

    # Define possible values for hyperparameters
    param_grid = {
        'estimator__C': [0.01, 1, 10],
        'estimator__penalty': ['l1', 'l2']
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro')
    grid_search.fit(X_train_vectorized, y_train_bin)

    # Get the best model with the best hyperparameters
    best_model = grid_search.best_estimator_

    # Convert text to numeric features for test set
    X_test_vectorized = vectorizer.transform(X_test)

    # Predict the labels in the test set
    y_pred_bin = best_model.predict(X_test_vectorized)

    # Convert the actual labels of the test set to a binary array
    y_test_bin = mlb.transform(y_test)

    # Calculate Average Precision and F1 Score per class
    avg_precision = average_precision_score(y_test_bin, y_pred_bin, average='micro')
    f1_per_class = f1_score(y_test_bin, y_pred_bin, average=None)

    print(f'Average Precision: {avg_precision:.2f}')

    class_names = mlb.classes_
    classification_rep = classification_report(y_test_bin, y_pred_bin, target_names=class_names)

    return avg_precision, f1_per_class, classification_rep