import pandas as pd
import argparse
import numpy as np
import octopus as dad

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


class BaselineClassifier():
    """
    Logistic Regression Classifier to predict whether an article will be real (0)
    or fake (1) based on the content of the article

    Uncomment print statements to get updates while running
    """
    def __init__(self, X_train, X_test, y_train, y_test, solver, analyzer):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.tolist(), self.X_test.tolist(), self.y_train.tolist(), self.y_test.tolist()
        self.vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=(1, 1)) # Unigrams
        self.classifier = LogisticRegression(solver=solver)
        self.train()

    def train(self):
        # Fit vectorizer to training text
        print('Fitting Vectorizer!')
        self.vectorizer = self.vectorizer.fit(self.X_train)

        # Transform training data and format labels
        X = self.vectorize(self.X_train)
        y = np.array(self.y_train)

        # Fit classifier
        print('classification training time!')
        self.classifier.fit(X, y)

    def vectorize(self, data):
        print('**Vectorizing**')
        return self.vectorizer.transform(data).toarray()

    def evaluate(self):
        X = self.vectorize(self.X_test)
        pred_y = self.classifier.predict(X)
        true_y = np.array(self.y_test)
        report = classification_report(true_y, pred_y)
        return report

def main(args):
    # Have dad load and preprocess data
    data = dad.load_data()
    X_train, X_test, y_train, y_test = dad.preprocess_data(data)

    # Build Classifier and evaluate
    classifier = BaselineClassifier(X_train, X_test, y_train, y_test, args.solver, args.analyzer)
    report = classifier.evaluate()
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--solver', default='liblinear', help='Optimization algorithm.')
    parser.add_argument('--analyzer', default='word', help='Tokenizer algorithm.')

    args = parser.parse_args()
    main(args)
