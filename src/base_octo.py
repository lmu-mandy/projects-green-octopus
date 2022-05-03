import re
import pandas as pd
import argparse
import numpy as np
import octopus as dad

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
        # print('Fitting Vectorizer!')
        self.vectorizer = self.vectorizer.fit(self.X_train)

        # Transform training data and format labels
        X = self.vectorize(self.X_train)
        y = np.array(self.y_train)

        # Fit classifier
        # print('classification training time!')
        self.classifier.fit(X, y)

    def vectorize(self, data):
        # print('**Vectorizing**')
        return self.vectorizer.transform(data).toarray()

    def evaluate(self):
        X = self.vectorize(self.X_test)
        pred_y = self.classifier.predict(X)
        true_y = np.array(self.y_test)
        report = classification_report(true_y, pred_y)
        return report

def make_prediction(raw_text, classifier, correct):
    sentence = dad.preprocess_sentence(raw_text)
    processed_sentence = classifier.vectorize(sentence)
    raw_prediction = classifier.classifier.predict(processed_sentence)[0]
    prediction = "real news!" if raw_prediction == 0 else "fake news!"
    color = bcolors.OKGREEN if raw_prediction == 0 else bcolors.FAIL
    print(f"'{raw_text}' is {color}{prediction}{bcolors.ENDC} and should be {bcolors.OKCYAN}{correct}{bcolors.ENDC}")

def main(args):
    # Have dad load and preprocess data
    data = dad.load_data()
    X_train, X_test, y_train, y_test = dad.preprocess_data(data)
    # Build Classifier and evaluate
    classifier = BaselineClassifier(X_train, X_test, y_train, y_test, args.solver, args.analyzer)

    make_prediction("The libertarian party is disbanding, in favor of starting an ant farm in Wisconsin", classifier)
    make_prediction("New discovery shows that your liver is filled with microplastics", classifier)
    make_prediction("Police in Paris ticketed protesters for carrying the French flag and saying the word 'freedom.'", classifier)
    make_prediction("ohns Hopkins University research shows that someone can 'be vaccinated with a PCR swab test without knowing.'", classifier)
    make_prediction("STEVEN DONZIGER WALKS FREE AFTER 993 DAYS OF ‘COMPLETELY UNJUST’ DETENTION", classifier)

    # print('Evaluation Time!')
    report = classifier.evaluate()
    # print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--solver', default='liblinear', help='Optimization algorithm.')
    parser.add_argument('--analyzer', default='word', help='Tokenizer algorithm.')

    args = parser.parse_args()
    main(args)
