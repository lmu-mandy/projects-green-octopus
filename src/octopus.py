import pandas as pd
import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import re
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(1)

class FFNet(nn.Module):
    """
    Baseline FF Neural Network to predict if an article is real or fake
    based on the content of the article
    """
    def __init__(self, num_words, emb_dim, num_y, embeds=None):
        """
        Constructor to instantiate important variables and information

            Params:
                num_words (int): number of words in vocabulary
                emb_dim (int): number of embedding dimensions
                num_y (int): number of output labels
                embeds (bool): None by default, embedding weights if given
        """
        super().__init__()
        self.emb = nn.EmbeddingBag(num_words, emb_dim)
        if embeds is not None:
            self.emb.weight = nn.Parameter(torch.Tensor(embeds))
        self.linear = nn.Linear(emb_dim, num_y)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Forward method for Feed-Forward Neural Network
        """
        embeds = self.emb(input)
        return self.sigmoid(self.linear(embeds))

def load_data():
    """
    Read and load in True and Fake news article data (structured as CSVs)
    Gives combined dataset with a labels row signifying 0 - true, 1 - fake

        Params: 
            None

        Returns: 
            combined (pd DF): columns - title, text, subject, date, label
    """
    true, fake = pd.read_csv('data/True.csv'), pd.read_csv('data/Fake.csv')
    true = pd.concat([true, pd.Series([0 for i in range(len(true))], name='label')], axis=1)
    fake = pd.concat([fake, pd.Series([1 for i in range(len(fake))], name='label')], axis=1)
    combined = pd.concat([true, fake], axis=0, ignore_index=True)
    return combined

def preprocess_data(df):
    """
    Randomize and split data into train and test sets

        Params:
            df (pd DF): data returned from load_data()

        Returns:
            X_train, X_test, y_train, y_test (sklearn train_test_split): randomized and split data
    """
    labels = df['label']
    data = df['title'].apply(clean_text) # excluded title, subject, and date for the moment
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
    return X_train, X_test, y_train, y_test


def load_vocab(df):
    """
    Return a dictionary mapping each word to its index in the vocabulary.

        Params: 
            df (pd DF): containing text data without labels

        Returns:
            word_to_ix (dict): vocabulary used for training
    """
    word_to_ix = {}
    for idx, text in df.items():
        for word in text.split():
            word_to_ix.setdefault(word, len(word_to_ix) + 1)
    return word_to_ix


def clean_text(text):
    """
    Helper method to remove special characters, extraneous punctuation, and whitespace

        Params:
            text (str): string text to be processed and cleaned

        Returns:
            <undef> (str): new cleaned and processed string
    """
    return re.sub('[^a-zA-Z0-9\s]', '', text).lower()

def main(args):
    """
    Main method at runs the core functionalities of the program. Loads data, processes
    and splits data into training and testing sets. Builds model, optimizer, and loss function
    and then trains and evaluates model.

        Params:
            args (argparser): arguments passed into command line to change hyperparameters
    """

    # Load in data and prepare it for modeling
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train = X_train[:int(len(X_train))]
    # X_train = X_train[:int(len(X_train)*0.05)] # Used to test evaluation because of long train times
    word_to_ix = load_vocab(X_train)

    # Build the model
    model = FFNet(len(word_to_ix) + 1, args.embed_dim, 1)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        print('\nEpoch:', epoch)
        model.train()
        i = 0
        for (idx, text), label in zip(X_train.items(), y_train):
            # Tokenize training data
            text = text.split()
            if not text: # one sample being pulled was empty
                continue

            text_data = [word_to_ix[text[ix]] for ix in range(len(text))]

            x = torch.LongTensor([text_data])
            y = torch.FloatTensor([label]) # weird work around due to not being able to make 0D tensor of 0 or 1

            pred_y = model(x)
            pred_y = torch.flatten(pred_y)

            loss = loss_fn(pred_y, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1

            print(i) if i % 500 == 0 else None
        print('Training Loss:', loss.item())

    pred_y_test = []
    true_y = []
    with torch.no_grad():
        print('EVAL TIME')
        model.eval()
        for (idx, text), label in zip(X_test.items(), y_test):
            text = text.split()
            if not text: # one sample being pulled was empty
                continue

            # sets unknown keys/tokens to 0
            text_data = [word_to_ix[text[ix]] if text[ix] in word_to_ix.keys() else 0 for ix in range(len(text))]
            x = torch.LongTensor([text_data])
            true_y.append(label)

            pred_y = model(x)
            # Model returns likelihood of 0 or 1 so round to find actual value
            pred_y_test.append(torch.round(pred_y).item())

        report = classification_report(true_y, pred_y_test)
        print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Default embedding dimension.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
