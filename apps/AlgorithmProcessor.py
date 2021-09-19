import pickle

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class AlgorithmProcessor:
    def __init__(self,
                 file_string: str,
                 pickle_path: str,
                 model_type: str):

        self.file_string: str = file_string
        self.pickle_path: str = pickle_path
        self.model_type: str = model_type

    def run(self):
        self.add_train_option()
        self.predict_input()

    def add_train_option(self):
        if st.sidebar.button("Click to train a new model"):
            self.vectorizer, self.classifier = self.train_model()
        else:
            try:
                self.vectorizer, self.classifier = self.load_pickle()
            except Exception:
                st.error(f"Error occurred while getting models, please train a new model.")

    def train_model(self):
        train_doc, test_doc, train_labels, test_labels = self.load_data()
        vectorizer = CountVectorizer()
        vector = vectorizer.fit_transform(train_doc)

        classifier = RandomForestClassifier()
        classifier.fit(vector, train_labels)

        self.dump_pickle(vectorizer, classifier)

        return vectorizer, classifier

    @st.cache
    def load_data(self):
        data = pd.read_csv('apps/datasets/language_identification.csv')
        data = data.drop_duplicates(subset='Text').reset_index(drop=True)
        train_doc, test_doc, train_labels, test_labels = train_test_split(data['Text'].values, data['language'].values,
                                                                          test_size=0.1, random_state=42)
        return train_doc, test_doc, train_labels, test_labels

    def get_classifier(self):
        if self.model_type == "RandomForestClassifier":
            return RandomForestClassifier()
        else:
            return BernoulliNB()

    def predict_input(self):
        input = st.text_input("Enter The Sentence", "Enter The Text Here...")
        if st.button('Predict The Result'):
            result = self.classifier.predict(self.vectorizer.transform([input]))[0]
            st.success(result)
        else:
            st.write("Press the above button..")

    def dump_pickle(self):
        pickle.dump(self.vectorizer, open(f"{self.pickle_path}{self.file_string}_vectorizer.pickle", "wb"))
        pickle.dump(self.classifier, open(f"{self.pickle_path}{self.file_string}_finalized_model.sav", 'wb'))

    def load_pickle(self):
        vectorizer = pickle.load(open(f"{self.pickle_path}{self.file_string}_vectorizer.pickle", 'rb'))
        classifier = pickle.load(open(f"{self.pickle_path}{self.file_string}_finalized_model.sav", 'rb'))

        return vectorizer, classifier
