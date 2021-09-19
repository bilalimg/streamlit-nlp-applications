import pickle

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


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
            self.train_model()
        else:
            try:
                self.vectorizer, self.classifier = self.load_pickle()
            except Exception:
                st.error(f"Error occurred while getting models, please train a new model.")

    def train_model(self):
        train_doc, train_labels = self.load_data()
        self.vectorizer = CountVectorizer()
        vector = self.vectorizer.fit_transform(train_doc)

        self.classifier = self.get_classifier()
        self.classifier.fit(vector, train_labels)

        self.dump_pickle()

        st.sidebar.success("New model is trained and saved successfully.")

        return

    def load_data(self):
        return

    def get_classifier(self):
        if self.model_type == "RandomForestClassifier":
            return RandomForestClassifier()
        else:
            return BernoulliNB()

    def predict_input(self):
        input = st.text_input("Enter The Sentence", "Enter The Text Here...")
        if st.button('Predict The Result'):
            result = self.classifier.predict(self.vectorizer.transform([input]))[0]
            self.print_result(result)
        else:
            st.write("Press the above button..")

    def print_result(self, result):
        st.success(result)

    def dump_pickle(self):
        pickle.dump(self.vectorizer, open(f"{self.pickle_path}{self.file_string}_vectorizer.pickle", "wb"))
        pickle.dump(self.classifier, open(f"{self.pickle_path}{self.file_string}_finalized_model.sav", 'wb'))

    def load_pickle(self):
        vectorizer = pickle.load(open(f"{self.pickle_path}{self.file_string}_vectorizer.pickle", 'rb'))
        classifier = pickle.load(open(f"{self.pickle_path}{self.file_string}_finalized_model.sav", 'rb'))

        return vectorizer, classifier
