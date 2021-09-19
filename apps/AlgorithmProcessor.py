import pickle

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


class AlgorithmProcessor:
    """
    Base class Algorithms that will run.
    """

    def __init__(self,
                 pickle_path: str,
                 model_type: str):
        """
        Base class of AlgorithmProcessor.

        @param pickle_path: Jobs that will run on cluster.
        @param model_type: All necessary tags for resources.
        """
        self.pickle_path: str = pickle_path
        self.model_type: str = model_type

    def run(self):
        """
        Init function of AlgorithmProcessor, which initiates actions to be displayed in the interface in Streamlit.
        """
        self.add_train_option()
        self.inference_the_input()

    def add_train_option(self):
        """
        If exits, loads classifier and vectorizer for NLP model. It also adds a button that triggers the new model
        training. If the user clicks, the new model is trained, and when completed, it replaces the old model.
        """
        if st.sidebar.button("Click to train a new model"):
            self.train_model()
        else:
            try:
                self.vectorizer, self.classifier = self.load_pickle()
            except Exception:
                st.error(f"Error occurred while getting models, please train a new model.")

    def train_model(self):
        """
        Trains the model for the selected algorithm. Saves classifier and vectorizer as pickle.
        """
        train_doc, train_labels = self.load_training_data()
        self.vectorizer = CountVectorizer()
        vector = self.vectorizer.fit_transform(train_doc)

        self.classifier = self.get_classifier()
        self.classifier.fit(vector, train_labels)

        self.dump_pickle()

        st.sidebar.success("New model is trained and saved successfully.")

        return

    def load_training_data(self):
        """
        Loads train data for training.

        @return: Train data and Train labels.
        """
        return

    def get_classifier(self):
        """
        Returns the required classifier for the selected model type.

        @return: Classifier.
        """
        if self.model_type == "RandomForestClassifier":
            return RandomForestClassifier()
        else:
            return BernoulliNB()

    def inference_the_input(self):
        """
        Provides the necessary input scheme to the user to enter the text and inferences this input.
        """
        input = st.text_input("Enter The Sentence", "Enter The Text Here...")

        if st.button('Predict The Result'):
            result = self.classifier.predict(self.vectorizer.transform([input]))[0]
            self.print_result(result)
        else:
            st.write("Press the above button..")

    def print_result(self, result):
        """
        Prints the inference result.

        @param result: Inference result.
        """
        st.success(result)

    def dump_pickle(self):
        """
        Saves vectorizer and classifier pickles.
        """
        pickle.dump(self.vectorizer, open(self.pickle_path + "vectorizer.pickle", "wb"))
        pickle.dump(self.classifier, open(self.pickle_path + "finalized_model.sav", 'wb'))

    def load_pickle(self):
        """
        Loads vectorizer and classifier pickles if they exit.

        @return: Vectorizer and Classifier.
        """
        vectorizer = pickle.load(open(self.pickle_path + "vectorizer.pickle", 'rb'))
        classifier = pickle.load(open(self.pickle_path + "finalized_model.sav", 'rb'))
        return vectorizer, classifier
