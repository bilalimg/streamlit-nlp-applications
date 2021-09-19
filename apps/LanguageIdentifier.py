import pandas as pd
import streamlit as st

from apps.AlgorithmProcessor import AlgorithmProcessor


class LanguageIdentifier(AlgorithmProcessor):
    def __init__(self,
                 pickle_path: str = "apps/models/language_identifier/",
                 model_type: str = "RandomForestClassifier"):
        super().__init__(pickle_path=pickle_path,
                         model_type=model_type)

    @st.cache
    def load_training_data(self):
        """
        Loads train data for training.

        @return: Train data and Train labels.
        """
        data = pd.read_csv('apps/datasets/language_identification.csv')
        data = data.drop_duplicates(subset='Text').reset_index(drop=True)

        train_doc = data['Text'].values
        train_labels = data['language'].values

        return train_doc, train_labels


def app():
    """The main body"""

    st.title("Language Identifier")
    st.write('\n\n')

    class_process = LanguageIdentifier()
    class_process.run()
