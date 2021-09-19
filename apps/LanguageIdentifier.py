import streamlit as st

from apps.AlgorithmProcessor import AlgorithmProcessor

file_string = "language_identifier"


class LanguageIdentifier(AlgorithmProcessor):
    def __init__(self,
                 file_string: str = "language_identifier",
                 pickle_path: str = "apps/models/language_identifier/",
                 model_type: str = "RandomForestClassifier"):
        super().__init__(file_string=file_string,
                         pickle_path=pickle_path,
                         model_type=model_type)

    def run(self):
        self.add_train_option()
        self.predict_input()


def app():
    """The main body"""

    st.title("Language Identifier")
    st.write('\n\n')

    class_process = LanguageIdentifier()
    class_process.run()
