import streamlit as st

from apps.AlgorithmProcessor import AlgorithmProcessor


class SentimentAnalysis(AlgorithmProcessor):
    def __init__(self,
                 pickle_path: str = "apps/models/sentiment_analysis/",
                 model_type: str = "BernoulliNB"):
        super().__init__(pickle_path=pickle_path,
                         model_type=model_type)

    @st.cache
    def load_training_data(self):
        """
        Loads train data for training.

        @return: Train data and Train labels.
        """
        with open("apps/datasets/sentiment_labeled.txt", "r") as text_file:
            data = text_file.read().split('\n')

        processed_data = []
        for single_data in data:
            if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
                processed_data.append(single_data.split("\t"))

        train_doc = [processed_data[0] for processed_data in processed_data]
        train_labels = [processed_data[1] for processed_data in processed_data]

        return train_doc, train_labels

    def print_result(self, result):
        """
        Prints the inference result.

        @param result: Inference result.
        """
        print_result = "Positive" if result[0] == '1' else "Negative"
        st.success(print_result)


def app():
    """The main body"""

    st.title("Sentiment Analyzer")
    st.write('\n\n')

    class_process = SentimentAnalysis()
    class_process.run()
