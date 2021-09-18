import pickle

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

file_string = "sentiment_analysis"


def dump_pickle(vectorizer, classifier):
    pickle.dump(vectorizer, open(f"models/{file_string}_vectorizer.pickle", "wb"))
    pickle.dump(classifier, open(f"models/{file_string}_finalized_model.sav", 'wb'))


def load_pickle():
    vectorizer = pickle.load(open(f"models/{file_string}_vectorizer.pickle", 'rb'))
    classifier = pickle.load(open(f"models/{file_string}_finalized_model.sav", 'rb'))

    return vectorizer, classifier


@st.cache
def preprocess_data():
    with open("../datasets/sentiment_labeled.txt", "r") as text_file:
        data = text_file.read().split('\n')

    processed_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            processed_data.append(single_data.split("\t"))

    train_doc = [processed_data[0] for processed_data in processed_data]
    train_labels = [processed_data[1] for processed_data in processed_data]

    return train_doc, train_labels


def train_model(train_doc, train_labels):
    vectorizer = CountVectorizer(binary='true')
    vector = vectorizer.fit_transform(train_doc)

    classifier = RandomForestClassifier()
    classifier.fit(vector, train_labels)

    dump_pickle(vectorizer, classifier)

    return vectorizer, classifier


def app():
    """The main body"""

    st.title("Sentiment Analyzer")
    st.write('\n\n')

    train_doc, train_labels = preprocess_data()

    if st.sidebar.button("Click to train a new model"):
        vectorizer, classifier = train_model(train_doc, train_labels)
    else:
        vectorizer, classifier = load_pickle()

    input = st.text_input("Enter The Sentence", "Write Here...")
    if st.button('Predict The Sentiment'):
        result = classifier.predict(vectorizer.transform([input]))[0]
        print_text = "Positive" if result[0] == '1' else "Negative"
        st.success(print_text)
    else:
        st.write("Press the above button..")
