import pickle

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

file_string = "language_identifier"

def dump_pickle(vectorizer, classifier):
    pickle.dump(vectorizer, open(f"models/{file_string}_vectorizer.pickle", "wb"))
    pickle.dump(classifier, open(f"models/{file_string}_finalized_model.sav", 'wb'))


def load_pickle():
    vectorizer = pickle.load(open(f"models/{file_string}_vectorizer.pickle", 'rb'))
    classifier = pickle.load(open(f"models/{file_string}_finalized_model.sav", 'rb'))

    return vectorizer, classifier


@st.cache
def load_data():
    data = pd.read_csv('../datasets/language_identification.csv')
    data = data.drop_duplicates(subset='Text').reset_index(drop=True)
    train_doc, test_doc, train_labels, test_labels = train_test_split(data['Text'].values, data['language'].values,
                                                                      test_size=0.33, random_state=42)
    return train_doc, test_doc, train_labels, test_labels


def train_model(train_doc, train_labels):
    vectorizer = CountVectorizer(ngram_range=(1, 4), analyzer='char', max_features=25000)
    vector = vectorizer.fit_transform(train_doc)

    classifier = RandomForestClassifier(n_estimators=1000)
    classifier.fit(vector, train_labels)

    dump_pickle(vectorizer, classifier)

    return vectorizer, classifier


def app():
    """The main body"""

    st.title("Language Identifier")
    st.write('\n\n')

    train_doc, test_doc, train_labels, test_labels = load_data()

    if st.sidebar.button("Click to train a new model"):
        vectorizer, classifier = train_model(train_doc, train_labels)
    else:
        vectorizer, classifier = load_pickle()


    input = st.text_input("Enter The Sentence", "Write Here...")
    if st.button('Predict The Language'):
        result = classifier.predict(vectorizer.transform([input]))[0]
        st.success(result)
    else:
        st.write("Press the above button..")









