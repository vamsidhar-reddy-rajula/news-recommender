from os import remove
from flask import Flask, request, jsonify, render_template, make_response
from flask_restful import Api, Resource
import joblib
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from string import punctuation


from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
import spacy

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
api = Api(app)

df_articles = pd.read_csv("data/interim/articles_processed.csv")
df_articles.article_published_on = df_articles.article_published_on.astype(
    np.datetime64
)
df_train, df_test = (
    df_articles[
        df_articles.article_published_on < datetime(year=2021, day=20, month=8)
    ],
    df_articles[
        df_articles.article_published_on >= datetime(year=2021, day=20, month=8)
    ],
)
vectorizer = joblib.load("models/vectorizer_0830_0350")
model = joblib.load("models/lda_model_0830_0350")
topic_vectors_train = joblib.load("models/topic_vector_train_0830_0350")


def text_pipeline(X):
    if isinstance(X, str):
        X = pd.Series(X)
    elif isinstance(X, (pd.Series, pd.DataFrame)):
        pass
    else:
        raise Exception(
            f"Input should either be in 'str' format or a 'series' or 'Dataframe' with a column of text. Received an object of type {type(X)}"
        )

    # punctuations
    removed_punctuation = X.apply(
        lambda x: "".join([c for c in x if c not in punctuation])
    )

    # stop words
    stop_words = stopwords.words("english")
    removed_stop_words = removed_punctuation.apply(
        lambda x: " ".join(
            [word for word in word_tokenize(x) if word not in stop_words]
        )
    )
    removed_stop_words = removed_stop_words.apply(lambda x: remove_stopwords(x))
    all_stopwords_gensim = STOPWORDS.union(
        set(["the", "say", "said", "get", "it", "in", "like", "new", "year"])
    )
    removed_stop_words = removed_stop_words.apply(
        lambda x: " ".join(
            [word for word in word_tokenize(x) if word not in all_stopwords_gensim]
        )
    )
    sp = spacy.load("en_core_web_sm")
    all_stopwords = sp.Defaults.stop_words
    removed_stop_words = removed_stop_words.apply(
        lambda x: " ".join(
            [word for word in word_tokenize(x) if word not in all_stopwords]
        )
    )

    # Stemming and Lematizing
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stem = removed_stop_words.apply(
        lambda x: " ".join([stemmer.stem(word) for word in word_tokenize(x)])
    )
    lemma = stem.apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)])
    )

    return lemma


def text_vectorizer(X, vectorizer, fit=False):
    if fit:
        return vectorizer.fit_transform(X)
    else:
        return vectorizer.transform(X)


def get_topic_vectors(X, model, fit=False):
    if fit:
        return model.fit_transform(X)
    else:
        return model.transform(X)


def get_similar_articles(text_vectors, X, top_n_values=10):
    """
    Evalute the cosine similarity between provided 'text_vectors' and trained X (articles trained and stored as a vecotr of topics).
    Return dataframe with index as trained articles and columns as text_vector indices with values as similarity scores
    """
    similarity_scores = cosine_similarity(X, text_vectors, dense_output=True)
    return np.argsort(similarity_scores, axis=0)[::-1, :][:top_n_values, :]


def get_result_dict(X):
    result = {}
    test_lemmas = text_pipeline(X)
    lemma_test_vectors = text_vectorizer(test_lemmas, vectorizer)
    topic_vectors_test = get_topic_vectors(lemma_test_vectors, model)
    similarity_scores = get_similar_articles(topic_vectors_test, topic_vectors_train)
    for i in range(similarity_scores.shape[0]):
        headline = df_train.iloc[similarity_scores[i]].article_heading.iloc[0]
        url = df_train.iloc[similarity_scores[i]].article_url.iloc[0]
        result[headline] = url
    return result


def print_similar_articles(test_indices, test, df_train, similarity_array):
    for i in range(similarity_array.shape[1]):
        indices = similarity_array[:, i]
        print("\n")
        print(test.iloc[test_indices[i]].article_heading)
        print("\n")
        print(
            df_train.iloc[indices]
            .sort_values(["article_published_on"], ascending=False)
            .article_heading
        )


class Recommender(Resource):
    def get(self):
        """
        Default page
        """
        args = request.args
        test = args["test"]
        return jsonify(get_result_dict(test))

    def post(self):
        """
        Return relevant articles as json for the posted text
        """
        pass


api.add_resource(Recommender, "/")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
