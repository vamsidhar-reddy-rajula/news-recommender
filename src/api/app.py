from os import remove
from flask import Flask, json, request, jsonify, render_template, make_response
from flask_restful import Api, Resource
import joblib
import numpy as np
import pandas as pd
import contractions
import random

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

df_articles = pd.read_csv("../data/interim/articles_processed.csv")
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


def text_pipeline(X):
    if isinstance(X, str):
        X = pd.Series(X)
    elif isinstance(X, (pd.Series, pd.DataFrame)):
        pass
    else:
        raise Exception(
            f"Input should either be in 'str' format or a 'series' or 'Dataframe' with a column of text. Received an object of type {type(X)}"
        )

    expanded_contractions = X.apply(lambda x: contractions.fix(x))

    lower = expanded_contractions.str.lower()

    custom_preprocessor = lower.apply(
        lambda x: x.replace("-", " ")
        .replace("'s", "")
        .replace("’s", "")
        .replace("–", "")
    )

    # punctuations
    removed_punctuation = custom_preprocessor.apply(
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


def get_saved_models(n_components):
    if n_components == 300:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_1513_300")
        vectorizer = joblib.load(f"../models/lda_model_0830_1513_300")
        model = joblib.load(f"../models/topic_vector_train_0830_1513_300")
    elif n_components == 240:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_1406_240")
        vectorizer = joblib.load(f"../models/lda_model_0830_1406_240")
        model = joblib.load(f"../models/topic_vector_train_0830_1406_240")
    elif n_components == 180:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_1304_180")
        vectorizer = joblib.load(f"../models/lda_model_0830_1304_180")
        model = joblib.load(f"../models/topic_vector_train_0830_1304_180")
    elif n_components == 150:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_1205_150")
        vectorizer = joblib.load(f"../models/lda_model_0830_1205_150")
        model = joblib.load(f"../models/topic_vector_train_0830_1205_150")
    elif n_components == 120:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_1109_120")
        vectorizer = joblib.load(f"../models/lda_model_0830_1109_120")
        model = joblib.load(f"../models/topic_vector_train_0830_1109_120")
    elif n_components == 90:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_1015_90")
        vectorizer = joblib.load(f"../models/lda_model_0830_1015_90")
        model = joblib.load(f"../models/topic_vector_train_0830_1015_90")
    elif n_components == 60:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_0925_60")
        vectorizer = joblib.load(f"../models/lda_model_0830_0925_60")
        model = joblib.load(f"../models/topic_vector_train_0830_0925_60")
    elif n_components == 30:
        topic_vectors_train = joblib.load(f"../models/vectorizer_0830_0838_30")
        vectorizer = joblib.load(f"../models/lda_model_0830_0838_30")
        model = joblib.load(f"../models/topic_vector_train_0830_0838_30")
    return topic_vectors_train, vectorizer, model


components_saved = [300, 240, 180, 150, 120, 90, 60, 30]


def get_similar_articles(similarity_scores, top_n_values=5):
    results = []
    values = np.sort(similarity_scores, axis=0)[::-1, :][:top_n_values, :]
    similarity_array = np.argsort(similarity_scores, axis=0)[::-1, :][:top_n_values, :]
    for i in range(similarity_array.shape[1]):
        indices = similarity_array[:, i]
        data = pd.DataFrame(
            {
                "article_heading": df_train.iloc[indices].article_heading,
                "similarity_score": np.round(values[:, i] * 100, decimals=2),
                "article_date": df_train.iloc[indices].article_published_on,
                "article_url": df_train.iloc[indices].article_url,
                "article_subheading": df_train.iloc[indices].article_subheading,
            }
        )
        data = data.fillna("")
        for _, row in data.iterrows():
            results.append(row.to_dict())
    return results


def get_article_importance_day_wise(growth=1000):
    diff_from_max_date = (
        df_train.article_published_on - (df_train.article_published_on.max())
    ).dt.days
    return np.exp(diff_from_max_date / growth)


def get_similarity_score(text_vectors, X, factor=None):
    """
    Evalute the cosine similarity between provided 'text_vectors' and trained X (articles trained and stored as a vecotr of topics).
    Return dataframe with index as trained articles and columns as text_vector indices with values as similarity scores
    """
    similarity_scores = cosine_similarity(X, text_vectors, dense_output=True)
    return similarity_scores * factor


#     return np.argsort(similarity_scores, axis=0)[::-1,:][:top_n_values,:]


def process_news_article(
    component,
    include_headings=False,
    heading_weightage=0.6,
    test_indices=None,
    factor=np.ones((df_train.shape[0])).reshape(-1, 1),
):
    topic_vectors_train, vectorizer, model = get_saved_models(component)
    test_lemmas = text_pipeline(df_test.iloc[test_indices].article_body)
    lemma_test_vectors = text_vectorizer(test_lemmas, vectorizer)
    topic_vectors_test = get_topic_vectors(lemma_test_vectors, model)
    similarity_scores = get_similarity_score(
        topic_vectors_test, topic_vectors_train, factor=factor
    )
    heading, result = get_similar_articles(test_indices, similarity_scores)
    if include_headings:
        test_lemmas = text_pipeline(df_test.iloc[test_indices].article_heading)
        lemma_test_vectors = text_vectorizer(test_lemmas, vectorizer)
        topic_vectors_test = get_topic_vectors(lemma_test_vectors, model)
        similarity_scores = (
            heading_weightage
            * (
                get_similarity_score(
                    topic_vectors_test, topic_vectors_train, factor=factor
                )
            )
            + (1 - heading_weightage) * similarity_scores
        )
        _, result = get_similar_articles(test_indices, similarity_scores)
    return heading, result


def ensemble_similarity_scores(
    heading=None,
    text=None,
    components=None,
    include_headings=False,
    heading_weightage=0.6,
    factor=np.ones((df_train.shape[0])).reshape(-1, 1),
):
    component_similarity_scores = []
    for component in components:
        topic_vectors_train, vectorizer, model = get_saved_models(component)
        test_lemmas = text_pipeline(text)
        lemma_test_vectors = text_vectorizer(test_lemmas, vectorizer)
        topic_vectors_test = get_topic_vectors(lemma_test_vectors, model)
        similarity_scores = get_similarity_score(
            topic_vectors_test, topic_vectors_train, factor=factor
        )
        if include_headings:
            test_lemmas = text_pipeline(heading)
            lemma_test_vectors = text_vectorizer(test_lemmas, vectorizer)
            topic_vectors_test = get_topic_vectors(lemma_test_vectors, model)
            similarity_scores = (
                heading_weightage
                * (
                    get_similarity_score(
                        topic_vectors_test, topic_vectors_train, factor=factor
                    )
                )
                + (1 - heading_weightage) * similarity_scores
            )
        component_similarity_scores.append(similarity_scores)
    return component_similarity_scores


# def get_result_dict(test_indices, similarity_scores, top_n_values=10):
#     result = {}
#     for i in range(similarity_scores.shape[0]):
#         headline = df_train.iloc[similarity_scores[i]].article_heading.iloc[0]
#         url = df_train.iloc[similarity_scores[i]].article_url.iloc[0]
#         result[headline] = url
#     return result


def get_results(heading=None, text=None):
    factor = np.ones((df_train.shape[0])).reshape(-1, 1)
    # weights = np.random.dirichlet(np.ones(8),size=1).reshape(8,)
    weights = [
        0.0392685,
        0.09838475,
        0.04760199,
        0.05147573,
        0.04382252,
        0.0741635,
        0.04844888,
        0.59683412,
    ]
    # factor = get_article_importance_day_wise(growth=1000).values.reshape(-1,1)
    component_similarity_scores = ensemble_similarity_scores(
        heading=heading,
        text=text,
        components=components_saved,
        include_headings=True,
        heading_weightage=0.75,
        factor=factor,
    )
    similarity_scores = np.average(
        np.array(component_similarity_scores), axis=0, weights=weights
    )
    return get_similar_articles(similarity_scores, top_n_values=10)


class Recommender(Resource):
    data = []
    input = {"heading": "", "article-body": ""}

    def get(self):
        """
        Default page
        """
        headers = {"Content-Type": "text/html"}
        return make_response(
            render_template(
                "index.html", data=Recommender.data, input=Recommender.input
            ),
            200,
            headers,
        )

    def post(self):
        """
        Return relevant articles as json for the posted text
        """
        if request.form["btn_identifier"] == "article_submission":
            # args = request.args
            # heading = args["heading"]
            # text = args["text"]
            heading = request.form["headline"]
            text = request.form["article-body"]
            Recommender.input["heading"] = heading
            Recommender.input["article-body"] = text
            # return jsonify(get_results(heading=heading, text=text))
            Recommender.data = get_results(
                heading=Recommender.input["heading"],
                text=Recommender.input["article-body"],
            )
        elif request.form["btn_identifier"] == "clear_values":
            Recommender.data.clear()
            Recommender.input["heading"] = ""
            Recommender.input["article-body"] = ""
        elif request.form["btn_identifier"] == "feeling_lucky":
            test_indices = random.sample(range(df_test.shape[0]), 1)
            Recommender.input["heading"] = df_test.iloc[
                test_indices
            ].article_heading.values[0]
            Recommender.input["article-body"] = df_test.iloc[
                test_indices
            ].article_body.values[0]
            # return jsonify(get_results(heading=heading, text=text))
            Recommender.data = get_results(
                heading=Recommender.input["heading"],
                text=Recommender.input["article-body"],
            )

        headers = {"Content-Type": "text/html"}
        return make_response(
            render_template(
                "index.html", data=Recommender.data, input=Recommender.input
            ),
            200,
            headers,
        )


api.add_resource(Recommender, "/", endpoint="home")

if __name__ == "__main__":
    app.run(port=5000, debug=False)
