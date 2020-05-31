import gensim
import pandas as pd
import numpy as np

from math import sqrt

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn import metrics

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import regression_paper

class Models:
    def __init__(self):
        self.reg_paper_obj = regression_paper.Gutenberg_Emotion()

    def emobank_split(self, corpus_path=regression_paper.EMOBANK_PATH):
        df = self.reg_paper_obj.get_corpus(corpus_path)
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "dev"]
        df_test = df[df["split"] == "test"]

        return (df_train, df_val, df_test)

    def clean_text(self, corpus):
        # TODO: Add option for Lemmetization
        punctuation = """@.,?!:;(){}[]"""
        corpus = [z.strip().lower().replace('\n','') for z in corpus]
        for c in punctuation:
            # inserts whitespace on both sides of a punctuation
            # so that in the next step it gets split
            corpus = [z.replace(c, ' %s '%c) for z in corpus]
        corpus = [z.split() for z in corpus]

        return corpus

    def labelize(self, text, label_type):
        LabeledSentence = gensim.models.doc2vec.LabeledSentence
        labelized = list()
        for i, v in enumerate(text):
            label = '%s_%s'%(label_type, i)
            labelized.append(LabeledSentence(v, [label]))

        return labelized

    def emobank_preprocess(self, df_train, df_val, emotion="D"):
        X_train, y_train = df_train["text"], df_train[emotion]
        X_val, y_val = df_val["text"], df_val[emotion]

        X_train = self.labelize(self.clean_text(X_train), 'TRAIN')
        X_val = self.labelize(self.clean_text(X_val), 'DEV')

        return (X_train, X_val)

    def gensim_build_vocab(self, X_train, X_val, size=300, window=4, negative=2, workers=3, sample=1e-3, min_count=1):
        model_dm = gensim.models.Doc2Vec(min_count=min_count, window=window, 
                                        size=size, sample=sample, 
                                        negative=negative, workers=workers)

        model_dbow = gensim.models.Doc2Vec(min_count=min_count, window=window, 
                                        size=size, sample=sample, 
                                        negative=negative, dm=0, 
                                        workers=workers)

        model_dm.build_vocab(X_train + X_val)
        model_dbow.build_vocab(X_train + X_val)

        return (model_dm, model_dbow)

    def gensim_train(self, model_dw, model_dbow, X, epochs=20):
        model_dm.train(X, total_examples = len(X), epochs=epochs)
        model_dbow.train(X, total_examples = len(X), epochs=epochs)

        return (model_dm, model_dbow)

    def gensim_vectors(self, model, corpus, size):
        vecs = [np.array(model[z.tags[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    def model_vectors(self, model_dm, model_dbow, X, size=300):
        vecs_dm = self.gensim_vectors(model_dm, X, size)
        vecs_dbow = self.gensim_vectors(model_dbow, X, size)

        vecs = np.hstack((vecs_dm, vecs_dbow))
        return vecs
