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

    def emobank_preprocess(self, df_train, df_val, emotion="V"):
        X_train, y_train = df_train["text"], df_train[emotion]
        X_val, y_val = df_val["text"], df_val[emotion]

        X_train = self.labelize(self.clean_text(X_train), 'TRAIN')
        X_val = self.labelize(self.clean_text(X_val), 'DEV')

        return (X_train, y_train, X_val, y_val)

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

    def gensim_train(self, model_dm, model_dbow, X, epochs=20):
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

    def mlp_regressor(self, train_vecs, y_train, 
                        val_vecs, y_val,
                        hidden_layers=(10, 5), max_iter=300, 
                        alpha=1e-4, learning_rate_init=0.0001, 
                        verbose=True, tol=0.00001, solver="adam", 
                        activation="logistic", pca_variance=0.95):
        regr = MLPRegressor(hidden_layer_sizes=hidden_layers, 
                            max_iter=max_iter, alpha=alpha, 
                            learning_rate_init=learning_rate_init, 
                            verbose=verbose, tol=tol, 
                            solver=solver, activation=activation)
        scaler = StandardScaler()
        scaler.fit(train_vecs)

        train_vecs = scaler.transform(train_vecs)
        val_vecs = scaler.transform(val_vecs)

        pca = PCA(pca_variance)
        pca.fit(train_vecs)

        train_vecs = pca.transform(train_vecs)
        val_vecs = pca.transform(val_vecs)

        regr.fit(train_vecs, y_train)

        train_score = regr.score(train_vecs, y_train)
        val_score = regr.score(val_vecs, y_val)
        loss = regr.loss_

        y_pred = regr.predict(val_vecs)
        val_rmse = sqrt(metrics.mean_squared_error(y_val, y_pred))
        
        y_pred = regr.predict(train_vecs)
        train_rmse = sqrt(metrics.mean_squared_error(y_train, y_pred))

        return {
            "model": regr,
            "score_train": train_score,
            "score_val": val_score,
            "loss": loss,
            "rmse_val": val_rmse,
            "rmse_train": train_rmse
        }

    def linear_regression(self, train_vecs, y_train, val_vecs, y_val):
        reg = LinearRegression(normalize=True).fit(train_vecs, y_train)

        train_score = reg.score(train_vecs, y_train)
        val_score = reg.score(val_vecs, y_val)

        return {
            "model": reg,
            "score_train": train_score,
            "score_val": val_score
        }

def custom_model():
    obj = Models()
    df_train, df_val, df_test = obj.emobank_split()
    X_train, y_train, X_val, y_val = obj.emobank_preprocess(df_train, df_val)
    model_dm, model_dbow = obj.gensim_build_vocab(X_train, X_val)
    
    obj.gensim_train(model_dm, model_dbow, X_train)
    train_vecs = obj.model_vectors(model_dm, model_dbow, X_train)

    obj.gensim_train(model_dm, model_dbow, X_val)
    val_vecs = obj.model_vectors(model_dm, model_dbow, X_val)

    # print(obj.mlp_regressor(train_vecs, y_train, val_vecs, y_val))
    print(obj.linear_regression(train_vecs, y_train, val_vecs, y_val))

if __name__ == "__main__":
    custom_model()