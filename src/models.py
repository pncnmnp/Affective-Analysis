import gensim
import pandas as pd
import numpy as np

from math import sqrt

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn import metrics

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR, NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import regression_paper

class Models:
    def __init__(self):
        self.reg_paper_obj = regression_paper.Gutenberg_Emotion()

    def emobank_split(self, corpus_path=regression_paper.EMOBANK_PATH):
        """
        Splits the emobank corpus into train, dev and test split.
        The split is done based on the "split" column in the dataset.

        Params: corpus_path - Path of the emobank corpus
                             Default - EMOBANK_PATH in regression_paper.py

        Returns: Tuple with 3 indexes - 
                 index 0: Pandas dataframe for train set 
                 index 1: Pandas dataframe for dev set 
                 index 2: Pandas dataframe for test set 
        """
        df = self.reg_paper_obj.get_corpus(corpus_path)
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "dev"]
        df_test = df[df["split"] == "test"]

        return (df_train, df_val, df_test)

    def clean_text(self, corpus):
        """
        Formats the corpus by - 
            * stripping whitespaces
            * text to lowercase
            * Replace '\n' with ''
            * separating out each punctuation
            * splitting out each word

        Params: corpus - Corpus to be formatted

        Returns: List of lists, with each sub-list containing the
                 list of processed words.
        """
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
        """
        Adds the Gensim text labels required for doc2vec

        Params: text - preprocessed output from clean_text()
               label_type - string containing label name

        Returns: Labelized list of text
        """
        LabeledSentence = gensim.models.doc2vec.LabeledSentence
        labelized = list()
        for i, v in enumerate(text):
            label = '%s_%s'%(label_type, i)
            labelized.append(LabeledSentence(v, [label]))

        return labelized

    def emobank_preprocess(self, df_train, df_val, emotion="V"):
        """
        Preprocessing of the text in emobank corpus

        Params: df_train - Pandas Dataframe of training set
               df_val - Pandas Dataframe of validation/dev set
               emotion - Type of emotion to perform training on
                         (i.e. V - Valence
                               A - Arousal
                               D - Dominance)
                         Default - V

        Returns: Tuple with 4 indexes - 
                 index 0 (X_train) - preprocessed training text 
                 index 1 (y_train) - Emotions corresponding to training data
                 index 2 (X_val) - preprocessed validation text
                 index 3 (y_val) - Emotions corresponding to validation data
        """
        X_train, y_train = df_train["text"], df_train[emotion]
        X_val, y_val = df_val["text"], df_val[emotion]

        X_train = self.labelize(self.clean_text(X_train), 'TRAIN')
        X_val = self.labelize(self.clean_text(X_val), 'DEV')

        return (X_train, y_train, X_val, y_val)

    def text_preprocess(self, text):
        X = self.labelize(self.clean_text(text), 'TEST')
        return X

    def gensim_build_vocab(self, X_train, X_val=None, size=300, window=4, 
                            negative=2, workers=3, sample=1e-3, min_count=1):
        """
        Returns the Distributed Memory (DM) and Distributed Bag of Words 
        (DBOW) models of Doc2Vec.

        Params: X_train - Preprocessed training text
                X_val - Preprocessed validation/dev text
                For the rest of parameter info see - 
                https://radimrehurek.com/gensim/models/doc2vec.html
        
        Returns: Tuple with 2 indexes - 
                 index 0 (model_dm) - Returns DM model
                 index 1 (model_dbow) - Returns DBOW model
        """
        model_dm = gensim.models.Doc2Vec(min_count=min_count, window=window, 
                                        size=size, sample=sample, 
                                        negative=negative, workers=workers)

        model_dbow = gensim.models.Doc2Vec(min_count=min_count, window=window, 
                                        size=size, sample=sample, 
                                        negative=negative, dm=0, 
                                        workers=workers)

        if X_val == None:
            model_dm.build_vocab(X_train)
            model_dbow.build_vocab(X_train)
        else:
            model_dm.build_vocab(X_train + X_val)
            model_dbow.build_vocab(X_train + X_val)

        return (model_dm, model_dbow)

    def gensim_train(self, model_dm, model_dbow, X, epochs=20):
        """
        Training the Distributed Memory (DM) and Distributed Bag of Words 
        (DBOW) models.

        Params: model_dm - Distributed Memory model
                model_dbow - Distributed Bag of Words model
                X - Preprocessed text on which training is to be done
                epochs - No. of iterations over the corpus
                         Default -  20

        Returns: Tuple with 2 indexes - 
                 index 0 (model_dm) - Returns trained DM model
                 index 1 (model_dbow) - Returns trained DBOW model
        """
        model_dm.train(X, total_examples = len(X), epochs=epochs)
        model_dbow.train(X, total_examples = len(X), epochs=epochs)

        return (model_dm, model_dbow)

    def gensim_vectors(self, model, corpus, size):
        """
        To get the Gensim vectors of a model.

        Params: model - Trained model (DM or DBOW model)
                corpus - Corpus on which the model was trained
                size - Reshape size

        Returns: Numpy array of vectors corresponding 
                 the model and corpus
        """
        vecs = [np.array(model[z.tags[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    def model_vectors(self, model_dm, model_dbow, X, size=300):
        """
        To get the Gensim vectors of DM and DBOW models.

        Params: model_dm - Distributed Memory model
                model_dbow - Distributed Bag of Words model
                X - Preprocessed text on which training is to be done
                size - Reshape size
        
        Returns: Numpy array containing combined vectors
                 of both DM and DBOW models of length - size*2
        """
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
        """
        Multilayer Perceptron model for training emotions.

        Params: train_vecs - Vectors of training set
                y_train - Emotions corresponding to training data
                val_vecs - Vectors of validation/dev set
                y_val - Emotions corresponding to validation/dev data

                For the rest of parameters see - 
                https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

        Returns: Dictionary of format - 
                 {
                     "model": Trained model,
                     "score_train": Score of model on training set
                     "score_val": Score of model on validation set
                     "loss": Final loss of the model
                             (before the iteration stops)
                     "rmse_val": RMSE (Root Mean Square Error) on
                                 validation set
                     "rmse_train": RMSE (Root Mean Square Error) on
                                 training set
                 }
        """
        regr = MLPRegressor(hidden_layer_sizes=hidden_layers, 
                            max_iter=max_iter, alpha=alpha, 
                            learning_rate_init=learning_rate_init, 
                            verbose=verbose, tol=tol, 
                            solver=solver, activation=activation)
        scaler = StandardScaler()
        scaler.fit(train_vecs)

        # Mean Normalization
        train_vecs = scaler.transform(train_vecs)
        val_vecs = scaler.transform(val_vecs)

        # Principal Component Analysis
        pca = PCA(pca_variance)
        pca.fit(train_vecs)

        # Reducing dimensions
        train_vecs = pca.transform(train_vecs)
        val_vecs = pca.transform(val_vecs)

        # Training model
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
        """
        Linear Regression model for training emotions.

        For more details on base model, see - 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        
        Params: train_vecs - Vectors of training set
                y_train - Emotions corresponding to training data
                val_vecs - Vectors of validation/dev set
                y_val - Emotions corresponding to validation/dev data

        Returns: Dictionary of format - 
                 {
                     "model": Trained model,
                     "score_train": Score of model on training set
                     "score_val": Score of model on validation set
                 }
        """
        reg = LinearRegression(normalize=True).fit(train_vecs, y_train)

        train_score = reg.score(train_vecs, y_train)
        val_score = reg.score(val_vecs, y_val)

        return {
            "model": reg,
            "score_train": train_score,
            "score_val": val_score
        }

    def svr(self, train_vecs, y_train, val_vecs, y_val, 
                kernel="rbf", C=0.8, epsilon=0.2):
        """
        Support Vector Regression model for training emotions.

        Params: train_vecs - Vectors of training set
                y_train - Emotions corresponding to training data
                val_vecs - Vectors of validation/dev set
                y_val - Emotions corresponding to validation/dev data

                For the rest of parameters see - 
                https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        Returns: Dictionary of format - 
                 {
                     "model": Trained model,
                     "score_train": Score of model on training set
                     "score_val": Score of model on validation set
                     "rmse_val": RMSE (Root Mean Square Error) on
                                 validation set
                     "rmse_train": RMSE (Root Mean Square Error) on
                                 training set
                 }
        """
        # TODO: Add PCA
        svr_reg = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon))
        svr_reg.fit(train_vecs, y_train)

        train_score = svr_reg.score(train_vecs, y_train)
        val_score = svr_reg.score(val_vecs, y_val)
        
        svr_y_pred = svr_reg.predict(train_vecs)
        train_rmse = sqrt(metrics.mean_squared_error(y_train, svr_y_pred))

        svr_y_pred = svr_reg.predict(val_vecs)
        val_rmse = sqrt(metrics.mean_squared_error(y_val, svr_y_pred))

        return {
            "model": svr_reg,
            "score_train": train_score,
            "score_val": val_score,
            "rmse_val": val_rmse,
            "rmse_train": train_rmse
        }

    def nu_svr(self, train_vecs, y_train, val_vecs, y_val, 
                C=1.0, nu=0.1):
        """
        Nu Support Vector Regression model for training emotions.

        Params: train_vecs - Vectors of training set
                y_train - Emotions corresponding to training data
                val_vecs - Vectors of validation/dev set
                y_val - Emotions corresponding to validation/dev data

                For the rest of parameters see - 
                https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html

        Returns: Dictionary of format - 
                 {
                     "model": Trained model,
                     "score_train": Score of model on training set
                     "score_val": Score of model on validation set
                     "rmse_val": RMSE (Root Mean Square Error) on
                                 validation set
                     "rmse_train": RMSE (Root Mean Square Error) on
                                 training set
                 }
        """
        # TODO: Add PCA
        nu_svr_reg = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))
        nu_svr_reg.fit(train_vecs, y_train)

        train_score = nu_svr_reg.score(train_vecs, y_train)
        val_score = nu_svr_reg.score(val_vecs, y_val)

        nu_svr_y_pred = nu_svr_reg.predict(train_vecs)
        train_rmse = sqrt(metrics.mean_squared_error(y_train, nu_svr_y_pred))

        nu_svr_y_pred = nu_svr_reg.predict(val_vecs)
        val_rmse = sqrt(metrics.mean_squared_error(y_val, nu_svr_y_pred))

        return {
            "model": nu_svr_reg,
            "score_train": train_score,
            "score_val": val_score,
            "rmse_val": val_rmse,
            "rmse_train": train_rmse
        }

def custom_model():
    obj = Models()
    df_train, df_val, df_test = obj.emobank_split()
    X_train, y_train, X_val, y_val = obj.emobank_preprocess(df_train, df_val)
    model_dm, model_dbow = obj.gensim_build_vocab(X_train, X_val=X_val)
    
    obj.gensim_train(model_dm, model_dbow, X_train)
    train_vecs = obj.model_vectors(model_dm, model_dbow, X_train)

    obj.gensim_train(model_dm, model_dbow, X_val)
    val_vecs = obj.model_vectors(model_dm, model_dbow, X_val)

    # print(obj.mlp_regressor(train_vecs, y_train, val_vecs, y_val))
    # print(obj.linear_regression(train_vecs, y_train, val_vecs, y_val))
    # print(obj.svr(train_vecs, y_train, val_vecs, y_val))
    print(obj.nu_svr(train_vecs, y_train, val_vecs, y_val))

def predict_gutenberg(book_id, stats_path="../books/stats/"):
    model_obj = Models()

    df_train, df_val, df_test = model_obj.emobank_split()
    X_train, y_train, X_val, y_val = model_obj.emobank_preprocess(df_train, df_val)
    model_dm, model_dbow = model_obj.gensim_build_vocab(X_train, X_val=X_val)

    model_obj.gensim_train(model_dm, model_dbow, X_train)
    train_vecs = model_obj.model_vectors(model_dm, model_dbow, X_train)

    model_obj.gensim_train(model_dm, model_dbow, X_val)
    val_vecs = model_obj.model_vectors(model_dm, model_dbow, X_val)

    svr = model_obj.svr(train_vecs, y_train, val_vecs, y_val)

    reg_obj = regression_paper.Gutenberg_Emotion()
    book_info = reg_obj.get_book(book_id)
    book = book_info["text"]
    train_df = pd.DataFrame(reg_obj.book_formatting(book), columns=["text"])

    train_df = reg_obj.split_text_read_time(train_df, 
                                        reading_time=10, 
                                        in_seconds=True)

    model = Models()
    X_book = model.text_preprocess(train_df["text"])
    model_dm, model_dbow = model.gensim_build_vocab(X_book)
    model.gensim_train(model_dm, model_dbow, X_book)
    book_vecs = model.model_vectors(model_dm, model_dbow, X_book)
    book_predict = svr["model"].predict(book_vecs)

    book_predict_df = pd.DataFrame(book_predict)
    book_predict_df.to_csv(stats_path+str(book_id)+"_ml.csv", index=False, encoding='utf-8')
    print(book_predict_df.describe())

if __name__ == "__main__":
    predict_gutenberg(11)