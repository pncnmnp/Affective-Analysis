from scipy.stats import truncnorm
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

import pandas as pd
import regression_paper

# To see how this mapping was derived, see - 
# Affect representation and recognition in 3D continuous valence–arousal–dominance space
# Gyanendra K Verma & Uma Shanker Tiwary
# DOI: https://doi.org/10.1007/s11042-015-3119-y
EMOTION_VAD_MAP_PATH = "../lemma/emotions.csv"

# TODO: Try both truncated normal and non-truncated normal

class Categorization:
    def __init__(self):
        self.reg_paper_obj = regression_paper.Gutenberg_Emotion()

    def get_and_scale_emobank(self, filename=regression_paper.EMOBANK_PATH):
        """
        Returns Emobank corpus as a Pandas Dataframe.
        The V,A,D values in Emobank corpus are scaled from 1-5 to 1-9.

        Params: filename - Path of the emobank corpus
                           (Default - EMOBANK_PATH in regression_paper.py)

        Returns: df - Pandas Dataframe containing Emobank corpus
        """
        df = self.reg_paper_obj.get_corpus(filename)
        emo_min, emo_max, map_min, map_max = 1, 5, 1, 9
        old_range = (emo_max - emo_min)
        new_range = (map_max - map_min)

        df["V"] = df["V"].apply(lambda x: (((x - emo_min) * new_range) / old_range) + map_min)
        df["A"] = df["A"].apply(lambda x: (((x - emo_min) * new_range) / old_range) + map_min)
        df["D"] = df["D"].apply(lambda x: (((x - emo_min) * new_range) / old_range) + map_min)

        return df

    def map_emobank(self, models=10, train_length=10000, to_save=False, filename="model.csv"):
        """
        Predict the emotions present in each text in Emobank corpus.
        Returns Emobank corpus as a Pandas Dataframe.
        Dataframe contains a new column - "emotion", where for each index, 
        emotion is a Counter dictionary (from library - "collections").

        The Counter dictionary contains the no. of times an emotion was predicted by a model for a particular text.
        For example: Text - "Remember what she said in my last letter?"
                     can have predicted emotions like - {'sentimental': 10, 'terrible': 4, 'exciting': 5, 'shock': 19, 'hate': 8, 'joy': 4}
                     (for models=50)

                     Above result can be interpreted as - Out of 50 trained models (models are trained using KNN),
                     19 models thought the emotion in the text was 'shock'
                     10 models thought the emotion was 'sentimental', etc.

        Implementation Details: X different models (where X = models(parameter) ) are trained on KNN.
                                For each KNN model, 
                                    New training data is generated using method - training_set().
                                    The length of the training data equals train_length(parameter).

                                Then all the trained KNN models are tested on each text to predict text's emotion.
                                Counter of the predictions is then inserted in the "emotions" column
        
        Params: models - No. of KNN models to train
                         (Default - 10)
                train_length - Length of the training corpus
                               (Default - 10000)
                to_save - If False, will return a Pandas Dataframe of emobank corpus with predictions
                          If True, will save the Pandas Dataframe  of emobank corpus in CSV format
                          (Default - False)
                filename - Filename for saving the model
                           Is used if to_save == True
                           (Default - "model.csv")
        
        Returns: Pandas Dataframe of the emobank corpus
        """
        # Training X different models (X = models)
        trained_models = list()
        for index in range(models):
            X, y = self.training_set(df=self.get_mapping(), length=train_length)
            neigh = self.train(X, y, 5, 'uniform')
            trained_models.append(neigh)

        df = self.get_and_scale_emobank()
        # Add a column 'emotions' with None values
        df["emotions"] = pd.Series("", index=df.index)

        for index in range(len(df)):
            emotion = list()
            for e in range(models):
                vals = (df.iloc[index]["V"], df.iloc[index]["A"], df.iloc[index]["D"])
                emotion += list(trained_models[e].predict([vals]))
            df.at[index, "emotions"] = dict(Counter(emotion))

        if to_save == True:
            df.to_csv(filename, encoding='utf-8', index=False)
        else:
            return df

    def get_mapping(self, filename=EMOTION_VAD_MAP_PATH):
        """
        Returns the mapping of emotions to the VAD space.
        To see how this mapping was derived, see - 
        Affect representation and recognition in 3D continuous valence–arousal–dominance space
        Gyanendra K Verma & Uma Shanker Tiwary
        DOI: https://doi.org/10.1007/s11042-015-3119-y

        (The file should be in CSV format)

        Params: filename - Path of mapping file
                           (Default - EMOTION_VAD_MAP_PATH)
        
        Returns: Pandas Dataframe of the mapping
        """
        return pd.read_csv(open(EMOTION_VAD_MAP_PATH))

    def get_truncated_normal(self, mean, sd, low, up, length):
        """
        Returns a list of truncated normal continuous random variables.
        For more info, see - 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

        Params: mean - mean value
                std - standard deviation value
                low - lower bound value
                up - upper bound value
                length - no. of random values to be returned

        Returns: List of random values
        """
        return truncnorm(((low-mean)/sd), 
                         ((up-mean)/sd), 
                          loc = mean, scale = sd).rvs(length)

    def generate_random_category(self, category, V_mean, V_std, 
                                    A_mean, A_std, D_mean, 
                                    D_std, length=10000):
        """
        Generates training set for a particular emotion.
        It uses the mean and standard deviation of VAD space (corresponding to the emotion),
        to generate a random training set.

        For each space (V, A, D), it generates a list of truncated normal continuous random variables.
        (using method - get_truncated_normal())

        For get_truncated_normal():
            The lower bound is (V/A/D)_mean - (V/A/D)_std
            The upper bound is (V/A/D)_mean + (V/A/D)_std
        So the values are predicted from the highly dense 68% space (in a Gaussian graph).

        Params: category - Name of the emotion
                V_mean - mean of Valence space for emotion in the 'category'
                V_std - standard deviation of Valence space for emotion in the 'category'
                A_mean - mean of Arousal space for emotion in the 'category'
                A_std - standard deviation of Arousal space for emotion in the 'category'
                D_mean - mean of Dominance space for emotion in the 'category'
                D_std - standard deviation of Dominance space for emotion in the 'category'
                length - size of the training set
                         (Default - 10000)

        Returns: Tuple with two indexes - 
                 index 0: X_points - Generated training set (X) of the emotion
                 index 1: y_categories - Labels (Y) of the training set of the emotion
        """
        V = self.get_truncated_normal(V_mean, V_std, 
                                    V_mean - V_std, V_mean + V_std, length)
        A = self.get_truncated_normal(A_mean, A_std, 
                                    A_mean - A_std, A_mean + A_std, length)
        D = self.get_truncated_normal(D_mean, D_std, 
                                    D_mean - D_std, D_mean + D_std, length)
        
        X_points = [(x, y, z) for x, y, z in zip(V, A, D)]
        y_categories = [category]*length

        return (X_points, y_categories)

    def training_set(self, df, length=10000):
        """
        Generates an entire training set for all emotions.

        Params: df - Pandas Dataframe as returned by method get_mapping()
                length - Length of each emotion's training set

        Returns: Tuple with two indexes - 
                 index 0: combined_X - Generated training set (X) of all emotion
                 index 1: combined_y - Labels (Y) of the training set of all emotion
        """
        combined_X, combined_y = list(), list()

        for index in range(len(df)):
            emotion = df.iloc[index]
            X, y = self.generate_random_category(*tuple(emotion.tolist()), length=10000)
            combined_X.extend(X)
            combined_y.extend(y)

        return (combined_X, combined_y)

    def train(self, X, y, n_neighbors, weights):
        """
        Trains a KNN model.

        Params: X - Generated training set
                y - Labels of the training set
                For the rest of the parameters, see: 
                https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        Returns: Trained KNN model
        """
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        neigh.fit(X, y)
        return neigh

if __name__ == "__main__":
    obj = Categorization()
    obj.map_emobank(models=100, train_length=1000000, to_save=True)
    # X, y = obj.training_set(obj.get_mapping())
    # neigh = obj.train(X, y, 5, 'uniform')
    # print(neigh.classes_)
    # print(neigh.predict_proba([(5.88, 5, 5.44)]))
    # print(neigh.predict([(5.88, 5, 5.44)]))