import pandas as pd
from scipy.stats import truncnorm

# To see how this mapping was derived, see - 
# Affect representation and recognition in 3D continuous valence–arousal–dominance space
# Gyanendra K Verma & Uma Shanker Tiwary
# DOI: https://doi.org/10.1007/s11042-015-3119-y
EMOTION_VAD_MAP_PATH = "../lemma/emotions.csv"

# Trying both truncated normal and non-truncated normal

class Categorization:
    def __init__(self):
        pass

    def get_mapping(self, filename=EMOTION_VAD_MAP_PATH):
        return pd.read_csv(open(EMOTION_VAD_MAP_PATH))

    def get_truncated_normal(self, mean, sd, low, up, length):
        return truncnorm(((low-mean)/sd), 
                         ((up-mean)/sd), 
                          loc = mean, scale = sd).rvs(length)

    def generate_random_category(self, category, V_mean, V_std, 
                                    A_mean, A_std, D_mean, 
                                    D_std, length=5):
        V = self.get_truncated_normal(V_mean, V_std, 
                                    V_mean - V_std, V_mean + V_std, length)
        A = self.get_truncated_normal(A_mean, A_std, 
                                    A_mean - A_std, A_mean + A_std, length)
        D = self.get_truncated_normal(D_mean, D_std, 
                                    D_mean - D_std, D_mean + D_std, length)
        
        X_points = [(x, y, z) for x, y, z in zip(V, A, D)]
        y_categories = [category]*length

        return (X_points, y_categories)

    def training_set(self, df):
        combined_X, combined_y = list(), list()

        for index in range(len(df)):
            emotion = df.iloc[index]
            X, y = self.generate_random_category(*tuple(emotion.tolist()))
            combined_X.extend(X)
            combined_y.extend(y)

        return (combined_X, combined_y)

if __name__ == "__main__":
    obj = Categorization()
    print(obj.training_set(obj.get_mapping()))