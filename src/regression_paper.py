from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt

import pandas as pd
import json
import nltk

# PATHS
NRC_LEMMA_PATH = "../lemma/NRC-VAD-Lexicon.csv"
BRM_LEMMA_PATH = "../lemma/BRM-emot-submit.csv"
WARRINER_LEMMA_PATH = "../lemma/LexiconWarriner2013_transformed.csv"
EMOBANK_PATH = "../corpus/emobank.csv"
BOOKS_PATH = "../books/books/"

# LEMMA RANGES
BRM_LEMMA_RANGE = {"min": 1, "max": 9}
NRC_LEMMA_RANGE = {"min": 0, "max": 1}
WARRINER_LEMMA_RANGE = {"min": -4, "max": 4}
EMOBANK_RANGE = {"min": 1, "max": 5}

# COLUMNS
BRM_COLS = {"lemma_col": "Word", "valence": "V.Mean.Sum", "arousal": "A.Mean.Sum", "dominance": "D.Mean.Sum"}
WARRINER_COLS = {"lemma_col": "Lemma", "valence": "Valence", "arousal": "Arousal", "dominance": "Dominance"}

class Gutenberg_Emotion:
    def __init__(self):
        pass

    def get_lemma(self, file_name=BRM_LEMMA_PATH):
        """
        Param: file_name - Path of the lemma file (Default: BRM_LEMMA_PATH)

        Returns: The lemma corpus from path file_name. (file should be in CSV)
        """
        df = pd.read_csv(open(file_name))
        return df

    def get_book(self, book_id, folder_name=BOOKS_PATH):
        """
        Param: book_id - Gutenberg id of the book to be returned
               folder_name - Path of the book folder (Default: BOOKS_PATH)

        Returns: Gutenberg JSON file from folder_name corresponding to book_id
        """
        book_path = folder_name + str(book_id) + ".json"
        return json.load(open(book_path))

    def get_corpus(self, file_name):
        """
        Param: file_name - Path of the corpus file

        Returns: The corpus from path file_name. (file should be in CSV)
        """
        df = pd.read_csv(open(file_name))
        return df

    def emobank(self, lemma_choice={"path": BRM_LEMMA_PATH, "range": BRM_LEMMA_RANGE, "cols": BRM_COLS, "scale": 0}):
        """
        Maps the text in emobank to VAD space using method mentioned in
        Paper: Emotion Analysis as a Regression Problem —
               Dimensional Models and Their Implications on Emotion
               Representation and Metrical Evaluation
               Sven Buechel and Udo Hahn
        DOI: 10.3233/978-1-61499-672-9-1114

        Params: lemma_choice (dict) - contains:
                        path - path of the lemma
                        range - range of the lemma in format - {"min": MIN_VALUE, "max": MAX_VALUE}
                        cols - column names in the lemma file according to the format:
                                {"lemma_col": LEMMA_NAME_COL, 
                                 "valence": V_COL, 
                                 "arousal": A_COL, 
                                 "dominance": D_COL}
                        scale - any scaling needed at each step.
                                for BRM - 0
                                for WARRINER - 5

        Returns: A tuple consisting of:
                 Index 0: Dictionary of format - {"document_text": (valence, arousal, dominance)}
                 Index 1: RMSE (root mean squared error) of the documents parsed
        """
        # Fetches the emobank corpus and extracts the training set
        df = self.get_corpus(file_name=EMOBANK_PATH)

        train_df = df[df["split"] == "train"]
        len_train_df = len(train_df)

        lemma = self.get_lemma(file_name=lemma_choice["path"])
        lemma_range = lemma_choice["range"]

        # TF-IDF on emobank training set's text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        tfidf_vectorizer_vector = tfidf_vectorizer.fit_transform(train_df["text"].tolist())
        
        # Stores the final VAD values for each document
        calc_vad = dict()

        rmse_V, rmse_A, rmse_D = 0, 0, 0

        for index in range(len_train_df):
            text = train_df.iloc[index]["text"]
            tokens = nltk.word_tokenize(text)

            # Finding TF-IDF for the current index
            index_vector_tfidfvectorizer = tfidf_vectorizer_vector[index]
            df = pd.DataFrame(index_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

            # Extracts the non-zero TF-IDF indices
            curr_df = df[df["tfidf"] > 0.0]
            curr_indexes = list(curr_df.index)
            
            total_upper, total_lower = (0, 0, 0), 0

            for token in tokens:
                token = token.lower()
                if token in curr_indexes:
                    # Calculating values
                    token_idf = curr_df["tfidf"][token]

                    # Is the token in lemma
                    try:
                        token_lemma_index = lemma[lemma[lemma_choice["cols"]["lemma_col"]] == token].index[0]
                    except:
                        continue

                    token_lemma = {"valence": lemma.iloc[token_lemma_index][lemma_choice["cols"]["valence"]], 
                                    "arousal": lemma.iloc[token_lemma_index][lemma_choice["cols"]["arousal"]], 
                                    "dominance": lemma.iloc[token_lemma_index][lemma_choice["cols"]["dominance"]]}

                    # Refer the paper's page 5, equation 3
                    token_upper = (token_idf*token_lemma["valence"], token_idf*token_lemma["arousal"], token_idf*token_lemma["dominance"])
                    token_lower = token_idf

                    total_upper = tuple(total_upper[x]+token_upper[x] for x in range(3))
                    total_lower += token_lower

                    # print(token, token_idf, token_lemma, token_upper, token_lower, total_upper, total_lower)

            # Dividing total_upper (tuple) with total_lower
            document_vad = tuple((val/total_lower)+lemma_choice["scale"] for val in total_upper if total_lower != 0)

            #Performing scaling from old range to new range
            old_range = BRM_LEMMA_RANGE["max"] - BRM_LEMMA_RANGE["min"]
            new_range = EMOBANK_RANGE["max"] - EMOBANK_RANGE["min"]
            document_vad = tuple((((val - BRM_LEMMA_RANGE["min"]) * new_range) / old_range) + EMOBANK_RANGE["min"] for val in document_vad)

            # storing value
            calc_vad[text] = document_vad

            # print(text, document_vad)

            # calculating RMSE
            if total_lower != 0:
                rmse_V += (train_df.iloc[index]["V"] - document_vad[0])**2
                rmse_A += (train_df.iloc[index]["A"] - document_vad[1])**2
                rmse_D += (train_df.iloc[index]["D"] - document_vad[2])**2

        # IF EXPERIMENTING WITH A SUBSET OF ORIGINAL CORPUS
        # REMEMBER TO MODIFY len_train_df to SUBSET LENGTH
        rmse = (sqrt(rmse_V/len_train_df), sqrt(rmse_A/len_train_df), sqrt(rmse_D/len_train_df))
        return (calc_vad, rmse)

if __name__ == "__main__":
    obj = Gutenberg_Emotion()
    check = obj.emobank()

    print(check[1])