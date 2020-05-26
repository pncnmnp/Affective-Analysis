from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt

import pandas as pd
import json
import nltk

NRC_LEMMA_PATH = "../lemma/NRC-VAD-Lexicon.csv"
BRM_LEMMA_PATH = "../lemma/BRM-emot-submit.csv"
EMOBANK_PATH = "../corpus/emobank.csv"
BOOKS_PATH = "../books/books/"

BRM_LEMMA_RANGE = {"min": 1, "max": 9}
NRC_LEMMA_RANGE = {"min": 0, "max": 9}

EMOBANK_RANGE = {"min": 1, "max": 5}

BRM_COLS = {"valence": "V.Mean.Sum", "arousal": "A.Mean.Sum", "domination": "D.Mean.Sum"}

class Gutenberg_Emotion:
    def __init__(self):
        pass

    def get_lemma(self, file_name=BRM_LEMMA_PATH):
        df = pd.read_csv(open(file_name))
        return df

    def get_book(self, book_id, folder_name=BOOKS_PATH):
        book_path = folder_name + str(book_id) + ".json"
        return json.load(open(book_path))

    def get_corpus(self, file_name):
        df = pd.read_csv(open(file_name))
        return df

    def emobank(self):
        df = self.get_corpus(file_name=EMOBANK_PATH)

        train_df = df[df["split"] == "train"]
        len_train_df = len(train_df)

        lemma = self.get_lemma(file_name=BRM_LEMMA_PATH)
        lemma_range = BRM_LEMMA_RANGE

        tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        tfidf_vectorizer_vector = tfidf_vectorizer.fit_transform(train_df["text"].tolist())
        
        calc_vad = dict()

        rmse_V, rmse_A, rmse_D = 0, 0, 0

        for index in range(len_train_df):
            text = train_df.iloc[index]["text"]
            tokens = nltk.word_tokenize(text)

            index_vector_tfidfvectorizer = tfidf_vectorizer_vector[index]
            df = pd.DataFrame(index_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

            curr_df = df[df["tfidf"] > 0.0]
            curr_indexes = list(curr_df.index)
            
            total_upper, total_lower = (0, 0, 0), 0

            for token in tokens:
                token = token.lower()
                if token in curr_indexes:
                    # Calculating values
                    token_idf = curr_df["tfidf"][token]

                    try:
                        token_lemma_index = lemma[lemma["Word"] == token].index[0]
                    except:
                        continue

                    token_lemma = {"valence": lemma.iloc[token_lemma_index]["V.Mean.Sum"], 
                                    "arousal": lemma.iloc[token_lemma_index]["A.Mean.Sum"], 
                                    "dominance": lemma.iloc[token_lemma_index]["D.Mean.Sum"]}

                    token_upper = (token_idf*token_lemma["valence"], token_idf*token_lemma["arousal"], token_idf*token_lemma["dominance"])
                    token_lower = token_idf

                    total_upper = tuple(total_upper[x]+token_upper[x] for x in range(3))
                    total_lower += token_lower

                    # print(token, token_idf, token_lemma, token_upper, token_lower, total_upper, total_lower)

            document_vad = tuple(val/total_lower for val in total_upper if total_lower != 0)

            #Performing scaling from old range to new range
            old_range = BRM_LEMMA_RANGE["max"] - BRM_LEMMA_RANGE["min"]
            new_range = EMOBANK_RANGE["max"] - EMOBANK_RANGE["min"]
            document_vad = tuple((((val - BRM_LEMMA_RANGE["min"]) * new_range) / old_range) + EMOBANK_RANGE["min"] for val in document_vad)

            # storing value
            calc_vad[text] = document_vad

            if total_lower != 0:
                rmse_V += (train_df.iloc[index]["V"] - document_vad[0])**2
                rmse_A += (train_df.iloc[index]["A"] - document_vad[1])**2
                rmse_D += (train_df.iloc[index]["D"] - document_vad[2])**2

        rmse = (sqrt(rmse_V/len_train_df), sqrt(rmse_A/len_train_df), sqrt(rmse_D/len_train_df))
        return (calc_vad, rmse)

if __name__ == "__main__":
    obj = Gutenberg_Emotion()
    check = obj.emobank()

    print(check[1])