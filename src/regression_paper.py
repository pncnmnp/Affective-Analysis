from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import clean_text_gutentag

import pandas as pd
import json
import nltk
import gc

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
NRC_COLS = {"lemma_col": "Word", "valence": "Valence", "arousal": "Arousal", "dominance": "Dominance"}

class Gutenberg_Emotion:
    def __init__(self):
        # Counter detects the no. of sentences which cannot be split in seconds
        # And are split in minutes
        self.issue_sentences = 0
        self.how_many = 0

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

    def emobank(self, lemma_choice={"path": NRC_LEMMA_PATH, "range": NRC_LEMMA_RANGE, "cols": NRC_COLS, "scale": 0}):
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
            old_range = NRC_LEMMA_RANGE["max"] - NRC_LEMMA_RANGE["min"]
            new_range = EMOBANK_RANGE["max"] - EMOBANK_RANGE["min"]
            document_vad = tuple((((val - NRC_LEMMA_RANGE["min"]) * new_range) / old_range) + EMOBANK_RANGE["min"] for val in document_vad)

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

    def book_formatting(self, text):
        text = clean_text_gutentag.clean_text(text)
        text = sent_tokenize(text)    
        return text

    def read_time(self, text, reading_time=5, in_seconds=False):
        # Average reading time is -
        # https://ezinearticles.com/?What-is-the-Average-Reading-Speed-and-the-Best-Rate-of-Reading?&id=2298503
        words = word_tokenize(text)
        reading_minutes = int(len(words)/200)
        reading_seconds = int((len(words)/200 - reading_minutes)*60)

        # If in_seconds == True, then reading_time is interpreted in seconds
        # Else reading_time is interpreted in minutes
        if in_seconds:
            # NOTE: In worst case scenario is is observed that the senteces can jump from a few seconds like
            # 5 seconds to 2 minutes or more! If this is an issue, consider splitting the sentence into parts.
            if (reading_minutes == 0 and reading_seconds >= reading_time):
                return True
            elif (reading_minutes >= 1):
                self.issue_sentences += 1
                return True
            return False
        elif in_seconds == False:
            if reading_minutes >= reading_time:
                return True
            return False

    def split_text_read_time(self, df, reading_time=5, in_seconds=False, text_col="text"):
        group_sentences = list()
        sentences = df[text_col].tolist()
        start_index, end_index = 0, 1

        while start_index < len(sentences):
            combined = ' '.join(sentences[start_index: end_index])
            if self.read_time(combined, reading_time=reading_time, in_seconds=in_seconds) == False:
                # Check if end of text is reached
                if end_index >= len(sentences):
                    group_sentences.append(combined)
                    break
                # End of text is not reached
                end_index += 1
            else:
                group_sentences.append(combined)
                start_index = end_index
                end_index += 1
        
        return pd.DataFrame(group_sentences, columns=[text_col])

    def gutenberg(self, book_id, reading_time_split=5, in_seconds=False, 
                    lemma_choice={"path": BRM_LEMMA_PATH, 
                                  "range": BRM_LEMMA_RANGE, 
                                  "cols": BRM_COLS, 
                                  "scale": 0}):
        """
        TODO: Remove redundancy by merging code shared by self.gutenberg() and self.emobank() 
        """
        # Fetches the emobank corpus and extracts the training set
        book_info = self.get_book(book_id)
        book = book_info["text"]
        
        # Set issue_sentences counter to 0
        self.issue_sentences = 0

        # Debugging info
        print("Scanning (ID - {}): {} BY {}".format(book_id, book_info["title"][0], book_info["author"][0]))
        if in_seconds:
            print("Text will be split into reading time of {} seconds".format(reading_time_split))
        else:
            print("Text will be split into reading time of {} minutes".format(reading_time_split))

        train_df = pd.DataFrame(self.book_formatting(book), columns=["text"])

        if reading_time_split != None:
            train_df = self.split_text_read_time(train_df, 
                                                reading_time=reading_time_split, 
                                                in_seconds=in_seconds)

        # Debug info
        print("Text splitting is completed.\n{} sentences were split in minutes.".format(self.issue_sentences))

        len_train_df = len(train_df)

        lemma = self.get_lemma(file_name=lemma_choice["path"])
        lemma_range = lemma_choice["range"]

        # TF-IDF on emobank training set's text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        tfidf_vectorizer_vector = tfidf_vectorizer.fit_transform(train_df["text"].tolist())
        
        # Stores the final VAD values for each document
        calc_vad = dict()

        rmse_V, rmse_A, rmse_D = 0, 0, 0

        progress = 0.01
        for index in range(len_train_df):
            # Checking and returning progress
            if (index/len_train_df) > progress:
                print("Progress: {}%\r".format(round(progress*100), 2), end="")
                progress += 0.01

            text = train_df.iloc[index]["text"]
            tokens = nltk.word_tokenize(text)

            # Finding TF-IDF for the current index
            index_vector_tfidfvectorizer = tfidf_vectorizer_vector[index]
            df = pd.DataFrame(index_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

            # Extracts the non-zero TF-IDF indices
            curr_df = df[df["tfidf"] > 0.0]
            curr_indexes = list(curr_df.index)
            
            total_upper, total_lower = (0, 0, 0), 0
            lemmatizer = WordNetLemmatizer()

            for token in tokens:
                token = token.lower()
                if token in curr_indexes:
                    # Calculating values
                    token_idf = curr_df["tfidf"][token]

                    # Is the token in lemma
                    try:
                        token_lemma_index = lemma[lemma[lemma_choice["cols"]["lemma_col"]] == token].index[0]
                    except:
                        try:
                            token_lemma_index = lemma[lemma[lemma_choice["cols"]["lemma_col"]] == lemmatizer.lemmatize(token)].index[0]
                            self.how_many += 1
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

            # storing value
            calc_vad[text] = document_vad

            # print(text, document_vad)

        print(self.how_many)

        return (calc_vad)

    def recommendation(self, answers, reading_time_split=15, in_seconds=True, 
                    lemma_choice={"path": BRM_LEMMA_PATH, 
                                  "range": BRM_LEMMA_RANGE, 
                                  "cols": BRM_COLS, 
                                  "scale": 0}):
        book = answers
        
        # Set issue_sentences counter to 0
        self.issue_sentences = 0

        train_df = pd.DataFrame(sent_tokenize(book), columns=["text"])

        if reading_time_split != None:
            train_df = self.split_text_read_time(train_df, 
                                                reading_time=reading_time_split, 
                                                in_seconds=in_seconds)

        len_train_df = len(train_df)
        # print(train_df)

        lemma = self.get_lemma(file_name=lemma_choice["path"])
        lemma_range = lemma_choice["range"]

        # TF-IDF on emobank training set's text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        tfidf_vectorizer_vector = tfidf_vectorizer.fit_transform(train_df["text"].tolist())
        
        # Stores the final VAD values for each document
        calc_vad = dict()

        rmse_V, rmse_A, rmse_D = 0, 0, 0

        progress = 0.01
        for index in range(len_train_df):
            # Checking and returning progress
            if (index/len_train_df) > progress:
                print("Progress: {}%\r".format(round(progress*100), 2), end="")
                progress += 0.01

            text = train_df.iloc[index]["text"]
            tokens = nltk.word_tokenize(text)

            # Finding TF-IDF for the current index
            index_vector_tfidfvectorizer = tfidf_vectorizer_vector[index]
            df = pd.DataFrame(index_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

            # Extracts the non-zero TF-IDF indices
            curr_df = df[df["tfidf"] > 0.0]
            curr_indexes = list(curr_df.index)
            
            total_upper, total_lower = (0, 0, 0), 0
            lemmatizer = WordNetLemmatizer()

            for token in tokens:
                token = token.lower()
                if token in curr_indexes:
                    # Calculating values
                    token_idf = curr_df["tfidf"][token]

                    # Is the token in lemma
                    try:
                        token_lemma_index = lemma[lemma[lemma_choice["cols"]["lemma_col"]] == token].index[0]
                    except:
                        try:
                            token_lemma_index = lemma[lemma[lemma_choice["cols"]["lemma_col"]] == lemmatizer.lemmatize(token)].index[0]
                            self.how_many += 1
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

            # Dividing total_upper (tuple) with total_lower
            document_vad = tuple((val/total_lower)+lemma_choice["scale"] for val in total_upper if total_lower != 0)

            # storing value
            calc_vad[text] = document_vad

        return (calc_vad)

def save_gutenberg_emotions(stats_path, full_para_path, ids):
    obj = Gutenberg_Emotion()
    # for gid in ids[146:]:
    for gid in [11]:
        check = obj.gutenberg(gid, reading_time_split=1, in_seconds=True)
        emotions = list(check.values())
        gutenberg_emotion_df = pd.DataFrame(emotions, columns=["valence", "arousal", "dominance"])

        # storing full paragraph-emotion data
        with open(full_para_path+str(gid)+'_sentences.json', 'w') as f:
            json.dump(check, f)
        
        # storing book emotion stats
        gutenberg_emotion_df.to_csv(stats_path+str(gid)+"_sentences.csv", index=False, encoding='utf-8')

        gc.collect()

if __name__ == "__main__":
    # obj = Gutenberg_Emotion()
    # check = obj.gutenberg(11, reading_time_split=15, in_seconds=True)
    # emotions = list(check.values())
    # gutenberg_emotion_df = pd.DataFrame(emotions, columns=["valence", "arousal", "dominance"])

    # print(gutenberg_emotion_df.describe())
    GUTENBERG_ID_PATH = "../books/gutenberg_book_ids.json"
    GUTENBERG_STATS_DIRECTORY = "../books/stats/"
    GUTENBERG_COMPLETE_DIRECTORY = "../books/complete/"
    save_gutenberg_emotions(GUTENBERG_STATS_DIRECTORY, GUTENBERG_COMPLETE_DIRECTORY, json.load(open(GUTENBERG_ID_PATH)))