import pandas as pd
from ast import literal_eval
import sys
import glob

positive = ["fun",
"exciting",
"happy",
"joy",
"cheerful",
"love",
"lovely"]

negative = ["sentimental",
"melancholy",
"sad",
"depressing",
"mellow",
"terrible",
"shock",
"hate"
]

all_emotions = ["fun",
"exciting",
"happy",
"joy",
"cheerful",
"love",
"lovely",
"sentimental",
"melancholy",
"sad",
"depressing",
"mellow",
"terrible",
"shock",
"hate"
]

def how_many(name, emotion):
    count = 0
    for i in range(len(emotion)):
        try:
            if literal_eval(emotion.iloc[i])[name] >= 5:
                count += literal_eval(emotion.iloc[i])[name]
        except:
            continue
    return count

def get_pos_neg_books(filepath):
    book_ids = [file.replace(".csv", "").replace("../books/predicted/", "") for file in glob.glob("../books/predicted/*.csv")]

    pos_neg = list()
    for book_id in sorted(book_ids):
        print("Scanning Book: {}".format(book_id))
        df = pd.read_csv(open("../books/predicted/" + book_id + ".csv"))
        emotion = df["emotions"]
        book_emotions = list()

        for e in all_emotions:
            book_emotions.append((e, how_many(e, emotion)))
        
        emotion_df = pd.DataFrame(book_emotions, columns=["emotion", "frequency"])
        emotion_df["percentage"] = (emotion_df["frequency"]/(emotion_df["frequency"].sum()))*100

        positive_ids = [emotion_df[emotion_df["emotion"] == emotion].index[0] for emotion in positive]
        negative_ids = [emotion_df[emotion_df["emotion"] == emotion].index[0] for emotion in negative]

        pos = emotion_df.iloc[positive_ids]["percentage"].sum()
        neg = emotion_df.iloc[negative_ids]["percentage"].sum()

        pos_neg.append([book_id, pos, neg])

    pos_neg_df = pd.DataFrame(pos_neg, columns=["book_id", "positive", "negative"])
    pos_neg_df.to_csv(filepath, encoding='utf-8', index=False)

if __name__ == "__main__":
    get_pos_neg_books("./pos_neg.csv")