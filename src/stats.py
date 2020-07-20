import pandas as pd
from ast import literal_eval
import sys

df = pd.read_csv(open("../books/predicted/" + sys.argv[1] + ".csv"))
# df = pd.read_csv(open("./1777_random_forest.csv"))
emotion = df["emotions"]

def how_many(name):
    count = 0
    for i in range(len(emotion)):
        try:
            if literal_eval(emotion.iloc[i])[name] >= 5:
                count += literal_eval(emotion.iloc[i])[name]
            # d = literal_eval(emotion.iloc[i])
            # if max(d, key=d.get) == name:
            #     count += 1
        except:
            continue
    return count    

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

book_emotions = list()

for e in all_emotions:
    book_emotions.append((e, how_many(e)))

emotion_df = pd.DataFrame(book_emotions, columns=["emotion", "frequency"])
emotion_df["percentage"] = (emotion_df["frequency"]/(emotion_df["frequency"].sum()))*100
print(emotion_df.sort_values('frequency', ascending=False))

positive_ids = [emotion_df[emotion_df["emotion"] == emotion].index[0] for emotion in positive]
print("Positive: {}".format(emotion_df.iloc[positive_ids]["percentage"].sum()))

negative_ids = [emotion_df[emotion_df["emotion"] == emotion].index[0] for emotion in negative]
print("Negative: {}".format(emotion_df.iloc[negative_ids]["percentage"].sum()))