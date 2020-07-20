from flask import Flask, render_template, request, redirect, abort
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import pandas as pd
import regression_paper
import categories
import json
import glob

app = Flask(__name__)

class Answers:
    def __init__(self):
        self.mood_para = None
        self.user_moods = None

answers = Answers()

@app.route('/')
def index():
	return render_template('questionnaire.html')

@app.route('/alternative')
def alternative():
	return render_template('questionnaire-2.html')

@app.route('/second', methods=['POST'])
def second():
    if request.method == 'POST':
        mood_para = request.form["mood-para"]
        answers.mood_para = mood_para
        return render_template('words.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_moods = list()
        for mood in ["mood1", "mood2", "mood3", "mood4"]:
            if request.form[mood].lower() != "none":
                user_moods.append(request.form[mood])

        answers.user_moods = user_moods

        multiply = 4
        answers.mood_para += " " + " ".join(answers.user_moods*multiply)
        # print(answers.mood_para)
        
        rec = regression_paper.Gutenberg_Emotion()
        # book_meta, book_id = get_book(list(rec.recommendation(answers.mood_para).values())[0])
        vad = list(rec.recommendation(answers.mood_para.lower()).values())[0] + (1,1,1)
        df_filtered = pd.DataFrame([list(vad)], columns=["V_mean", "A_mean", "D_mean", "V_std", "A_std", "D_std"])
        vads = predict_vad(df_filtered)
        print(vads)
        book_metas, book_ids, genre_len = get_book(vads)

        return render_template('recommendations.html', metas=book_metas, book_ids=book_ids, book_range=len(book_ids), genre_len=genre_len)

@app.route('/recommend_scale', methods=['POST'])
def recommend_scale():
    if request.method == 'POST':
        N = 2
        is_depressed = False
        emotions = ["exciting", "happy", "love", "hate", "melancholy", "sad", "depressing"]
        ratings = {emotion: request.form[emotion] for emotion in emotions}

        ranking = sorted(ratings, key=lambda emotion: ratings[emotion], reverse=True)[:N]
        df = pd.read_csv("../lemma/emotions.csv")
        print(ranking)

        negatives = emotions[3:]
        # If negative, recommend positive stories
        if set(negatives).intersection(set(emotions)) == set(negatives):
            ranking = emotions[0:3]
            is_depressed = True

        indexes = list()
        for rank in ranking:
            indexes.append(df[df["emotion"] == rank].index[0])
        
        df_filtered = df.iloc[indexes]
        # vad = (df_filtered["V_mean"].mean(), df_filtered["A_mean"].mean(), df_filtered["D_mean"].mean())
        vads = predict_vad(df_filtered)
        book_metas, book_ids, genre_len = get_book(vads, is_depressed=is_depressed)

        return render_template('recommendations.html', metas=book_metas, book_ids=book_ids, book_range=len(book_ids), genre_len=genre_len)

def predict_vad(df_filtered):
    vm = df_filtered["V_mean"].mean()
    am = df_filtered["A_mean"].mean()
    dm = df_filtered["D_mean"].mean()
    vs = df_filtered["V_std"].mean()
    ars = df_filtered["A_std"].mean()
    ds = df_filtered["D_std"].mean()
    print(vm, vs, am, ars, dm, ds)

    trunc_norm = categories.Categorization()
    guesses = trunc_norm.generate_random_category(None, vm, vs, am, ars, dm, ds, 20)
    return guesses[0]

def get_book(vads, is_depressed=False):
    GUTENBERG_META = "../books/gutenberg-metadata.json"
    POS_NEG = "./pos_neg.csv"
    CUT_OFF = 31

    ids = [file.replace(".csv", "").replace("../books/predicted/", "") for file in glob.glob("../books/predicted/*.csv")]
    meta = json.load(open(GUTENBERG_META))
    pos_neg_df = pd.read_csv(POS_NEG)

    dataset = pd.DataFrame()
    for book_id in ids:
        df_book = pd.read_csv("../books/predicted/{}.csv".format(book_id))
        df_book["Y"] = str(book_id)
        dataset = dataset.append(df_book[['valence', 'arousal', 'dominance', 'Y']])

    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm="ball_tree")
    neigh.fit(dataset[['valence', 'arousal', 'dominance']], dataset["Y"])

    predicted_metas = list()
    predicted_ids = list()
    predicted_genre_len = list()
    for vad in vads:
        rec_book = neigh.predict([list(vad)])
        pos = pos_neg_df[pos_neg_df["book_id"] == int(rec_book[0])]["positive"].iloc[0]
        neg = pos_neg_df[pos_neg_df["book_id"] == int(rec_book[0])]["negative"].iloc[0]
        if neg > CUT_OFF and is_depressed == True:
            continue

        predicted_ids.append(rec_book[0])

    # Threshold
    N = 10

    for book_id in list(Counter(predicted_ids))[:N]:
        book_meta = meta[book_id]
        predicted_metas.append(book_meta)
        predicted_genre_len.append(len(book_meta["subject"]))

    return predicted_metas, list(Counter(predicted_ids))[:N], predicted_genre_len

if __name__ == "__main__":
    app.run(debug=True)