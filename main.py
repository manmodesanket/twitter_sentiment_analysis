from flask import Flask, request, jsonify, render_template
import pickle
import time
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

app = Flask(__name__)

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SEQUENCE_LENGTH = 300
SENTIMENT_THRESHOLDS = (0.4, 0.7)

path = "training.csv"
df = pd.read_csv(path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
df_train, df_test = train_test_split(
    df, test_size=1-TRAIN_SIZE, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)
vocab_size = len(tokenizer.word_index) + 1

file = open('sentiment_analysis_pickle', 'rb')
model = pickle.load(file)
file.close()


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods = ['POST'])
def prediction():
    if request.method == 'POST':
        text = request.form['text']
        include_neutral = True
        start_at = time.time()
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
        label = NEUTRAL
        score = model.predict([x_test])[0]
        if include_neutral:
            if score <= SENTIMENT_THRESHOLDS[0]:
                label = NEGATIVE
            elif score >= SENTIMENT_THRESHOLDS[1]:
                label = POSITIVE
        else:
            if score < 0.5:
                label = NEGATIVE 
            else:
                label = POSITIVE
        result = {"label": label, "score": float(score),
            "elapsed_time": time.time()-start_at}
        return render_template("prediction.html", result = result)
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=False, threaded=False)