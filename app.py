from flask import Flask, render_template, request
import joblib
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", " <num> ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {
        "nim": "",
        "nama": "",
        "judul": "",
        "abstrak": ""
    }

    if request.method == "POST":
        form_data["nim"] = request.form.get("nim", "").strip()
        form_data["nama"] = request.form.get("nama", "").strip()
        form_data["judul"] = request.form.get("judul", "").strip()
        form_data["abstrak"] = request.form.get("abstrak", "").strip()

        if form_data["judul"] and form_data["abstrak"]:
            full_text = f'{form_data["judul"]} {form_data["abstrak"]}'
            clean = clean_text(full_text)
            vector = tfidf.transform([clean])
            pred = model.predict(vector)
            prediction = le.inverse_transform(pred)[0]
        else:
            prediction = "Judul dan abstrak wajib diisi"

    return render_template(
        "index.html",
        prediction=prediction,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)