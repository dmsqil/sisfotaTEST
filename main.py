from fastapi import FastAPI, Form, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import joblib
import re

from database import SessionLocal, engine, Base
from models import KlasifikasiTA

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = FastAPI()

# =========================
# BUAT TABEL DI MYSQL
# =========================
Base.metadata.create_all(bind=engine)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# =========================
# PREPROCESSING
# =========================
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

# =========================
# DATABASE SESSION
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# ROUTE HTML
# =========================
@app.get("/")
def home():
    return FileResponse("templates/index.html")

# =========================
# API PREDICT + SIMPAN DB
# =========================
@app.post("/predict")
def predict(
    judul: str = Form(...),
    abstrak: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        full_text = f"{judul} {abstrak}".strip()

        if len(full_text.split()) < 5:
            return {"error": "Teks terlalu pendek"}

        clean = clean_text(full_text)
        vector = tfidf.transform([clean])

        if vector.nnz == 0:
            return {"error": "Teks tidak dikenali"}

        pred = model.predict(vector)
        prediction = le.inverse_transform(pred)[0]

        # =========================
        # SIMPAN KE MYSQL
        # =========================
        data = KlasifikasiTA(
            judul=judul,
            abstrak=abstrak,
            hasil=prediction
        )

        db.add(data)
        db.commit()

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

# =========================
# API LIHAT DATA
# =========================
@app.get("/data")
def get_data(db: Session = Depends(get_db)):
    return db.query(KlasifikasiTA).all()