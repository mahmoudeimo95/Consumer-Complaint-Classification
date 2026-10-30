"""Standalone Gradio app. Needs the artifacts/ folder produced by the notebook next to this file."""
import json, re, numpy as np, gradio as gr
import nltk
nltk.download("stopwords", quiet=True); nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ART = "artifacts"
cfg = json.load(open(f"{ART}/config.json"))
id2label = {int(k): v for k, v in json.load(open(f"{ART}/label_map.json"))["id2label"].items()}
best = open(f"{ART}/best_model.txt").read().strip()
stop_words, lemmatizer = set(stopwords.words("english")), WordNetLemmatizer()

def clean_text(t):
    t = re.sub(r"[^a-z\s]", " ", re.sub(r"x{2,}", " ", str(t).lower()))
    toks = [w for w in t.split() if w not in stop_words and len(w) > 1]
    if cfg["USE_LEMMATIZER"]:
        toks = [lemmatizer.lemmatize(w) for w in toks]
    return " ".join(toks)

def light_clean(t):
    return re.sub(r"\s+", " ", re.sub(r"x{2,}", " ", str(t).lower())).strip()

if best == "Transformer":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(f"{ART}/transformer_model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{ART}/transformer_model").eval()
    def probs_for(text):
        enc = tok(light_clean(text), truncation=True, max_length=cfg["TRANSFORMER_MAXLEN"], return_tensors="pt")
        with torch.no_grad():
            return torch.softmax(model(**enc).logits, dim=-1)[0].numpy()
else:
    import tensorflow as tf
    model = tf.keras.models.load_model(f"{ART}/{best.lower()}_model.keras")
    word_index = json.load(open(f"{ART}/tokenizer.json"))["word_index"]
    def probs_for(text):
        seq = [word_index.get(w, 1) for w in clean_text(text).split()][: cfg["MAX_LEN"]]
        x = np.zeros((1, cfg["MAX_LEN"]), dtype="int32"); x[0, : len(seq)] = seq
        return model.predict(x, verbose=0)[0]

def classify(text):
    if not text or not text.strip():
        return "Please enter a complaint.", "", {}
    p = probs_for(text); i = int(np.argmax(p))
    return id2label[i], f"{p[i]:.1%}", {id2label[j]: float(v) for j, v in enumerate(p)}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=8, label="Complaint narrative", placeholder="Paste or type the customer's complaint here..."),
    outputs=[gr.Textbox(label="Predicted category"), gr.Textbox(label="Confidence score"),
             gr.Label(num_top_classes=len(id2label), label="Probability per category")],
    examples=["I found accounts on my credit report that I never opened and the bureau refuses to remove them.",
              "The debt collector keeps calling me at work for a debt I already paid."],
    title="Consumer Complaint Classifier",
    description=f"Predicts the complaint category from the complaint text. Model in use: {best}.",
    flagging_mode="never",
)
if __name__ == "__main__":
    demo.launch()
