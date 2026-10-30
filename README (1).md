# Consumer Complaint Classification

A deep learning NLP project that predicts the product category of a customer complaint from its written narrative. It compares three sequence models built from scratch (SimpleRNN, LSTM, GRU) against a fine-tuned HuggingFace transformer (DistilBERT) and deploys the best one as a Gradio web app.

## Objectives
- Build an end-to-end NLP classification pipeline
- Compare SimpleRNN, LSTM and GRU models built from scratch
- Fine-tune a transformer and benchmark it against the custom models
- Improve complaint routing efficiency and reduce manual effort and cost

## Dataset
[Consumer Complaints dataset (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp): customer complaint narratives labeled by product category. The notebook downloads it automatically with `kagglehub`.

## Pipeline
1. Load and explore the data
2. Text preprocessing: lowercasing, removing punctuation, numbers and special characters, stop word removal, lemmatization
3. Class distribution analysis and imbalance handling (top categories, per-class cap, class weights)
4. Tokenization and sequence padding
5. Embedding layer feeding the RNN models
6. Train SimpleRNN, LSTM, GRU and a fine-tuned DistilBERT
7. Evaluate with accuracy, precision, recall, F1-score and confusion matrix
8. Compare all models and pick the best
9. Classify new complaints and deploy with Gradio

## Results
Test-set results (weighted averages). Replace with the values from your run (`artifacts/model_comparison.csv`).

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| SimpleRNN | | | | |
| LSTM | | | | |
| GRU | | | | |
| DistilBERT | | | | |

## Project structure
```
.
├── Consumer_Complaint_Classification.ipynb   # full pipeline: data to training to evaluation
├── app.py                                    # standalone Gradio app
├── requirements.txt
└── artifacts/                                # created by the notebook (models, tokenizer, label map)
```

## How to run

### 1. Train (Google Colab)
1. Open `Consumer_Complaint_Classification.ipynb` in Google Colab
2. Set the runtime to GPU (Runtime > Change runtime type > T4 GPU)
3. Run all cells. The last cells save everything the app needs into `artifacts/`

### 2. Launch the app
In Colab, run the Gradio cell at the end of the notebook. It prints a public link.

To run it on your own machine:
```bash
pip install -r requirements.txt
python app.py
```
`app.py` expects the `artifacts/` folder (from the notebook) in the same directory.

## The app
Paste a complaint and get:
- the predicted category
- the confidence score
- the probability for every category

Example input:
```
A collection agency keeps calling me at work about a debt I already paid. They threatened to sue me and never sent me written validation of the debt.
```

## Tech stack
Python, TensorFlow/Keras, PyTorch, HuggingFace Transformers, scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn, Gradio

## Author
Mahmoud: [GitHub](https://github.com/mahmoudeimo95)
