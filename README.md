# Fake News Detector

A simple fake news detection project using text classification.

## Project structure

- `data/` - dataset files, including `fake_or_real_news.csv`
- `src/` - training, prediction, and utility scripts
- `models/` - saved model artifacts
- `app/` - Streamlit app for interactive predictions

## Setup

1. Create a Python environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in `data/fake_or_real_news.csv`.

## Training

Run training to build the model:

```bash
python src\train.py --data-path data\fake_or_real_news.csv --model-path models\model.pkl
```

## Prediction

Use the prediction script to classify single texts:

```bash
python src\predict.py --model-path models\model.pkl --text "This is a news headline to classify."
```

## Streamlit app

Run the app locally:

```bash
streamlit run app\app.py
```
