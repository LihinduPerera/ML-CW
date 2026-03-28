# Adult Income Predictor Pro

A polished Streamlit software package for the UCI Adult / Census Income dataset.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## What is included

- Dataset preview and EDA screens
- Model comparison page
- Saved deployment model for live predictions
- Prediction history download

## ML design

- Logistic Regression
- KNN
- Random Forest
- Feature engineering: `capital_net`, `has_capital_gain`, `has_capital_loss`, `age_band`, `hours_band`
- Preprocessing: imputation, scaling, one-hot encoding
