# Adult Income Predictor Pro

A polished Streamlit software package for the UCI Adult / Census Income dataset.

## Run locally (Python scripts)

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Run locally (Jupyter notebook workflow)

```bash
pip install -r requirements.txt
jupyter notebook notebooks/adult_income_coursework.ipynb
```

Then open the notebook and run all cells from top to bottom.

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
