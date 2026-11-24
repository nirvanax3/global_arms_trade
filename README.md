# 🔍 ArmsLens: AI-Powered Analysis and Prediction of Global Arms Trade (2000–2023)

This is an interactive **Streamlit dashboard** that analyzes and visualizes global arms transfers from 2000 to 2023. It is built with **Python**, **Pandas**, **scikit-learn**, **Seaborn**, **Matplotlib**, and **Altair**, and allows users to explore various aspects of arms trading activities between countries.

---

## 📊 Features

- **Top Exporters & Importers** by number of trades
- **Most Traded Weapon Categories & Designations**
- **Trends Over Time** (annual transfers, category popularity)
- **Country-Specific Insights** (average quantity, delivery delays, trade diversity)
- **Custom Filters**: Filter by year, country, category, etc.
- **Multiple Visualizations**: Bar charts, line plots, heatmaps, bubble charts, pie charts, violin plots, and more.

## 🤖 Machine Learning

| | Model A: Delivery Delay Predictor | Model B: Weapon Category Classifier |
|---|---|---|
| Task | Regression | Text classification (NLP) |
| Input | Exporter, importer, category, quantity, order year | Weapon description + designation |
| Models compared | Baseline (mean), Linear Regression, **Random Forest** | Naive Bayes, **TF-IDF + Logistic Regression** |
| Result (test set) | MAE 1.67 years (21% better than baseline), R² 0.31 | 98.9% accuracy, macro F1 0.98 |

- Model A trains on orders 2000–2018 to avoid right-censoring (recent long-delay deals are not delivered yet).
- Model B removes duplicate texts before the train/test split to avoid data leakage.

## 📁 Project Structure

```
app.py          # Streamlit dashboard (EDA Q1–Q22 + ML section)
ml_models.py    # Model training, evaluation and prediction functions
data/           # global_arms_transfer_2000_2023.csv
requirements.txt
```

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`

---

## 🚀 Demo

You can deploy this app on [Streamlit Cloud](https://share.streamlit.io/) in just a few clicks.
