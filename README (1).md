# Heart Disease Prediction

A small ML project that predicts whether a person has heart disease based on medical attributes like age, cholesterol, blood pressure, max heart rate etc.

The data is from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
It gets downloaded automatically with kagglehub when you run the notebook.

## What I did

The dataset turned out to be clean, no missing values. I checked for duplicates, then trained 3 classifiers: LogisticRegression, RandomForest and GradientBoosting, and compared them using accuracy, precision, recall and F1.

RandomForest and GradientBoosting usually get around 85-90% accuracy on this dataset.

## Project structure

```
heart-disease-prediction/
├── notebooks/
│   └── heart_disease_model.ipynb   # training notebook
├── app/
│   └── streamlit_app.py            # small web app to try the model
├── requirements.txt
└── .gitignore
```

## How to run

1. Open `notebooks/heart_disease_model.ipynb` in Google Colab and run all cells. This downloads the dataset, trains the models and saves the best one in a `model_artifacts` folder.
2. Download the `model_artifacts` folder from Colab and put it in the project root.
3. Install the requirements and run the app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The `model_artifacts` folder is not pushed to GitHub because of its size, you generate it locally by running the notebook.

## Note

This is a learning project, not medical advice.
