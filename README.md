# Artificial Summer Intern Technical Task
Classification model to predict the country of origin of wines.

| Section | File Name |
|-|-|
| Initial exploratory data analysis | explore_data.ipynb
| Using traditional NLP techniques | nlp.py

## Methodology
An initial exploratory data analysis was done to get an overview and general understanding of the dataset. Various plots were used to see basic relationships within the data.

Experimenting with feature engineering using traditional NLP techniques was then undertaken to test if the techniques could build successful classification models for the data. Four different NLP methods were tested:
- Bag of words
- Bi-gram
- Tri-gram
- TF-IDF

A train test split of 80/20 was used to train the model using one of the various methods and then test the model's success. The success of these models was then tested using traditional metrics, which included precision, recall, F1 score and a confusion matrix.
