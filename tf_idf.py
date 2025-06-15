import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

# Читаем файл
train = pd.read_csv("data/output/train.csv", index_col=0)
test = pd.read_csv("data/output/test.csv", index_col=0)

# Векторизуем
model = TfidfVectorizer()
X_train_vector = model.fit_transform(train["text"])
X_test_vector = model.transform(test["text"])


# Обучаем логистическую регрессию
lr = LogisticRegression()
lr.fit(X_train_vector, train["class"])
y_pred = lr.predict(X_test_vector)

# Метрика качества
print("Accuracy score:", accuracy_score(test["class"], y_pred))
print("Precision score:", precision_score(test["class"], y_pred, average="weighted"))
print("Recall score:", recall_score(test["class"], y_pred, average="weighted"))
print("F1 score:", f1_score(test["class"], y_pred, average="weighted"))

# Сохраним модели
with open("models/tf_idf_start.pkl", "wb") as file:
    pickle.load(model, file=file)

with open("models/lr_start.pkl", "wb") as file:
    pickle.load(lr, file=file)
