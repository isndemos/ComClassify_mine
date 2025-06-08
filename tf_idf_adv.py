import logging
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logging.info("Start tf-idf advanced script")
# Читаем файл
train = pd.read_csv("data/output/train.csv", index_col=0)
test = pd.read_csv("data/output/test.csv", index_col=0)

# Создаём pipeline: векторизация + модель
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),  # увеличим max_iter для надёжности
    ]
)

# Задаём сетку параметров
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "lbfgs"],
}

# GridSearchCV с 5-кратной кросс-валидацией
logging.info("Start grid search")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
grid_search.fit(train["text"], train["class"])

# Лучшие параметры
logging.info("Лучшие параметры:", grid_search.best_params_)

# Предсказания на тесте
y_pred = grid_search.predict(test["text"])

# Метрики
print("Accuracy score:", accuracy_score(test["class"], y_pred))
print("Precision score:", precision_score(test["class"], y_pred, average="weighted"))
print("Recall score:", recall_score(test["class"], y_pred, average="weighted"))
print("F1 score:", f1_score(test["class"], y_pred, average="weighted"))

# Сохраняем модели
with open("models/tf_idf_adv.pkl", "wb") as file:
    pickle.dump(grid_search.best_estimator_.named_steps["tfidf"], file)

with open("models/lr_adv.pkl", "wb") as file:
    pickle.dump(grid_search.best_estimator_.named_steps["clf"], file)
