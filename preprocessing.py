import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

TEXT2ID = {"негативная": 0, "позитивная": 1, "мусор": 2, "нейтральная": 3}

logging.info("Читаю файл")
df = pd.read_csv("data/processing/data_read.csv", index_col=0)

logging.info("Делаю обработку")
df["Эмоциональная окраска"] = df["Эмоциональная окраска"].apply(
    lambda x: x.strip().lower()
)
df["Эмоциональная окраска"] = df["Эмоциональная окраска"].map(TEXT2ID)

# Переименуем для удобства
df.columns = ["text", "class"]

# Удалим NA (заметим, что там всего 1 NA)
logging.info("Удаляем пропуски")
df = df.dropna()

### Посмотри еще 1 график
class_values = df["class"].value_counts()

plt.figure(figsize=(7, 7))
plt.bar(
    class_values.index,
    class_values,
    label=TEXT2ID.keys(),
    color=["blue", "red", "yellow", "green"],
)
plt.title("Кол-во комментариев разных классов")
plt.xlabel("Окрас")
plt.ylabel("Кол-во")
plt.legend()
plt.xticks(class_values.index)
plt.savefig("analyze.svg")


# Заметим, что есть дисбаланс, поэтому сделаем train/test split с помощью дисбаланса
X = df["text"]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, stratify=y, shuffle=True
)

# Сохраним все, для тренировки
pd.concat([X_train, y_train], axis=1).to_csv("data/output/train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("data/output/test.csv")
