# ----- Logistinės regresijos klasifikatorius -----

import json
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump

dataset = 'dataset_unbalanced.csv'
df = pd.read_csv("./../datasets/" + dataset, encoding="latin1")

total_rows = len(df)
ham_count = df[df['value'] == 'ham'].shape[0]
model_effectiveness = (ham_count / total_rows) * 100

X_num = df[['length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']]
Y = df['value']

X_num.columns = X_num.columns.astype(str)

vectorizer = TfidfVectorizer()
X_message = vectorizer.fit_transform(df['message'])

X_combined = pd.concat([pd.DataFrame(X_message.toarray()), X_num], axis=1, ignore_index=True)

X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y)

lr_model = LogisticRegression(solver='lbfgs', max_iter=500)
lr_model.fit(X_train, Y_train)

predictions = lr_model.predict(X_test)
print(metrics.accuracy_score(Y_test, predictions))

dump(lr_model, "./../trained_models/LRK/model.pkl")
dump(vectorizer, "./../trained_models/LRK/vectorizer.pkl")

predicted_ham = (predictions == 'ham').sum()
predicted_spam = (predictions == 'spam').sum()

metadata = {
    "model": "Logistinės regresijos klasifikatorius",
    "dataset": dataset,
    "dataset_effectiveness": f"{model_effectiveness:.2f}%",
    "training_data": {
        "size": len(X_train),
        "distribution": Y_train.value_counts().to_dict()
    },
    "testing_data": {
        "size": len(X_test),
        "distribution": Y_test.value_counts().to_dict()
    },
    "model_results": {
        "ham": int(predicted_ham),
        "spam": int(predicted_spam)
    },
    "metrics": {
        "accuracy":  f"{metrics.accuracy_score(Y_test, predictions) * 100:.2f}%",
        "report": metrics.classification_report(Y_test, predictions)
    }
}

metadata_path = './../trained_models/LRK/metadata.json'
with open(metadata_path, 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=4)