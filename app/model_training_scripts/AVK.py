# ----- Atraminių vektorių klasifikatorius -----

import json
from joblib import dump
import scipy.sparse as sp
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

dataset = 'dataset_unbalanced.csv'
df = pd.read_csv("./../datasets/" + dataset, encoding="latin1")

total_rows = len(df)
ham_count = df[df['value'] == 'ham'].shape[0]
model_effectiveness = (ham_count / total_rows) * 100

X = df[['message', 'length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']]
Y = df['value']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

vectorizer = TfidfVectorizer()
X_train_message = vectorizer.fit_transform(X_train['message'])
X_test_message = vectorizer.transform(X_test['message'])

X_train_combined = sp.hstack([X_train_message, sp.csr_matrix(X_train[['length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']].values)])
X_test_combined = sp.hstack([X_test_message, sp.csr_matrix(X_test[['length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']].values)])

svc_model = SVC(gamma="scale")
svc_model.fit(X_train_combined, Y_train)

predictions = svc_model.predict(X_test_combined)
accuracy = metrics.accuracy_score(Y_test, predictions)
print(f"Atlikta. Modelio Tikslumas: {accuracy * 100:.2f}%")
print(metrics.classification_report(Y_test, predictions))

dump(svc_model, './../trained_models/AVK/model.pkl')
dump(vectorizer, './../trained_models/AVK/vectorizer.pkl')

predicted_ham = (predictions == 'ham').sum()
predicted_spam = (predictions == 'spam').sum()

metadata = {
    "model": "Atraminių vektorių klasifikatorius",
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
        "accuracy": f"{accuracy * 100:.2f}%",
        "report": metrics.classification_report(Y_test, predictions)
    }
}

metadata_path = './../trained_models/AVK/metadata.json'
with open(metadata_path, 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=4)