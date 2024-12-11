# ----- Naive Bayes klasifikatorius -----

from sklearn import metrics
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump

df = pd.read_csv("./../datasets/dataset_unbalanced.csv", encoding="latin1")

X = df[['message', 'length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']]
Y = df['value']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

vectorizer = TfidfVectorizer()
X_train_message = vectorizer.fit_transform(X_train['message'])
X_test_message = vectorizer.transform(X_test['message'])

X_train_combined = sp.hstack([X_train_message, sp.csr_matrix(X_train[['length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']].values)])
X_test_combined = sp.hstack([X_test_message, sp.csr_matrix(X_test[['length', 'punct_count', 'word_count', 'number_count', 'standalone_number_count', 'average_word_length', 'ratio_words_punctuation']].values)])

nb_model = MultinomialNB()
nb_model.fit(X_train_combined, Y_train)

predictions = nb_model.predict(X_test_combined)
accuracy = metrics.accuracy_score(Y_test, predictions)
print(f"Atlikta. Modelio Tikslumas: {accuracy * 100:.2f}%")
print(metrics.classification_report(Y_test, predictions))

dump(nb_model, './../trained_models/NBK/model.pkl')
dump(vectorizer, './../trained_models/NBK/vectorizer.pkl')

print(metrics.accuracy_score(Y_test, predictions))