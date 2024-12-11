# ----- Atraminių vektorių klasifikatorius -----

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump

df = pd.read_csv("./../datasets/dataset_unbalanced.csv", encoding="latin1")

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