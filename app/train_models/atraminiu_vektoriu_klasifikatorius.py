import joblib
import scipy.sparse as sp
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import re

# Load and preprocess the dataset
df = pd.read_csv("./../datasets/dataset-spam.csv", encoding="latin1")

# Preprocessing
punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
df['length'] = df['message'].astype(str).apply(len)
df['punct_count'] = df['message'].apply(lambda x: len(re.findall(punctuation_pattern, x)))

# Split dataset
X = df[['message', 'length', 'punct_count']]
Y = df['value']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Vectorization and model training
vectorizer = TfidfVectorizer()
X_train_message = vectorizer.fit_transform(X_train['message'])
X_test_message = vectorizer.transform(X_test['message'])

X_train_combined = sp.hstack([X_train_message, sp.csr_matrix(X_train[['length', 'punct_count']].values)])
X_test_combined = sp.hstack([X_test_message, sp.csr_matrix(X_test[['length', 'punct_count']].values)])

svc_model = SVC(gamma="scale")
svc_model.fit(X_train_combined, Y_train)

# Save model and vectorizer
joblib.dump(svc_model, './../models/AVK/model.pkl')
joblib.dump(vectorizer, './../models/AVK/vectorizer.pkl')

# Evaluate accuracy
predictions = svc_model.predict(X_test_combined)
accuracy = metrics.accuracy_score(Y_test, predictions)
print(f"Model accuracy: {accuracy:.3f}")
print("Model and vectorizer saved successfully.")