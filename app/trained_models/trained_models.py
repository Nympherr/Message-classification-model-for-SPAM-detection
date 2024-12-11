import joblib
import re
import numpy as np
import scipy.sparse as sp

# Atraminių vektorių klasifikatorius
model_avk = joblib.load('./trained_models/AVK/model.pkl')
vectorizer_avk = joblib.load('./trained_models/AVK/vectorizer.pkl')

# Logistinės regresijos klasifikatorius
model_lrk = joblib.load('./trained_models/LRK/model.pkl')
vectorizer_lrk = joblib.load('./trained_models/LRK/vectorizer.pkl')

# Naive Bayes klasifikatorius
model_nbk = joblib.load('./trained_models/NBK/model.pkl')
vectorizer_nbk = joblib.load('./trained_models/NBK/vectorizer.pkl')

# K-artimiausių kaimynų klasifikatorius
model_k_nearest = joblib.load('./trained_models/k-nearest/model.pkl')
vectorizer_k_nearest = joblib.load('./trained_models/k-nearest/vectorizer.pkl')

def use_classifying_algorithm(message, algorithm):
    model = get_correct_model(algorithm)
    vectorizer = get_correct_vectorizer(algorithm)

    message_length = len(message)
    message_punct_count = len(re.findall(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', message))
    message_vector = vectorizer.transform([message])

    word_count = len(message.split())
    number_count = len(re.findall(r'\d+', message))
    standalone_number_count = len(re.findall(r'\b\d+\b', message))
    average_word_length = sum(len(word) for word in message.split()) / word_count if word_count > 0 else 0
    ratio_words_punctuation = word_count / message_punct_count if message_punct_count > 0 else 0

    if model in [model_avk, model_nbk, model_k_nearest]:
        additional_features = sp.csr_matrix([
            [message_length, message_punct_count, word_count, number_count, standalone_number_count, average_word_length, ratio_words_punctuation]
        ])
    else:
        additional_features = np.array([
            [message_length, message_punct_count, word_count, number_count, standalone_number_count, average_word_length, ratio_words_punctuation]
        ])

    vectorized_message_values = sp.hstack([message_vector, additional_features])

    result = model.predict(vectorized_message_values)
    return result[0]

def get_correct_model(algorithm):
    models = {
        'AVK': model_avk,
        'LRK': model_lrk,
        'NBK': model_nbk,
        'k-nearest': model_k_nearest
    }

    return models.get(algorithm, model_avk)

def get_correct_vectorizer(algorithm):
    vectorizers = {
        'AVK': vectorizer_avk,
        'LRK': vectorizer_lrk,
        'NBK': vectorizer_nbk,
        'k-nearest': vectorizer_k_nearest
    }

    return vectorizers.get(algorithm, vectorizer_avk)