# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 03:41:08 2023

@author: Vikki
"""
print("------------------one-hot ---------------")
# example sentence
sentence = "The cat sat on the mat."

# create a set of unique words in the sentence
words = set(sentence.split())

# create a dictionary of one-hot encoded vectors for each word
ohe_dict = {}
for i, word in enumerate(words):
    ohe_dict[word] = [0] * len(words)
    ohe_dict[word][i] = 1

# print the one-hot encoded vectors
for word, vector in ohe_dict.items():
    print(word, vector)


print("------------------Count Matrix: ---------------")
from sklearn.feature_extraction.text import CountVectorizer

# example sentences
sentences = ["This is the first document.", "This document is the second document."]

# create CountVectorizer object
vectorizer = CountVectorizer()

# fit the vectorizer on the sentences
vectorizer.fit(sentences)

# transform the sentences into document-term matrix
count_matrix = vectorizer.transform(sentences)

# print the vocabulary and document-term matrix
print("Vocabulary: ", vectorizer.vocabulary_)
print("Count Matrix: ")
print(count_matrix.toarray())
print("-------------------BOW Matrix:--------------")


# create CountVectorizer object with binary=True to get BOW representation
vectorizer = CountVectorizer(binary=True)

# fit the vectorizer on the sentences
vectorizer.fit(sentences)

# transform the sentences into document-term matrix
bow_matrix = vectorizer.transform(sentences)

# print the vocabulary and document-term matrix
print("Vocabulary: ", vectorizer.vocabulary_)
print("BOW Matrix: ")
print(bow_matrix.toarray())

print("----------------N-gram Matrix:-----------------")


# create CountVectorizer object with ngram_range=(1,2) to get bigram representation
vectorizer = CountVectorizer(ngram_range=(1,2))

# fit the vectorizer on the sentences
vectorizer.fit(sentences)

# transform the sentences into document-term matrix
ngram_matrix = vectorizer.transform(sentences)

# print the vocabulary and document-term matrix
print("Vocabulary: ", vectorizer.vocabulary_)
print("N-gram Matrix: ")
print(ngram_matrix.toarray())
