import math

def calculate_tf(word, document):
    # Calculate term frequency (tf) of a word in a document
    return document.count(word) / len(document)

def calculate_idf(word, documents):
    # Calculate inverse document frequency (idf) of a word across all documents
    n = len(documents)
    df = sum([1 for document in documents if word in document])
    return math.log(n / (1 + df))

def calculate_tfidf(word, document, documents):
    # Calculate tf-idf score of a word in a document
    tf = calculate_tf(word, document)
    idf = calculate_idf(word, documents)
    return tf * idf

# Example usage
documents = [
    "This is the first document",
    "This is the second document",
    "And this is the third one",
]

# Create vocabulary list
vocab = set()
for doc in documents:
    for word in doc.split():
        vocab.add(word)

# Calculate tf-idf scores for each word in each document
tfidf_scores = []
for doc in documents:
    scores = []
    for word in vocab:
        score = calculate_tfidf(word, doc.split(), documents)
        scores.append(score)
    tfidf_scores.append(scores)

# Print the tf-idf scores for each word in each document
for i, doc_scores in enumerate(tfidf_scores):
    print("Document", i+1)
    for word, score in zip(vocab, doc_scores):
        print(word, ":", score)
    print()
