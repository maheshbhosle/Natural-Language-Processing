import numpy as np

#TF-IDF (Term Frequency- Inverse Document Frequency)
#TF-IDF is a weight often used in information retrieval and text mining.
#This weight is a statistical measure used to evaluate how important a word is to a document in a collection (corpus).

#The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in corpus.

#Variations of the tf-idf weighing scheme are often used
#by search engines as a central tool in scoring and ranking
#a document's relevance given a user query.

#How to Compute:
#TF(t) = number of times the term t appears in a document / Total number of terms in the document
#IDF(t) = ln(total number of documents/Number of documents having term t)
#TF-IDF(t) = TF(t) * IDF(t)

def tfidf(term, doc, corpus):

    times_t = doc.count(term)
    terms_doc = len(doc)
    tf = times_t / terms_doc

    total_docs = len(corpus)
    docs_having_term = 0

    for document in corpus:
        if term in document:
            docs_having_term+=1

    idf = np.log(total_docs/docs_having_term)

    return tf*idf

doc1 = ['apple', 'apple', 'apple']
doc2 = ['apple', 'mango', 'mango']
doc3 = ['apple', 'mango', 'banana']

corpus = [doc1, doc2, doc3]

print('----------------------------------------------')
weight = tfidf('apple', doc1, corpus)
print('TFIDF weight of apple in ', doc1, ':', weight)
weight = tfidf('apple', doc2, corpus)
print('TFIDF weight of apple in ', doc2, ':', weight)
weight = tfidf('apple', doc3, corpus)
print('TFIDF weight of apple in ', doc3, ':', weight)
print('----------------------------------------------')

print('----------------------------------------------')
weight = tfidf('mango', doc1, corpus)
print('TFIDF weight of mango in ', doc1, ':', weight)
weight = tfidf('mango', doc2, corpus)
print('TFIDF weight of mango in ', doc2, ':', weight)
weight = tfidf('mango', doc3, corpus)
print('TFIDF weight of mango in ', doc3, ':', weight)
print('----------------------------------------------')

print('----------------------------------------------')
weight = tfidf('banana', doc1, corpus)
print('TFIDF weight of banana in ', doc1, ':', weight)
weight = tfidf('banana', doc2, corpus)
print('TFIDF weight of banana in ', doc2, ':', weight)
weight = tfidf('banana', doc3, corpus)
print('TFIDF weight of banana in ', doc3, ':', weight)
print('----------------------------------------------')
