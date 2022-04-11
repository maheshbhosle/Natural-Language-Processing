#Extending the abilities of the CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import re


class Lemma_CountVectorizer (CountVectorizer):
    #build_analyzer() is called on init

    def build_analyzer(self):
        #Invoke the CountVectorizer build_analyzer()
        #It loads a function in memory and returns the reference (analyzer).
        #That function (analyzer) when called would act on a sentence and return a list of words that should be included in the vocabulary.
        #Ideally it (analyzer) performs tokenization, stop word exclusion, etc

        analyzer = CountVectorizer.build_analyzer(self)
        lemmatizer = WordNetLemmatizer()
        #Inner function to
        # 1) Call the CountVectorizers analyzer
        # 2) Stem the words returned by the analyzer
        # 3) returns the stemmed words

        def lemma_analyzer(sentence):
            data = []
            for x in analyzer(sentence):
                data.append(lemmatizer.lemmatize(x))
            return data
        return lemma_analyzer

def load_document(doc):
    file_handle = open(doc)
    mem_doc = []
    for x in file_handle: #read it line by line
        #Get rid of the leading and trailing spaces,\n
        x = x.strip()
        if len(x) > 0:
            x =  re.sub('\(.*\)','', x)
            x = re.sub('\[.*?\]', '', x)
            mem_doc.append(x)

    file_handle.close()
    return mem_doc

def preprocess(mem_doc):
    #Text data requires preprocessing before use for predictive modeling.
    #The text must be parsed to remove stop wrods, called as tokenization.
    #Then the words need to be encoded as numbers for use in a ML algorithm.
    #This preprocessing is feature extraction is Vectorization.

    #cnt_vectorize = CountVectorizer(stop_words='english')
    lcv = Lemma_CountVectorizer(stop_words='english')

    #learn a vocabulary from the document
    #A dict is created with words as keys and their occurrence (in a sorted order) as value
    #cnt_vectorize.fit(mem_doc)
    lcv.fit(mem_doc)
    #print(cnt_vectorize.vocabulary_)

    #Look at the words in vocabulary
    #print(cnt_vectorize.get_feature_names())
    print(lcv.get_feature_names())

    #For every element (sentence) of the mem_doc, create a vector. The words of the sentences get encoded into numbers using the vocabulary.
    #bow = cnt_vectorize.transform(mem_doc)
    bow = lcv.transform(mem_doc)
    #See the Bag of words
    print(bow) #way 1
    print(bow.toarray())

    query = ['Horse is a four legged animal']
    #qbow = cnt_vectorize.transform(query)
    qbow = lcv.transform(query)
    print('-------------------')
    print(qbow)


def main():
    mem_doc = load_document('d:/batches/SE_PBL/horse.txt')
    mem_doc.append('I like Black horse and I hate non black Horse')
    print(len(mem_doc), mem_doc)
    preprocess(mem_doc)

main()
# lemmatizer = WordNetLemmatizer()
# for x in ['adults', 'adult', 'age', 'ages', 'approximately', 'allowing']:
#     print(x, lemmatizer.lemmatize(x, pos = 'n'))

