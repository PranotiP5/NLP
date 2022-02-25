# NLP
This folder contains codes related to Natural Language Processing(NLP).

Many applications like Google assistant, Amazon Alexa, Siri, Gmail spam classifier, etc. contain input as text. This requires natural langauge processing to convert the input text into data which can be modelled.

**(NLP 1)Tokenization** is the first step in preprocessing in NLP. It involves converting text into sentences using packages in nltk called punkt. Punkt uses unsupervised algorithm to build model to translate text into sentences and words. It is trained on large collection of text in target language. (refer Kiss, Tibor and Strunk, Jan (2006): Unsupervised Multilingual Sentence Boundary Detection. Computational Linguistics 32: 485-525. Hence forth Paper I)

**Sentence Tokenization :** The determination of sentence boundaries is key to tokenize a text into sentences. Though period is used as sentence boudary, it is also used for ending abbreviations, initials, ordinal numbers etc. In case, period is used to end abbreviation and also a sentence, only one period will be used to end abbrevation as well as sentence (Nunberg (1990)). 
Example 1 : CELLULAR COMMUNICATIONS INC. sold 1,550,000 common shares at $21.75 each yesterday, according to lead underwriter L.F. Rothschild & Co.(cited from Wall Street Journal 05/29/1987)
In example 1, there is ambiguity as period is used in lot of places which are not sentence segmentation or boundary markers. In Kiss, Tibor and Strunk, Jan (2006), language independent way of determining sentence boudaries for such text with high accuracy is discussed. This paper is then implemented in punkt library in nltk python library. 

**(NLP 2)Stemming:** The process of reducing words into their stem words is called stemming. The stem word for history and historical is histori. Many applications like sentiment analysis, spam classifier etc. use words to understand the meaning. However, stemming just provides stem words. Stem words many times do not carry any meaning. Here, we need lemmatization.

**(NLP 3)Lemmatization:** Lemmatization is similar to stemming except that it provides meaningful stem words. It is slower than stemming. It's applications include chatbot, question answering system.

**Cleaning or Pre-processing:**  This may involve removing unnecessary punctuations, commas etc. , removing stop words which do not add to the meaning of a paragraph. It will also involve lowering of sentences. It includes stemming or lemmatization.

**(NLP 4)Bag of Words:** Its a document matrix. It translates words to numerical representation called vectors. CounterVectorizer from sklearn library creates a histogram of words. CounterVectorizer.fit_transform will then create a matrix of sentences to words. It will tell which words occur in a sentence. This can then be fed to machine learning models for various applications like sentiment analysis.

**(NLP 5)Term Frequency Inverse Document Frequency (TFIDF)** TF is ratio of number of repetition of words in sentence to number of words in that sentence. IDF is log of ratio of number of sentences to the number of sentences containing the word. TFIDF is multiplication of TF and IDF. It gives less importance to frequently occuring words. 

**(NLP 6)Spam Classifier** Using Naive Bayes classification algorithm and around 5000 SMS(UCI Machine Learning Repository), spam is classified with 99.28% accuracy as either ham or spam. Pre-processing steps were performed on the dataset of SMS. Lemmatization performed better than stemming for same parameters like max_features and test_size. Bag of words and TF-IDF performed equally well. max_features parameter of 2500 perfomed well. test_size of 0.05 gave maximum accuracy of 99.28. 
