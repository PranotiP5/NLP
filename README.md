# NLP
This folder contains codes related to Natural Language Processing(NLP).

Many applications like Google assistant, Amazon Alexa, Siri, Gmail spam classifier, etc. contain input as text. This requires natural langauge processing to convert the input text into data which can be modelled.

**Tokenization** is the first step in preprocessing in NLP. It involves converting text into sentences using packages in nltk called punkt. Punkt uses unsupervised algorithm to build model to translate text into sentences and words. It is trained on large collection of text in target language. (refer Kiss, Tibor and Strunk, Jan (2006): Unsupervised Multilingual Sentence Boundary Detection. Computational Linguistics 32: 485-525. Hence forth Paper I)

**Sentence Tokenization :** The determination of sentence boundaries is key to tokenize a text into sentences. Though period is used as sentence boudary, it is also used for ending abbreviations, initials, ordinal numbers etc. In case, period is used to end abbreviation and also a sentence, only one period will be used to end abbrevation as well as sentence (Nunberg (1990)). 
Example 1 : CELLULAR COMMUNICATIONS INC. sold 1,550,000 common shares at $21.75 each yesterday, according to lead underwriter L.F. Rothschild & Co.(cited from Wall Street Journal 05/29/1987)
In example 1, there is ambiguity as period is used in lot of places which are not sentence segmentation or boundary markers. In Kiss, Tibor and Strunk, Jan (2006), language independent way of determining sentence boudaries for such text with high accuracy is discussed. This paper is then implemented in punkt library in nltk python library. 

**Stemming:** The process of reducing words into their stem words is called stemming. The stem word for history and historical is histori. Many applications like sentiment analysis, spam classifier etc. use words to understand the meaning. However, stemming just provides stem words. Stem words many times do not carry any meaning. Here, we need lemmatization.
