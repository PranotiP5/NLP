#!/usr/bin/env python
# coding: utf-8

# In[16]:



import nltk


# In[17]:


paragraph =  """Construction of the mausoleum was essentially completed in 1643, but work continued on other phases of the project for another 10 years. The Taj Mahal complex is believed to have been completed in its entirety in 1653 at a cost estimated at the time to be around ₹32 million, which in 2020 would be approximately ₹70 billion (about U.S. $1 billion). The construction project employed some 20,000 artisans under the guidance of a board of architects led by the court architect to the emperor, Ustad Ahmad Lahauri.

The Taj Mahal was designated as a UNESCO World Heritage Site in 1983 for being "the jewel of Muslim art in India and one of the universally admired masterpieces of the world's heritage". It is regarded by many as the best example of Mughal architecture and a symbol of India's rich history. The Taj Mahal attracts more than 6 million visitors a year[3] and in 2007, it was declared a winner of the New 7 Wonders of the World (2000–2007) initiative. """


# In[18]:


# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[19]:


ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []


# In[20]:


for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])#substitute everything in sentences that is not a-z or A-Z with space
    review = review.lower()#lower the sentences to a-z
    review = review.split()#split the sentence into words
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]#Lemmatize the words which are not stop words
    review = ' '.join(review)#join the words with space
    corpus.append(review)#append review to cospus


# In[21]:


print(corpus)


# In[29]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import  TfidfVectorizer
cv = TfidfVectorizer()#create an object of TfidfVectorizer
X = cv.fit_transform(corpus).toarray()#creates document matrix with dimensions of number of sentences x number of total words.
#words occusing frequently get less importance. X can then be fed to machine learning model for example for sentiment analysis.


# In[30]:


print(X)


# In[ ]:





# In[ ]:




