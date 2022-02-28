#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])


# In[7]:


#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')


# In[116]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    #lemmatizer improves accuracy
    review = ' '.join(review)
    corpus.append(review)


# In[184]:


# Creating the Bag of Words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=3000)
#X = cv.fit_transform(corpus).toarray()

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)#can change max_features
X = cv.fit_transform(corpus).toarray()

#both bag of words and TF-IDF give same accuracy after fine tuning max_features and test_size


# In[185]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# In[186]:


print(y)


# In[187]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.050, random_state = 0)#can change test_size

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


# In[188]:


from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)


# In[189]:


print(confusion_m)


# In[190]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


# In[191]:


print(accuracy)


# In[ ]:




