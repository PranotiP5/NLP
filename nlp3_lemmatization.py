#!/usr/bin/env python
# coding: utf-8

# $$x=3$$
# 

# In[8]:


import nltk


# In[9]:


nltk.download()


# In[37]:


from nltk.stem import WordNetLemmatizer


# In[39]:


from nltk.corpus import stopwords


# In[40]:


paragraph = """ Construction of the mausoleum was essentially completed in 1643, but work continued on other phases of the project for another 10 years. The Taj Mahal complex is believed to have been completed in its entirety in 1653 at a cost estimated at the time to be around ₹32 million, which in 2020 would be approximately ₹70 billion (about U.S. $1 billion). The construction project employed some 20,000 artisans under the guidance of a board of architects led by the court architect to the emperor, Ustad Ahmad Lahauri.

The Taj Mahal was designated as a UNESCO World Heritage Site in 1983 for being "the jewel of Muslim art in India and one of the universally admired masterpieces of the world's heritage". It is regarded by many as the best example of Mughal architecture and a symbol of India's rich history. The Taj Mahal attracts more than 6 million visitors a year[3] and in 2007, it was declared a winner of the New 7 Wonders of the World (2000–2007) initiative.  """


# In[41]:


# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)


# In[42]:


print(sentences)


# In[43]:


print(len(sentences))


# In[45]:


lemmatizer = WordNetLemmatizer()


# In[46]:


# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words) 


# In[47]:


print(sentences)


# In[49]:


print(len(sentences))


# In[ ]:




