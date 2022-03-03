#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved 28/02/2022 from https://www.kaggle.com/aaron7sun/stocknews.

import pandas as pd



# In[3]:


df=pd.read_csv('/home/pranoti/Downloads/Data.csv', encoding = "ISO-8859-1")

#append extra row to df
#df2 = {'Index': '4101', 'Date': '2022-02-28', 'Label': 1,
#      'Top1': 'Ukraine asks for ceasefire ahead of talks with Russia', 
#     'Top2': 'Would Putin press the nuclear button?', 
##
#      'Top4': 'The Ukraine crisis is a major challenge for China', 
#      'Top5': 'People have swapped their pens and keyboards for guns',
#      'Top6': 'My husband stayed to fight, but I had to leave',
#      'Top7': 'Russia doubles interest rate after rouble slumps', 
#      'Top8': 'Putin nuclear alert a distraction attempt UK says', 
#      'Top9': 'Ukrainian cities on alert after night of shelling', 
#      'Top10': 'Eight dead as Australian floods break records', 
#      'Top11': 'Indian students stuck in Ukraine desperate for help', 
#      'Top12': 'EU shuts its airspace to Russian planes',
#      'Top13': 'Refugee tearfully describes why she fled Ukraine', 
#      'Top14': 'UN report to show true scale of warming impacts',
#      'Top15': 'Should the West arm a Ukrainian resistance?',
#      'Top16': 'Ukrainian sailor tried to sink Russians yacht',
#      'Top17': 'BP to offload stake in Russian oil firm Rosneft',
#      'Top18': 'Ukrainian sailor tried to sink Russians yacht',
#      'Top19': 'BP to offload stake in Russian oil firm Rosneft',
#      'Top20': 'How badly will Russia be hit by new sanctions?',
#      'Top21': 'Kharkiv residents describe intense battle to defend city',
#      'Top22': 'Fifa orders Russia not to play under flag',
#      'Top23': 'Can Mayawati save her political future?',
#      'Top24': 'BBC World News TV',
#      'Top25': 'BBC World Service Radio'
       
#       }
#df = df.append(df2, ignore_index = True)


# In[4]:


df.head()


# In[6]:




train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']


# In[7]:


#feature engineering
# Removing punctuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)


# In[11]:




# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)


# In[12]:




' '.join(str(x) for x in data.iloc[1,0:25])


# In[13]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[14]:


headlines[0]


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[16]:


## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(1,4))
traindataset=countvector.fit_transform(headlines)

#Implement TF-IDF
#cv = TfidfVectorizer()
#traindataset = cv.fit_transform(headlines)


# In[ ]:


# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[ ]:


## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
#test_dataset = cv.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[ ]:


## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)


# In[ ]:


print(predictions[-1])

import matplotlib.pyplot as plt
%matplotlib inline


x=test['Date']
y1=test['Label']
y2=predictions
# plotting figures by creating aexs object
# using subplots() function
fig, ax = plt.subplots(figsize = (10, 5))
plt.title('Example of Two Y labels')
 
# using the twinx() for creating another
# axes object for secondary y-Axis
ax2 = ax.twinx()
ax.plot(x, y1, color = 'g', marker="o")
ax2.plot(x, y2, color = 'b', marker="*")
 
# giving labels to the axises
ax.set_xlabel('Date', color = 'r')
ax.set_ylabel('Real', color = 'g')
 
# secondary y-axis label
ax2.set_ylabel('Predicted', color = 'b')
 
# defining display layout
plt.tight_layout()
 
# show plot
plt.show()

# In[ ]:




