#!/usr/bin/env python
# coding: utf-8

# # **Applied Machine Learning Homework 5**
# **Due 12 Dec,2022 (Monday) 11:59PM EST**

# Your Name: Liwen Zhu
# 
# Your UNI: lz2512

# ### Natural Language Processing
# We will train a supervised model to predict if a movie has a positive or a negative review.

# ####  **Dataset loading & dev/test splits**

# **1.0) Load the movie reviews dataset from NLTK library**

# In[1]:


import nltk
nltk.download("movie_reviews")
import pandas as pd
from nltk.corpus import twitter_samples 
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop = stopwords.words('english')
import string
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


from nltk.corpus import movie_reviews


# In[3]:


negative_fileids = movie_reviews.fileids('neg')
positive_fileids = movie_reviews.fileids('pos')

pos_document = [(' '.join(movie_reviews.words(file_id)),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id) if category == 'pos']
neg_document = [(' '.join(movie_reviews.words(file_id)),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id) if category == 'neg']

# List of postive and negative reviews
pos_list = [pos[0] for pos in pos_document]
neg_list = [neg[0] for neg in neg_document]


# **1.1) Make a data frame that has reviews and its label**

# In[4]:


# code here
movie = pd.DataFrame(pos_document+neg_document,columns=["Review","Label"])


# **1.2 look at the class distribution of the movie reviews**

# In[5]:


# code here
movie["Label"].value_counts()


# **1.3) Create a development & test split (80/20 ratio):**

# In[6]:


# code here
X_dev,X_test,y_dev,y_test = train_test_split(movie["Review"],movie["Label"],test_size = 0.2)


# #### **Data preprocessing**
# We will do some data preprocessing before we tokenize the data. We will remove `#` symbol, hyperlinks, stop words & punctuations from the data. You may use `re` package for this. 

# **1.4) Replace the `#` symbol with '' in every review**

# In[7]:


# code here
X_dev = X_dev.str.replace('#','')
X_test = X_test.str.replace('#','')


# **1.5) Replace hyperlinks with '' in every review**

# In[8]:


# code here


# **1.6) Remove all stop words**

# In[9]:


# code here
for word in stop:
    X_dev = X_dev.str.replace(' '+word+' ',' ').replace(' '+word,' ').replace(word+' ',' ')
    X_test = X_test.str.replace(' '+word+' ',' ').replace(' '+word,' ').replace(word+' ',' ')


# **1.7) Remove all punctuations**

# In[10]:


# code here
for punctuation in string.punctuation:
    X_dev = X_dev.str.replace(punctuation,' ')
    X_test = X_test.str.replace(punctuation,' ')


# **1.8) Apply stemming on the development & test datasets using Porter algorithm**

# In[11]:


#code here
def stemSentence(sentece):
    porter = PorterStemmer()
    token_words = word_tokenize(sentece)
    stem_sentence = [porter.stem(word) for word in token_words]
    return " ".join(stem_sentence)

for index,sentence in X_dev.iteritems():
    X_dev[index] = stemSentence(sentence)
for index,sentence in X_test.iteritems():
    X_test[index] = stemSentence(sentence)


# #### **Model training**

# **1.9) Create bag of words features for each review in the development dataset**

# In[12]:


#code here
vector = CountVectorizer(stop_words = 'english')
X_dev_bow = vector.fit_transform(X_dev)
feature_names = vector.get_feature_names()
print(feature_names[:10])
print(feature_names[10000:10020])
print(feature_names[::5000])


# **1.10) Train a Logistic Regression model on the development dataset**

# In[13]:


#code here
lr = LogisticRegression().fit(X_dev_bow,y_dev)


# **1.11) Create TF-IDF features for each review in the development dataset**

# In[14]:


#code here
vector_tf = TfidfVectorizer()
X_dev_tf = vector_tf.fit_transform(X_dev)
feature_names_tf = vector_tf.get_feature_names()
print(feature_names_tf[:10])
print(feature_names_tf[10000:10020])
print(feature_names_tf[::5000])


# **1.12) Train the Logistic Regression model on the development dataset with TF-IDF features**

# In[15]:


#code here
lr_tf = LogisticRegression().fit(X_dev_tf,y_dev)


# **1.13) Compare the performance of the two models on the test dataset. Explain the difference in results obtained?**

# In[16]:


#code here
X_test_lr = vector.transform(X_test)
X_test_tf = vector_tf.transform(X_test)
print(f"The performance of bag of words is {lr.score(X_test_lr,y_test)}")
print(f"The performance of tf-idf is {lr_tf.score(X_test_tf,y_test)}")
print("The bag of words weights more on the high frequency word. On the other hand, the TF-IDF model weights more on the unique word. The two models weight words differetly, which causes the difference in results.")


# In[ ]:




