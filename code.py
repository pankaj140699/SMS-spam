#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[11]:


df=pd.read_csv('E:\\ds\\projects\\sms_spam\\spam.csv')


# In[12]:


df.head()


# In[14]:


df.shape


# In[20]:


df=df.drop_duplicates()
df.shape


# In[17]:


df.dtypes


# In[18]:


df.info()


# In[19]:


df=df.dropna()
df.shape


# In[21]:


df.head()


# In[22]:


df['Category']=df['Category'].map({'ham':0,'spam':1})


# In[23]:


df.head()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


sns.countplot(x='Category',data=df)


# In[26]:


## imbalance dataset


# In[28]:


only_spam = df[df['Category']==1]
print('Number of Spam records: {}'.format(only_spam.shape[0]))
print('Number of Ham records: {}'.format(df.shape[0]-only_spam.shape[0]))


# In[29]:


count = int((df.shape[0]-only_spam.shape[0])/only_spam.shape[0])
for i in range(0, count-1):
    df=pd.concat([df, only_spam])
df.shape


# In[30]:


sns.countplot(x='Category',data=df)


# In[35]:


df['word_count'] = df['Message'].apply(lambda x: len(x.split()))


# In[36]:


df.head()


# In[37]:


plt.subplot(1, 2, 1)
g = sns.distplot(a=df[df['Category']==0].word_count)
p = plt.title('Distribution of word_count for Ham messages')

# 1-row, 2-column, go to the second subplot
plt.subplot(1, 2, 2)
g = sns.distplot(a=df[df['Category']==1].word_count, color='red')
p = plt.title('Distribution of word_count for Spam messages')

plt.tight_layout()
plt.show()


# In[38]:


# spam has word count 20-30 and ham has 0-20


# In[95]:


# Creating feature contains_currency_symbol
def currency(x):
    currency_symbols = ['€', '$', '¥', '£', '₹']
    for i in x:
        if i in currency_symbols:
            return 1
    return 0
df['contains_currency_symbol'] = df['Message'].apply(currency)


# In[96]:


df.head()


# In[103]:


g = sns.countplot(x='contains_currency_symbol', data=df,hue='Category')
p = plt.title('Countplot for contain_currency')
p = plt.xlabel('Does SMS contain currency symbol?')
p = plt.ylabel('Count')
p = plt.legend(labels=['Ham', 'Spam'], loc=9)


# In[104]:


# Almost 1/3 of Spam messages contain currency symbols, and currency symbols are rarely used in Ham messages


# In[106]:


def numbers(x):
    for i in x:
        if ord(i)>=48 and ord(i)<=57:
            return 1
    return 0
df['contains_number'] = df['Message'].apply(numbers)


# In[107]:


df.head()


# In[108]:


plt.figure(figsize=(8,8))
g = sns.countplot(x='contains_number', data=df, hue='Category')
p = plt.title('Countplot for contain_numbers')
p = plt.xlabel('Does SMS contain number?')
p = plt.ylabel('Count')
p = plt.legend(labels=['Ham', 'Spam'], loc=9)


# In[110]:


#It is evident that most of the Spam messages contain numbers, and majority of the Ham messages donot contain numbers.


# In[111]:


# Importing essential libraries for performing NLP
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[113]:


# Cleaning the messages
corpus = []
wnl = WordNetLemmatizer()

for sms_string in list(df.Message):

  # Cleaning special character from the sms
  message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)

  # Converting the entire sms into lower case
  message = message.lower()

  # Tokenizing the sms by words
  words = message.split()

  # Removing the stop words
  filtered_words = [word for word in words if word not in set(stopwords.words('english'))]

  # Lemmatizing the words
  lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]

  # Joining the lemmatized words
  message = ' '.join(lemmatized_words)

  # Building a corpus of messages
  corpus.append(message)


# In[114]:


corpus[0:3]


# In[116]:


df.head()


# In[136]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names()

# Extracting independent and dependent variables from the dataset
X = pd.DataFrame(vectors, columns=feature_names)
Y=df['Category']


# In[161]:

import pickle
pickle.dump(tfidf, open('E:/ds/projects/sms_spam/cv-transform.pkl', 'wb'))




# In[142]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# In[144]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[145]:


# In[151]:


# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
cv = cross_val_score(rf, X, Y, scoring='f1', cv=10)
print('--- Average F1-Score for Random Forest model: {} ---'.format(round(cv.mean(), 3)))
print('Standard Deviation: {}'.format(round(cv.std(), 3)))


# In[152]:



# Classification report for Random Forest model
rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('--- Classification report for Random Forest model ---')
print(classification_report(y_test, y_pred))


# In[153]:



def predict_spam(sample_message):
  sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
  sample_message = sample_message.lower()
  sample_message_words = sample_message.split()
  sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
  final_message = [wnl.lemmatize(word) for word in sample_message_words]
  final_message = ' '.join(final_message)

  temp = tfidf.transform([final_message]).toarray()
  return rf.predict(temp)


# In[154]:


# Prediction 1 - Lottery text message
sample_message = 'IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.'

if predict_spam(sample_message):
  print('Gotcha! This is a SPAM message.')
else:
  print('This is a HAM (normal) message.')


# In[155]:


# Prediction 2 - Casual text chat
sample_message = 'Came to think of it. I have never got a spam message before.'

if predict_spam(sample_message):
  print('Gotcha! This is a SPAM message.')
else:
  print('This is a HAM (normal) message.')


# In[156]:


sample_message = 'Sam, your rent payment for Jan 19 has been received. $1,300 will be drafted from your Wells Fargo Account ******0000 within 24-48 business hours. Thank you!'

if predict_spam(sample_message):
  print('Gotcha! This is a SPAM message.')
else:
  print('This is a HAM (normal) message.')


# In[157]:


# Predicting values 4 - Feedback message
sample_message = 'Tammy, thanks for choosing Carl’s Car Wash for your express polish. We would love to hear your thoughts on the service. Feel free to text back with any feedback. Safe driving!'

if predict_spam(sample_message):
  print('Gotcha! This is a SPAM message.')
else:
  print('This is a HAM (normal) message.')


# In[158]:



# In[159]:


import pickle
file=open('E:/ds/projects/sms_spam/random_forest_classification_model.pkl','wb') # write byte mode
pickle.dump(rf,file)


# In[ ]:




