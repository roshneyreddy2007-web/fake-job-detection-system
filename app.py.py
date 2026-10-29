# -*- coding: utf-8 -*-
"""Fake_Job_posting_Detection.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib

path="/content/drive/MyDrive/Fake_Job_Posting_Project/fake_job_postings.csv"

df=pd.read_csv(path)

df.head()

df.info()

df.shape

df.columns

df.describe(include="all")

df.fillna("",inplace=True)

sns.countplot(x='fraudulent',data=df)

plt.show()

text=df["title"]+" "+df["company_profile"]+" "+df["description"]+" "+df["requirements"]+" "+df["benefits"]

target=df["fraudulent"]

stop_words=set(stopwords.words("english"))

def clean_text(text):

    text=text.lower()

    text=re.sub(r'http\S+'," ",text)

    text=re.sub(r'[^a-zA-Z]'," ",text)

    text=text.split()

    text=[word for word in text if word not in stop_words]

    return " ".join(text)

text=text.apply(clean_text)

tfidf=TfidfVectorizer(max_features=5000)

X=tfidf.fit_transform(text)

y=target

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42)

lr=LogisticRegression()

lr.fit(X_train,y_train)

pred1=lr.predict(X_test)

print(accuracy_score(y_test,pred1))

print(classification_report(y_test,pred1))

nb=MultinomialNB()

nb.fit(X_train,y_train)

pred2=nb.predict(X_test)

print(accuracy_score(y_test,pred2))

print(classification_report(y_test,pred2))

rf=RandomForestClassifier()

rf.fit(X_train,y_train)

pred3=rf.predict(X_test)

print(accuracy_score(y_test,pred3))

print(classification_report(y_test,pred3))

results=pd.DataFrame({

'Model':['Logistic Regression',
         'Naive Bayes',
         'Random Forest'],

'Accuracy':[

accuracy_score(y_test,pred1),

accuracy_score(y_test,pred2),

accuracy_score(y_test,pred3)]

})

results

results.to_csv("/content/drive/MyDrive/Fake_Job_Posting_Project/model_results.csv",index=False)

cm=confusion_matrix(y_test,pred1)

plt.figure(figsize=(6,5))

sns.heatmap(cm,
annot=True,
fmt='d',
cmap='Blues')

plt.savefig("/content/drive/MyDrive/Fake_Job_Posting_Project/confusion_matrix.png")

plt.show()

joblib.dump(lr,
"/content/drive/MyDrive/Fake_Job_Posting_Project/fake_job_model.pkl")

joblib.dump(tfidf,
"/content/drive/MyDrive/Fake_Job_Posting_Project/tfidf.pkl")

model=joblib.load("/content/drive/MyDrive/Fake_Job_Posting_Project/fake_job_model.pkl")

vectorizer=joblib.load("/content/drive/MyDrive/Fake_Job_Posting_Project/tfidf.pkl")

sample="""

Software Engineer

Python

Machine Learning

TensorFlow

Remote Job

Excellent Salary

"""

sample=clean_text(sample)

sample=vectorizer.transform([sample])

prediction=model.predict(sample)

if prediction[0]==0:
    print("Legitimate Job Posting")
else:
    print("Fraudulent Job Posting")

import joblib

# Save the trained model
joblib.dump(model, "model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "tfidf.pkl")

from google.colab import files

# Download the files
files.download("model.pkl")
files.download("tfidf.pkl")
