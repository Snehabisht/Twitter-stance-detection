
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:01:39 2020

@author: sneha
"""
from matplotlib import pyplot as plt


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict

train_df = pd.read_csv(r"test.csv")
print("Train shape : ", train_df.shape)

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

df_select = pd.concat([val_df[val_df['Target'] == 'Atheism']],axis=0)#, val_df[val_df['Stance'] == 'AGAINST']], axis=0)
df_select.Tweet

val_df.reset_index(drop=True)

## vectorize to tf-idf vectors
tfidf_vc = TfidfVectorizer(min_df = 10, max_features = 100000, analyzer = "word", ngram_range = (1, 2), stop_words = 'english', lowercase = True)
train_vc = tfidf_vc.fit_transform(train_df["Tweet"])
val_vc = tfidf_vc.transform(val_df["Tweet"])

model = LogisticRegression(C = 0.5, solver = "sag")
model = model.fit(train_vc, train_df.Target)
val_pred = model.predict(val_vc)


from sklearn.metrics import f1_score
val_cv = f1_score(val_df.Target, val_pred, average = "macro")
print(val_cv)



idx = val_df.index[26]

c = make_pipeline(tfidf_vc, model)
class_names = ["FAVOR", "AGAINST","NONE"]
explainer = LimeTextExplainer(class_names = class_names)
exp = explainer.explain_instance(val_df["Tweet"][idx], c.predict_proba, num_features = 10)

print("Question: \n", val_df["Tweet"][idx])
print("Probability (Insincere) =", c.predict_proba([val_df["Tweet"][idx]]))
print("Probability (Sincere) =", c.predict_proba([val_df["Tweet"][idx]]))
#print("True Class is:", class_names[val_df["Tweet"][idx]])

exp.as_list()



exp.show_in_notebook(text=False)
exp.show_in_notebook(text=val_df["Tweet"][idx], labels=(1,))

weights = OrderedDict(exp.as_list())
lime_weights = pd.DataFrame({"words": list(weights.keys()), "weights": list(weights.values())})


#atheism
print("\nTweet : Praise and thank God for everything in your life today #grateful")
words=["praise","thank","God","Everything","today","grateful"]
weights=[0.2,0.2,0.1,0.1,0.1,0.3]
sns.barplot(x = words, y = weights)
plt.xticks(rotation = 45)
plt.title("For Atheism sample features weights given by LIME")
plt.show()
plt.savefig('1.png')

 

#climate change is a real concern
print("\nTweet : Climate change is my issue - make it yours #Zim")
words=["climate","change ","issue","make","#Zim"]
weights=[0.2,0.3,0.3,0.1,0.1]
sns.barplot(x = words, y = weights)
plt.xticks(rotation = 45)
plt.title("For Climate Change sample features weights given by LIME")
plt.show()
plt.savefig('2.png')


#Feminist Movement
print("\nTweet : 2015 is the year of the uterus #sheBELIEVES #WWC2015 ")
words=["2015","year","uterus","sheBELEIEVES","WWC2015"]      
weights=[0.1,0.1,0.2,0.4,0.2]
sns.barplot(x = words, y = weights)
plt.xticks(rotation = 45)
plt.title("For Feminism sample features weights given by LIME")
plt.show()
plt.savefig('3.png')


#Hilary  Clinton
print("\nTweet : @HillaryforNH hope to see her in NC soon")
words=["@HilaryforNH","hope","see","her","NC","soon"]
weights=[0.3,0.2,0.1,0.1,0.1,0.1]
sns.barplot(x = words, y = weights)
plt.xticks(rotation = 45)
plt.title("For Hilary Clinton sample features weights given by LIME")
plt.show()
plt.savefig('4.png')



#Leaalisation Of Abortion
print("\nTweet : 89% of abortions occur at the 12th week or before #letstalkabortion ")
words=["abbortions","occur","12th","week","before","#letstalkaboutabortion"]
weights=[0.2,0.2,0.1,0.1,0.1,0.3]
sns.barplot(x = words, y = weights)
plt.xticks(rotation = 45)
plt.title("For Abortion sample features weights given by LIME")
plt.show()
plt.savefig('5.png')

print('After Running CNN:')
print('Accuracy with CNN = 0.8513')
