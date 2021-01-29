# imports utiles
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# commande à enlever si ipymbl n'est pas installé
# %matplotlib widget


# # Préparation

# ## Lecture des données

# In[32]:


df = pd.read_json("data.json").set_index('Id')
labels = pd.read_csv("label.csv", index_col='Id')
category_names = pd.read_csv("categories_string.csv")['0'].to_dict()

df['label'] = labels['Category']
df['job'] = labels['Category'].map(category_names)
# df["description_lower"] = [x.lower() for x in df.description]

df


# ## Définition de fonctions

# ### Fairness

# In[33]:


def disparate_impact(df):
    counts = df.groupby(['label', 'gender']).size().unstack('gender')
    di = counts[['M', 'F']].max(axis='columns') / counts[['M', 'F']].min(axis='columns')
    di.index = di.index.map(category_names)
    return di.sort_values(ascending=False)


# ### Utilities

# In[34]:


from sklearn import metrics
def show_confusion_matrix(y_true, y_pred):
    _=metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_true, y_pred)).plot()


# In[35]:


import re
def score_report(y_true, y_pred):
    r = classification_report(y_true, y_pred, target_names=category_names.values()).split("\n")
    for line in [re.sub("\\s+[^ ]+$","",e) for e in r[:1] + r[31:-1]]: print(line)


# In[36]:


import contextlib
import joblib
import os


@contextlib.contextmanager
def persist(name:str, dump:bool=True):
    if type(name) != str:
        raise TypeError("name must be a string")
    try:
        try: 
            var = globals()[name]
            yield var
        except KeyError:
            if os.path.isfile(name):
                print("loaded " + name)
                yield joblib.load(name)
            else: yield
    finally:
        if dump:
            try:
                var = globals()[name]
                if var != None and not os.path.isfile(name):
                    joblib.dump(var, name)
                    print("dumped " + name)
            except KeyError: raise RuntimeError(name + " is not defined")


# # Visualisation

# In[37]:


d=df.groupby(['job', 'gender']).size().unstack('gender')
d['sum']=d['F']+d['M']
_=d.sort_values(by='sum').drop(columns='sum').plot.barh(figsize=(15,8), stacked=True)


# # Représentation de texte

# ## TF - IDF

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer

with persist('vectorizer') as vectorizer:
    if vectorizer == None:
        vectorizer = TfidfVectorizer(stop_words={'english'}).fit(df.description)

with persist('X') as X:
    if X == None:
        X = vectorizer.transform(df.description)
X.shape


# # Entraînement

# In[39]:


X_train, X_test, Y_train, Y_test = train_test_split(X, df.label, random_state=0)


# ## KNN

# ## SGD

# In[40]:


N_train = 5_000
x=X[:N_train,:]
x_train, x_test, y_train, y_test = train_test_split(x, df.label.head(N_train), random_state=0)


# In[41]:


from sklearn.linear_model import SGDClassifier
# Always scale the input. The most convenient way is to use a pipeline.
with persist('clf_sgd') as clf_sgd:
    if clf_sgd == None:
        clf_sgd = SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1).fit(x_train, y_train)

y_pred_train = clf_sgd.predict(x_train)
y_pred_test = clf_sgd.predict(x_test)


# In[42]:


print("train")
score_report(y_train, y_pred_train)
print("test")
score_report(y_test, y_pred_test)


# In[43]:


# show_confusion_matrix(y_test, y_pred_test)


# In[50]:


predict = pd.DataFrame({
    'pred': y_pred_test,
    'true': y_test
})
predict['description'] = df.iloc[predict.index].description
predict
joblib.dump(predict.to_csv(), 'predict.csv')

