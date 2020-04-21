#!/usr/bin/env python
# coding: utf-8

# # Selecting the Right Threshold value using ROC Curve

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# ROC curve and auc Curve
from sklearn.datasets import make_classification 


# In[2]:


from sklearn.model_selection import train_test_split
X,y = make_classification(n_samples=2000, n_classes=2,
                         weights=[1,1], random_state=1)


# In[3]:


print(X.shape)
print(y.shape)


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[5]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# #Random Forest

# In[6]:


## Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
#model
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
#prediction on train
ytrain_pred = rf_model.predict_proba(X_train)
print("Rf train roc-auc:{}".format(roc_auc_score(y_train,ytrain_pred[:,1])))
# preddiction on test 
ytest_pred = rf_model.predict_proba(X_test)
print("Rf test roc-auc:{}".format(roc_auc_score(y_test,ytest_pred[:,1])))


# In[7]:


ytrain_pred = rf_model.predict_proba(X_train)
ytrain_pred


# In[8]:


print(ytest_pred)


# #Logistic Regression

# In[9]:


from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression()
log_classifier.fit(X_train,y_train)
# Predcition on Train
ytrain_pred = log_classifier.predict_proba(X_train)
print("Logistic train roc-auc curver: {}".format(roc_auc_score(y_train,ytrain_pred[:,1])))
# Predictio on Test
ytest_pred = log_classifier.predict_proba(X_test)
print("Logistic test roc-auc curve: {}".format(roc_auc_score(y_test,ytest_pred[:,1])))


# #Adaboost Classifier

# In[37]:


from sklearn.ensemble import AdaBoostClassifier
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train,y_train)

# Prediction on Train
ytrain_pred = ada_classifier.predict_proba(X_train)
print("Adaboost train roc-auc: {}".format(roc_auc_score(y_train,ytrain_pred[:,1])))

# Prediction on Test
ytest_pred = ada_classifier.predict_proba(X_test)
print("Adaboost test roc-auc :{}".format(roc_auc_score(y_test,ytest_pred[:,1])))


# # KNN CLassifier

# In[38]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)
# Prediction on train
ytrain_pred = knn_classifier.predict_proba(X_train)
print("Adaboost train roc-auc :{}".format(roc_auc_score(y_train,ytrain_pred[:,1])))

# Prediction on test
ytest_pred = knn_classifier.predict_proba(X_test)
print("Adaboost test roc-auc :{}".format(roc_auc_score(y_test,ytest_pred[:,1])))


# #No we will focus on selecting the best threshold for maximum accuracy

# In[39]:


pred = []
for model in [rf_model, log_classifier, ada_classifier,knn_classifier]:
    pred.append(pd.Series(model.predict_proba(X_test)[:,1]))
final_prediction = pd.concat(pred, axis=1).mean(axis=1)
print("Ensemble test roc-auc: {}".format(roc_auc_score(y_test, final_prediction)))


# In[40]:


pd.concat(pred,axis=1)


# In[41]:


final_prediction


# In[43]:


# Calculate the ROC Curve
fpr , tpr , thresholds = roc_curve(y_test, final_prediction)
thresholds


# In[45]:


from sklearn.metrics import accuracy_score
accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(final_prediction>thres,1,0)
    accuracy_ls.append(accuracy_score(y_test,y_pred,normalize = True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],axis =1)
accuracy_ls.columns = ["Tresholds", "Accuracy"]
accuracy_ls.sort_values(by="Accuracy", ascending = False, inplace = True)
accuracy_ls.head()


# In[46]:


accuracy_ls


# In[47]:


def plot_roc_curve(fpr,tpr):
    plt.plot(fpr, tpr , color = "orange", label = "ROC")
    plt.plot([0,1],[0,1], color = "darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


# In[48]:


plot_roc_curve(fpr,tpr)

