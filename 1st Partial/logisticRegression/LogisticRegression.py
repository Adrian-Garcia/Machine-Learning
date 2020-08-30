#!/usr/bin/env python
# coding: utf-8

# ### Importing Required Libraries

# In[49]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# ### Loading data

# In[50]:


# Pima Indians Diabetes Database
# Predict the onset of diabetes based on diagnostic measures
# https://www.kaggle.com/uciml/pima-indians-diabetes-database

# load dataset
pima = pd.read_csv("diabetes.csv")


# In[51]:


pima.head()


# In[53]:


#pima.info()


# ### Feature Selection

# In[54]:


#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = pima[feature_cols] # Features (independent variables)
y = pima.Outcome # Target variable


# ### Splitting Data

# In[55]:


# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) 


# ### Building the Logistic Regression Model

# In[56]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

# Predict labels
y_pred=logreg.predict(X_test)


# ### Evaluating the Model

# In[57]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[58]:


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[60]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[24]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# ### Explaining the Logistic Regression Model

# In[61]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())


# ### Using the most relevant features

# In[77]:


feature_cols = ['BMI', 'Age','Glucose']
X = pima[feature_cols] # Features (independent variables)
y = pima.Outcome # Target variable


# In[78]:


# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) 


# In[79]:


import statsmodels.api as sm
logit_model_2=sm.Logit(y_train,X_train)
result_2=logit_model_2.fit()
print(result_2.summary2())


# In[80]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

# Predict labels
y_pred=logreg.predict(X_test)


# In[81]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[82]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




