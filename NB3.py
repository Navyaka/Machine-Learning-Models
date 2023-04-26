#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder,LabelBinarizer


# In[10]:


#Reading csv file into df dataframe
df = pd.read_csv('Airline_Delay_Cause_2.csv')
print(df)


# In[11]:


Month = df['Month'].values.reshape(-1,1)
Carrier = df['Carrier'].values.reshape(-1,1) 
Airport = df['Airport'].values.reshape(-1,1)


# In[12]:


Month.shape,Carrier.shape


# In[13]:


def preprocess():
    # Using ordinal encoder to convert the categories in the range from 0 to n-1
    mon_enc = OrdinalEncoder()
    month_ = mon_enc.fit_transform(Month)

    car_enc = OrdinalEncoder()
    carrier_ = car_enc.fit_transform(Carrier)

    ap_enc = OrdinalEncoder()
    airport_ = ap_enc.fit_transform(Airport)
    # Stacking all the features
    X = np.column_stack((month_,carrier_,airport_))
    # Changing the type to int
    X = X.astype(int)
    # Doing one hot encoding on the target data
    y = df['Cancelled']
    lb = LabelBinarizer()
    y_ = lb.fit_transform(y)
    if y_.shape[1] == 1:
        y_ = np.concatenate((1 - y_, y_), axis=1)
    return X,y_,lb.classes_


# In[14]:


X,y,classes = preprocess()
X.shape, y.shape


# In[15]:


def counts_based_onclass(X,y):
    
    # No of feature
    n_features = X.shape[1]
    # No of classes
    n_classes = y.shape[1]
    
    count_matrix = []
    # For each feature
    for i in range(n_features):
        count_feature = []
        # Get that particuar feature from the dataset
        X_feature = X[:,i]
        # For each class
        for j in range(n_classes):
            # Get the datapoints that belong to the class - j
            mask = y[:,j].astype(bool)
            # Using masking filter out the datapoints that belong to this class- j in the given feature - i
            # Using bincount -- count all the different categories present in the given feature
            counts = np.bincount(X_feature[mask])
            
            count_feature.append(counts)
            
        count_matrix.append(np.array(count_feature))
        # Finding the count of datapoints beloging to each class -- we will use it to calculate prior probabilities.
        class_count = y.sum(axis=0)
        
    return count_matrix,n_features,n_classes,class_count


# In[16]:


count_matrix,n_features,n_classes,class_count = counts_based_onclass(X,y)


# In[17]:


count_matrix


# In[18]:


n_features


# In[19]:


def calculate_likelihood_probs(count_matrix,alpha,n_features):
    log_probabilities = []
    for i in range(n_features):
        num = count_matrix[i] + alpha
        den = num.sum(axis = 1).reshape(-1,1)
        log_probability = np.log(num) - np.log(den)
        log_probabilities.append(log_probability)
    return log_probabilities


# In[20]:


def calculate_prior_probs(class_count):
    
    num = class_count
    den = class_count.sum()
    
    return np.log(num)-np.log(den)


# In[21]:


prior_probs = calculate_prior_probs(class_count)


# In[22]:


log_probs = calculate_likelihood_probs(count_matrix,1,n_features)


# In[23]:


log_probs


# In[24]:


def predict(query_point,log_probs,prior_probs):
    
    # Intializing an empty array
    probs = np.zeros((1,n_classes))
    # For each feature
    for i in range(n_features):
        # Get the category_id of the feature - i from the query_point
        category = query_point[i]
        # Fetch the corresponding log_probability table and add continue to add them for all the features
        probs+=log_probs[i][:,category]
    # Finally add posterior probability
    probs+=prior_probs
    # Finding the maximum of the probabilities and fetching the corresponding class
    return classes[np.argmax(probs)]


# In[25]:


from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X, df['Cancelled'])
print('Sklearn feature log-probabilities\n',clf.feature_log_prob_)
print('Manually implemented likelihood probabilities\n',log_probs)

print('Sklearn feature prior-probabilities\n',clf.class_log_prior_)
print('Manually implemented prior probabilities\n',prior_probs)

print()
print('Sklearn predict',clf.predict(X[4:5]))
print('Manual predict',predict(X[4],log_probs,prior_probs))

