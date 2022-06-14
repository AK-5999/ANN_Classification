#!/usr/bin/env python
# coding: utf-8

# ### Topic: ANN (Artificial Neural Network)
# ### Problem: Churn or not Churn
# ### Author: Aman_Kumar
# ### Domain:  Deep Learning

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns


# In[2]:


import keras as krs


# In[3]:


df =  pd.read_csv("E:\COLLINS\Rajdeep\DeepLearning\ANN\BankCoustomer.csv")


# In[4]:


df


# In[5]:


df.head


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


sns.pairplot(data=df, x_vars=["CreditScore","Age","Tenure","NumOfProducts", "HasCrCard", "IsActiveMember","EstimatedSalary"], y_vars ="Exited")  # Plot with numeric dATA


# ### Graphical info
# : Above graphs as showing that the most of the information or customer or output variable is either 1 or 0 which means it's a categorical out.
# : because it is a categorical output then we will classified the objects into classes.
# : In machine Learning, we can use Logistic Regression, Decision Tree , Random Forest etc.
# : Here, we will use ANN (Artificial Neural Network) to predict wheater a customer will churn or not.

# In[10]:


x = df.iloc[:,3:13]
y=df.iloc[:,13]


# In[11]:


x


# In[12]:


y


# In[13]:


Geography = pd.get_dummies(x["Geography"], drop_first = True)


# In[14]:


gender = pd.get_dummies(x["Gender"], drop_first = True)


# In[15]:


Geography


# In[16]:


gender


# In[17]:


x = x.drop(["Geography","Gender"], axis = 1)


# In[18]:


x


# In[19]:


x = pd.concat([x,Geography,gender], axis=1)


# In[20]:


x


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state = 0)


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


Y_train


# In[26]:


Y_test


# In[27]:



from sklearn.preprocessing import StandardScaler


# In[28]:



sc = StandardScaler()


# In[29]:


sc


# In[30]:


X_train = sc.fit_transform(X_train)


# In[31]:


X_test= sc.fit_transform(X_test)


# In[32]:


X_train


# In[33]:


X_test


# In[34]:


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages


# In[35]:


from keras.models import Sequential


# In[36]:


from keras.layers import Dense


# In[37]:


from keras.layers import LeakyReLU, PReLU, ELU


# In[38]:


from keras.layers import Dropout


# In[39]:


# Initialising the ANN
classifier = Sequential()


# In[40]:


classifier


# In[41]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))


# In[42]:


# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))


# In[43]:


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))


# In[44]:


# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[45]:


# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, Y_train,validation_split = 0.33, batch_size = 10, epochs = 100)


# In[46]:


print(model_history.history.keys())


# In[47]:


# summarize history for accuracy
plt.plot(model_history.history['accuracy'])


# In[48]:


plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[49]:


plt.plot(model_history.history['loss'])


# In[50]:



plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[51]:


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[52]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


# In[53]:


cm


# In[54]:


# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,Y_test)


# In[55]:


score


# In[ ]:




