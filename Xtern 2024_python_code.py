#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

def prepare_data(filepath):
    """Prepares the data for training a machine learning model.
    
    Args:
    - filepath (str): Path to the CSV dataset file.

    Returns:
    - X (array-like): Processed feature matrix.
    - y (array-like): Target vector.
    """
    
    # Load the dataset from the given filepath
    data = pd.read_excel("C:\\Users\\wcp\\Documents\\filepath.xlsx")
    
    # Impute missing values using forward fill
    data.fillna(method='ffill', inplace=True)
    
    # One-hot encode the categorical features
    encoder = OneHotEncoder(drop='first')
    encoded_features = encoder.fit_transform(data[['Year', 'Major', 'University']]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Year', 'Major', 'University']))
    
    # Append the encoded features to the original dataset
    data = pd.concat([data, encoded_df], axis=1)
    
    # Drop the original categorical columns and the 'Order' column to create feature matrix X
    X = data.drop(columns=['Year', 'Major', 'University', 'Order'])
    
    # Label encode the target variable 'Order' to create target vector y
    le = LabelEncoder()
    y = le.fit_transform(data['Order'])
    
    # Standardize the feature matrix X for better model performance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


# In[34]:


from sklearn.linear_model import LogisticRegression
import pickle

def train_and_pickle_model(X, y, model_path):
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save/pickle the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Using the data preparation function
X, y = prepare_data('path_to_dataset.csv')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Using the model training function
model = train_and_pickle_model(X_train, y_train, 'model.pkl')

# Testing the model
y_pred = model.predict(X_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[ ]:




