#!/usr/bin/env python
# coding: utf-8

# # Lab 3: ML Life Cycle: Modeling

# In[3]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Decision Trees (DTs) and KNNs have many similarities. They are models that are fairly simple and intuitive to understand, can be used to solve both classification and regression problems, and are non-parametric models, meaning that they don't assume a particular relationship between the features and the label prior to training. However, KNNs and DTs each have their own advantages and disadvantages. In addition, one model may be better suited than the other for a particular machine learning problem based on multiple factors, such as the size and quality of the data, the problem-type and the hyperparameter configuration. For example, KNNs require feature values to be scaled, whereas DTs do not. DTs are also able to handle noisy data better than KNNs. 
# 
# Often times, it is beneficial to train multiple models on your training data to find the one that performs the best on the test data. 

# In this lab, you will continue practicing the modeling phase of the machine learning life cycle. You will train Decision Trees and KNN models to solve a classification problem. You will experiment training multiple variations of the models with different hyperparameter values to find the best performing model for your predictive problem. You will complete the following tasks:
#     
#     
# 1. Build your DataFrame and define your ML problem:
#     * Load the Airbnb "listings" data set
#     * Define the label - what are you predicting?
#     * Identify the features
# 2. Prepare your data:
#     * Perform feature engineering by converting categorical features to one-hot encoded values
# 3. Create labeled examples from the data set
# 4. Split the data into training and test data sets
# 5. Train multiple decision trees and evaluate their performances:
#     * Fit Decision Tree classifiers to the training data using different hyperparameter values per classifier
#     * Evaluate the accuracy of the models' predictions
#     * Plot the accuracy of each DT model as a function of hyperparameter max depth
# 6. Train multiple KNN classifiers and evaluate their performances:
#     * Fit KNN classifiers to the training data using different hyperparameter values per classifier
#     * Evaluate the accuracy of the models' predictions
#     * Plot the accuracy of each KNN model as a function of hyperparameter $k$
# 7. Analysis:
#    * Determine which is the best performing model 
#    * Experiment with other factors that can help determine the best performing model

# ## Part 1. Build Your DataFrame and Define Your ML Problem

# #### Load a Data Set and Save it as a Pandas DataFrame
# 

# We will work with a new preprocessed, slimmed down version of the Airbnb NYC "listings" data set. This version is almost ready for modeling, with missing values and outliers taken care of. Also note that unstructured fields have been removed.

# In[4]:


# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "airbnbData_Prepared.csv")


# <b>Task</b>: Load the data set into a Pandas DataFrame variable named `df`.

# In[5]:


# YOUR CODE HERE
df = pd.read_csv(filename)


# ####  Inspect the Data

# <b>Task</b>: In the code cell below, inspect the data in DataFrame `df` by printing the number of rows and columns, the column names, and the first ten rows. You may perform any other techniques you'd like to inspect the data.

# In[6]:


# YOUR CODE HERE
# Print number of rows and columns
print("Shape of the DataFrame:", df.shape)

# Print column names
print("\nColumn names:", df.columns.tolist())

# Display the first 10 rows
df.head(10)


# #### Define the Label
# 
# Assume that your goal is to train a machine learning model that predicts whether an Airbnb host is a 'super host'. This is an example of supervised learning and is a binary classification problem. In our dataset, our label will be the `host_is_superhost` column and the label will either contain the value `True` or `False`. Let's inspect the values in the `host_is_superhost` column.

# In[7]:


df['host_is_superhost']


# #### Identify Features

# Our features will be all of the remaining columns in the dataset. 
# 
# <b>Task:</b> Create a list of the feature names.

# In[8]:


# YOUR CODE HERE
# Create a list of all columns
all_columns = df.columns.tolist()

# Remove the label column
features = [col for col in all_columns if col != 'host_is_superhost']

# Print the list of feature names
print("Number of features:", len(features))
print("Feature list:\n", features)


# ## Part 2. Prepare Your Data
# 
# Many of the data preparation techniques that you practiced in Unit two have already been performed and the data is almost ready for modeling. The one exception is that a few string-valued categorical features remain. Let's perform one-hot encoding to transform these features into numerical boolean values. This will result in a data set that we can use for modeling.

# #### Identify the Features that Should be One-Hot Encoded

# **Task**: Find all of the columns whose values are of type 'object' and add the column names to a list named `to_encode`.

# In[9]:


# YOUR CODE HERE
# Find all object-type columns to one-hot encode
to_encode = df[features].select_dtypes(include='object').columns.tolist()

# Print the columns that will be encoded
print("Columns to one-hot encode:", to_encode)


# **Task**: Find the number of unique values each column in `to_encode` has:

# In[10]:


# YOUR CODE HERE
# Count unique values for each column in to_encode
for col in to_encode:
    print(f"{col}: {df[col].nunique()} unique values")


# #### One-Hot Encode the Features

# Instead of one-hot encoding each column using the NumPy `np.where()` or Pandas `pd.get_dummies()` functions, we can use the more robust `OneHotEncoder` transformation class from `sklearn`. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). 

# 
# <b><i>Note:</i></b> We are working with `sklearn` version 0.22.2. You can find documentation for the `OneHotEncoder` class that that corresponds to our version of `sklearn` [here](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OneHotEncoder.html). When choosing which features of the  `OneHotEncoder` class to use, do not use features that have been introduced in newer versions of `sklearn`. For example, you should specify the parameter `sparse=False` when calling `OneHotEncoder()` to create an encoder object. The documentation notes that the latest version of `sklearn` uses the `sparse_ouput` parameter instead of `sparse`, but you should stick with `sparse`.
# 
# <b>Task</b>: Refer to the documenation and follow the instructions in the code cell below to create one-hot encoded features.

# In[13]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Step 1: Create the encoder
enc = OneHotEncoder(sparse=False)

# Step 2: Fit and transform
encoded_array = enc.fit_transform(df[to_encode])

# Step 3: Convert to DataFrame using get_feature_names (for older sklearn)
df_enc = pd.DataFrame(encoded_array, columns=enc.get_feature_names(to_encode))

# Optional: Inspect
df_enc.head()


# Let's inspect our new DataFrame `df_enc` that contains the one-hot encoded columns.

# In[14]:


df_enc.head()


# Notice that the column names are numerical. 
# 
# <b>Task:</b> Complete the code below to reinstate the original column names.
# 

# In[15]:


# Use the method enc.get_feature_names() to resintate the original column names. 
# Call the function with the original two column names as arguments.
# Save the results to 'df_enc.columns'

df_enc.columns = enc.get_feature_names(to_encode) # YOUR CODE HERE


# Let's inspect our new DataFrame `df_enc` once again.

# In[16]:


df_enc.head(10)


# <b>Task</b>: You can now remove the original columns that we have just transformed from DataFrame `df`.
# 

# In[17]:


# YOUR CODE HERE
df = df.drop(columns=to_encode)


# <b>Task</b>: You can now join the transformed features contained in `df_enc` with DataFrame `df`

# In[18]:


# YOUR CODE HERE
df = pd.concat([df, df_enc], axis=1)


# Glance at the resulting column names:

# In[19]:


df.columns


# ## Part 3. Create Labeled Examples from the Data Set 

# <b>Task</b>: Obtain the feature columns from DataFrame `df` and assign to `X`. Obtain the label column from DataFrame `df` and assign to `y`.
# 

# In[20]:


# YOUR CODE HERE
# Assign features and label
X = df.drop(columns=['host_is_superhost'])
y = df['host_is_superhost']


# In[21]:


print("Number of examples: " + str(X.shape[0]))
print("\nNumber of Features:" + str(X.shape[1]))
print(str(list(X.columns)))


# ## Part 4. Create Training and Test Data Sets

# <b>Task</b>: In the code cell below create training and test sets out of the labeled examples using Scikit-learn's `train_test_split()` function. Save the results to variables `X_train, X_test, y_train, y_test`.
# 
# Specify:
# 1. A test set that is one third (.33) of the size of the data set.
# 2. A seed value of '123'. 

# In[26]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)


# <b>Task</b>: Check the dimensions of the training and test datasets.

# In[27]:


# YOUR CODE HERE
# Print shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ## Part 5. Train Decision Tree Classifers and Evaluate their Performances

# The code cell below contains a function definition named `train_test_DT()`. This function should:
# 1. train a Decision Tree classifier on the training data (Remember to use ```DecisionTreeClassifier()``` to create a model object.)
# 2. test the resulting model on the test data
# 3. compute and return the accuracy score of the resulting predicted class labels on the test data. 
# 
# <b>Task:</b> Complete the function to make it work.

# In[28]:


def train_test_DT(X_train, X_test, y_train, y_test, depth, leaf=1, crit='entropy'):
    
    # YOUR CODE HERE
    # Create and train the Decision Tree model
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, criterion=crit, random_state=123)
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Compute and return accuracy
    acc = accuracy_score(y_test, y_pred)
    return acc


# #### Train Two Decision Trees and Evaluate Their Performances

# <b>Task:</b> Use your function to train two different decision trees, one with a max depth of $8$ and one with a max depth of $32$. Print the max depth and corresponding accuracy score.

# In[29]:


# YOUR CODE HERE
# Train with max depth = 8
acc_8 = train_test_DT(X_train, X_test, y_train, y_test, depth=8)
print(f"Max Depth: 8, Accuracy: {acc_8:.4f}")

# Train with max depth = 32
acc_32 = train_test_DT(X_train, X_test, y_train, y_test, depth=32)
print(f"Max Depth: 32, Accuracy: {acc_32:.4f}")


# #### Visualize Accuracy

# We will be creating multiple visualizations that plot a specific model's hyperparameter value (such as max depth) and the resulting accuracy score of the model.
# 
# To create more clean and maintainable code, we will create one visualization function that can be called every time a plot is needed. 
# 
# <b>Task:</b> In the code cell below, create a function called `visualize_accuracy()` that accepts two arguments:
# 
# 1. a list of hyperparamter values
# 2. a list of accuracy scores
# 
# Both lists must be of the same size.
# 
# Inside the function, implement a `seaborn` lineplot in which hyperparameter values will be on the x-axis and accuracy scores will be on the y-axis. <i>Hint</i>: You implemented a lineplot in this week's assignment.

# In[30]:


# YOUR CODE HERE
def visualize_accuracy(hyperparams, accuracies):
    plt.figure(figsize=(8,5))
    sns.lineplot(x=hyperparams, y=accuracies, marker='o')
    plt.title('Accuracy vs Hyperparameter Value')
    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Accuracy Score')
    plt.grid(True)
    plt.show()


# <b>Task</b>: Test your visualization function below by calling the function to plot the max depth values and accuracy scores of the two decision trees that you just trained.

# In[31]:


# YOUR CODE HERE
# Call the visualization function with your two depth/accuracy pairs
visualize_accuracy([8, 32], [acc_8, acc_32])


# <b>Analysis</b>: Does this graph provide a sufficient visualization for determining a value of max depth that produces a high performing model?

# Yes, this graph provides a clear and sufficient visualization for determining an optimal value of max_depth that produces a high-performing decision tree model.
# 
# From the plot, we observe that the model with a max_depth of 8 achieves a higher accuracy (~0.8335) compared to the model with a max_depth of 32 (~0.8070). This indicates that deeper trees do not always lead to better performance and may result in overfitting.
# 
# The graph helps visually confirm that a moderate tree depth generalizes better to unseen data and is more suitable for this classification problem.

# #### Train Multiple Decision Trees Using Different Hyperparameter Values and Evaluate Their Performances

# <b>Task:</b> Let's train on more values for max depth.
# 
# 1. Train six different decision trees, using the following values for max depth: $1, 2, 4, 8, 16, 32$
# 2. Use your visualization function to plot the values of max depth and each model's resulting accuracy score.

# In[32]:


# YOUR CODE HERE
# List of max_depth values to test
depths = [1, 2, 4, 8, 16, 32]

# List to store accuracy results
accuracies = []

# Train and evaluate a decision tree for each max_depth
for depth in depths:
    acc = train_test_DT(X_train, X_test, y_train, y_test, depth=depth)
    accuracies.append(acc)

# Plot the results
visualize_accuracy(depths, accuracies)


# <b>Analysis</b>: Analyze this graph. Pay attention to the accuracy scores. Answer the following questions in the cell below.<br>
# 
# How would you go about choosing the best model configuration based on this plot? <br>
# What other hyperparameters of interest would you want to tune to make sure you are finding the best performing model?

# How would you go about choosing the best model configuration based on this plot?
# I would choose the model configuration that gives the highest accuracy score on the test data. According to the graph, the model with a max_depth of 8 has the highest accuracy. After depth 8, the accuracy starts to decline, suggesting that deeper trees begin to overfit the training data. Therefore, depth 8 represents a good balance between bias and variance and is likely the best choice.
# 
# What other hyperparameters of interest would you want to tune to make sure you are finding the best performing model?
# Other hyperparameters worth tuning include:
# 
# - min_samples_split: Minimum number of samples required to split an internal node.
# - min_samples_leaf: Minimum number of samples required to be at a leaf node.
# - max_leaf_nodes: To limit the number of leaf nodes and prevent overfitting.
# - criterion: The function used to measure the quality of a split (e.g., "gini" vs. "entropy").
# - max_features: The number of features to consider when looking for the best split.
# - Performing cross-validation instead of a single train/test split would also give a more reliable estimate of model performance.

# ## Part 6. Train KNN Classifiers and Evaluate their Performances
# 

# The code cell below contains function definition named `train_test_knn()`. This function should:
# 1. train a KNN classifier on the training data (Remember to use ```KNeighborsClassifier()``` to create a model object).
# 2. test the resulting model on the test data
# 3. compute and return the accuracy score of the resulting predicted class labels on the test data. 
# 
# <i>Note</i>: You will train KNN classifiers using the same training and test data that you used to train decision trees.
# 
# <b>Task:</b> Complete the function to make it work.

# In[33]:


def train_test_knn(X_train, X_test, y_train, y_test, k):
    # Create the KNN classifier with the given number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model on the training data
    knn.fit(X_train, y_train)
    
    # Evaluate the model on the test data and return the accuracy
    return knn.score(X_test, y_test)


# #### Train Three KNN Classifiers and Evaluate Their Performances
# 
# <b>Task:</b> Use your function to train three different KNN classifiers, each with a different value for hyperparameter $k$: $3, 30$, and $300$. <i>Note</i>: This make take a second.
# 

# In[34]:


# YOUR CODE HERE
# Train 3 KNN classifiers with different values of k
acc_k3 = train_test_knn(X_train, X_test, y_train, y_test, k=3)
acc_k30 = train_test_knn(X_train, X_test, y_train, y_test, k=30)
acc_k300 = train_test_knn(X_train, X_test, y_train, y_test, k=300)


# <b>Task:</b> Now call the function `visualize_accuracy()` with the appropriate arguments to plot the results.

# In[35]:


# YOUR CODE HERE
# Plot the accuracy results for the three K values
visualize_accuracy([3, 30, 300], [acc_k3, acc_k30, acc_k300])


# #### Train Multiple KNN Classifiers Using Different Hyperparameter Values and Evaluate Their Performances

# <b>Task:</b> Let's train on more values for $k$.
# 
# 1. Array `k_range` contains multiple values for hyperparameter $k$. Train one KNN model per value of $k$
# 2. Use your visualization function to plot the values of $k$ and each model's resulting accuracy score.
# 
# <i>Note</i>: This make take a second.

# In[36]:


k_range = np.arange(1, 40, step = 3) 
k_range


# In[38]:


# YOUR CODE HERE
# Step 1: Train a KNN model for each k in k_range and store accuracies
knn_accuracies = []

for k in k_range:
    acc = train_test_knn(X_train, X_test, y_train, y_test, k)
    knn_accuracies.append(acc)

# Step 2: Visualize the accuracy scores
visualize_accuracy(k_range, knn_accuracies)


# ## Part 7. Analysis
# 
# 1. Compare the performance of the KNN model relative to the Decision Tree model, with various hyperparameter values. Which model performed the best (yielded the highest accuracy score)? Record your findings in the cell below.
# 
# 2. We tuned hyperparameter $k$ for KNNs and hyperparamter max depth for DTs. Consider other hyperparameters that can be tuned in an attempt to find the best performing model. Try a different combination of hyperparamters for both KNNs and DTs, retrain the models, obtain the accuracy scores and record your findings below. 
# 
#     <i>Note:</i> You can consult Scikit-learn documentation for both the [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) class and the [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) class to see how specific hyperparameters are passed as parameters to the model object.

# 1. Based on the accuracy scores, the Decision Tree model with a max_depth of 8 performed the best with an accuracy of approximately 0.8335. In contrast, the best performing KNN model had a slightly lower accuracy of approximately 0.776, achieved with k = 22. While KNN shows consistent accuracy across many k values, Decision Trees can reach higher accuracy more quickly, especially with a well-chosen depth. However, Decision Trees can also overfit when max_depth is too large (e.g., 32).
# 
# 2. To find the best performing model, we can consider tuning additional hyperparameters:
# For KNN:
# - weights: try 'uniform' vs 'distance'
# - p: explore distance metrics (e.g., p=1 for Manhattan, p=2 for Euclidean)
# 
# For Decision Trees:
# - min_samples_split: the minimum number of samples required to split an internal node.
# - min_samples_leaf: the minimum number of samples required to be at a leaf node.
# - max_leaf_nodes: to limit the complexity of the tree.
# - criterion: try 'gini' vs 'entropy'.
# Tuning these hyperparameters using techniques like GridSearchCV or RandomizedSearchCV could further improve the models' performance and generalization.
# 

# In[40]:


#1. Based on the accuracy scores, the Decision Tree model with a max_depth of 8 performed the best with an accuracy of approximately 0.8335. In contrast, the best performing KNN model had a slightly lower accuracy of approximately 0.776, achieved with k = 22.
#While KNN shows consistent accuracy across many k values, Decision Trees can reach higher accuracy more quickly, especially with a well-chosen depth. However, Decision Trees can also overfit when max_depth is too large (e.g., 32).

#2.To find the best performing model, we can consider tuning additional hyperparameters:
#For KNN:
#- weights: try 'uniform' vs 'distance'
#- p: explore distance metrics (e.g., p=1 for Manhattan, p=2 for Euclidean)

#For Decision Trees:
#- min_samples_split: the minimum number of samples required to split an internal node.
#- min_samples_leaf: the minimum number of samples required to be at a leaf node.
#- max_leaf_nodes: to limit the complexity of the tree.
#- criterion: try 'gini' vs 'entropy'.

#Tuning these hyperparameters using techniques like GridSearchCV or RandomizedSearchCV could further improve the models' performance and generalization.


# In[ ]:




