#!/usr/bin/env python
# coding: utf-8

# # Lab 8: Define and Solve an ML Problem of Your Choosing

# In[20]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# In this lab assignment, you will follow the machine learning life cycle and implement a model to solve a machine learning problem of your choosing. You will select a data set and choose a predictive problem that the data set supports.  You will then inspect the data with your problem in mind and begin to formulate a  project plan. You will then implement the machine learning project plan. 
# 
# You will complete the following tasks:
# 
# 1. Build Your DataFrame
# 2. Define Your ML Problem
# 3. Perform exploratory data analysis to understand your data.
# 4. Define Your Project Plan
# 5. Implement Your Project Plan:
#     * Prepare your data for your model.
#     * Fit your model to the training data and evaluate your model.
#     * Improve your model's performance.

# ## Part 1: Build Your DataFrame
# 
# You will have the option to choose one of four data sets that you have worked with in this program:
# 
# * The "census" data set that contains Census information from 1994: `censusData.csv`
# * Airbnb NYC "listings" data set: `airbnbListingsData.csv`
# * World Happiness Report (WHR) data set: `WHR2018Chapter2OnlineData.csv`
# * Book Review data set: `bookReviewsData.csv`
# 
# Note that these are variations of the data sets that you have worked with in this program. For example, some do not include some of the preprocessing necessary for specific models. 
# 
# #### Load a Data Set and Save it as a Pandas DataFrame
# 
# The code cell below contains filenames (path + filename) for each of the four data sets available to you.
# 
# <b>Task:</b> In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`. 
# 
# You can load each file as a new DataFrame to inspect the data before choosing your data set.

# In[21]:


# File names of the four data sets
adultDataSet_filename = os.path.join(os.getcwd(), "data", "censusData.csv")
airbnbDataSet_filename = os.path.join(os.getcwd(), "data", "airbnbListingsData.csv")
WHRDataSet_filename = os.path.join(os.getcwd(), "data", "WHR2018Chapter2OnlineData.csv")
bookReviewDataSet_filename = os.path.join(os.getcwd(), "data", "bookReviewsData.csv")

df = pd.read_csv(bookReviewDataSet_filename)
df.head()


# ## Part 2: Define Your ML Problem
# 
# Next you will formulate your ML Problem. In the markdown cell below, answer the following questions:
# 
# 1. List the data set you have chosen.
# 2. What will you be predicting? What is the label?
# 3. Is this a supervised or unsupervised learning problem? Is this a clustering, classification or regression problem? Is it a binary classificaiton or multi-class classifiction problem?
# 4. What are your features? (note: this list may change after your explore your data)
# 5. Explain why this is an important problem. In other words, how would a company create value with a model that predicts this label?

# <Double click this Markdown cell to make it editable, and record your answers here.>
# 1. I selected the Book Review dataset (bookReviewsData.csv), which includes written book reviews along with a label indicating whether each review is considered positive (True) or not (False). Each record contains two columns:
#     - Review: a free-text string representing a user’s written opinion
#     - Positive Review: a boolean value representing sentiment (our label)
#     This dataset is ideal for natural language processing (NLP) tasks involving text classification.
# 2. The target variable is Positive Review, a boolean value:
#     - True indicates that the review is positive
#     - False indicates that the review is negative or neutral
#     I aim to build a model that can predict this sentiment automatically, based solely on the content of the Review field.
# 
# 3. This is a supervised machine learning problem because the training data contains labeled outcomes.
#    It is a classification problem, and more specifically a binary classification problem, because there are only two possible output classes:
#     - Class 1: Positive review (True)
#     - Class 0: Not positive review (False)
#     I will train the model using historical labeled examples to learn how to classify unseen reviews.
# 
# 4. The primary feature is the free-text review in the Review column.
#     Since machine learning models cannot directly process raw text, I will apply NLP preprocessing steps such as:
#     - Lowercasing and punctuation removal
#     - Tokenization
#     - Removing stopwords (e.g., "and", "the", "is")
#     - Vectorization using TF-IDF or word embeddings like Word2Vec or BERT, 
#     I may also engineer the following features:
#     - Review length (word or character count)
#     - Sentiment score using a lexicon-based sentiment analyzer (e.g., TextBlob, VADER)
#     - Presence of positive/negative keywords (e.g., “excellent”, “boring”)
#     These features will help convert unstructured text into structured numerical input suitable for ML models.
# 
# 5. Automatically detecting sentiment in book reviews is valuable for multiple reasons. For online platforms such as Amazon or Goodreads, a sentiment classifier can help personalize recommendations, summarize public opinion, and moderate user-generated content more efficiently. Authors and publishers can use the insights to track reader satisfaction, spot trends in feedback, and evaluate the impact of marketing campaigns. From a user perspective, sentiment-based filtering can improve the browsing experience by surfacing reviews that align with their preferences. Beyond the book industry, this kind of sentiment analysis model is widely transferable to other domains such as product reviews, movie ratings, or customer support making it a powerful tool for improving user engagement, streamlining decision-making, and unlocking business intelligence across different industries.

# ## Part 3: Understand Your Data
# 
# The next step is to perform exploratory data analysis. Inspect and analyze your data set with your machine learning problem in mind. Consider the following as you inspect your data:
# 
# 1. What data preparation techniques would you like to use? These data preparation techniques may include:
# 
#     * addressing missingness, such as replacing missing values with means
#     * finding and replacing outliers
#     * renaming features and labels
#     * finding and replacing outliers
#     * performing feature engineering techniques such as one-hot encoding on categorical features
#     * selecting appropriate features and removing irrelevant features
#     * performing specific data cleaning and preprocessing techniques for an NLP problem
#     * addressing class imbalance in your data sample to promote fair AI
#     
# 
# 2. What machine learning model (or models) you would like to use that is suitable for your predictive problem and data?
#     * Are there other data preparation techniques that you will need to apply to build a balanced modeling data set for your problem and model? For example, will you need to scale your data?
#  
#  
# 3. How will you evaluate and improve the model's performance?
#     * Are there specific evaluation metrics and methods that are appropriate for your model?
#     
# 
# Think of the different techniques you have used to inspect and analyze your data in this course. These include using Pandas to apply data filters, using the Pandas `describe()` method to get insight into key statistics for each column, using the Pandas `dtypes` property to inspect the data type of each column, and using Matplotlib and Seaborn to detect outliers and visualize relationships between features and labels. If you are working on a classification problem, use techniques you have learned to determine if there is class imbalance.
# 
# <b>Task</b>: Use the techniques you have learned in this course to inspect and analyze your data. You can import additional packages that you have used in this course that you will need to perform this task.
# 
# <b>Note</b>: You can add code cells if needed by going to the <b>Insert</b> menu and clicking on <b>Insert Cell Below</b> in the drop-drown menu.

# In[22]:


# YOUR CODE HERE
# Structure & Missingness
print("DataFrame info:")
df.info()
print("\n- Any missing values per column?")
print(df.isna().sum())

# Class balance
counts = df["Positive Review"].value_counts()
print("\n- Class balance:")
print(counts)
print("\n- Class balance (percent):")
print(counts / counts.sum() * 100)

plt.figure(figsize=(4,3))
sns.barplot(x=counts.index.astype(str), y=counts.values)
plt.xticks([0,1], ["Negative (False)", "Positive (True)"])
plt.ylabel("Number of reviews")
plt.title("Positive vs. Negative Reviews")
plt.show()

# Review length distribution
# Compute number of words per review
df["word_count"] = df["Review"].str.split().apply(len)

plt.figure(figsize=(6,4))
sns.histplot(df["word_count"], bins=30, kde=True)
plt.xlabel("Words per review")
plt.title("Distribution of Review Lengths")
plt.show()

# Peek at a few examples
print("\n- A few NEGATIVE reviews (Positive Review == False):")
display(df[df["Positive Review"] == False].head(3)["Review"])

print("\n- A few POSITIVE reviews (Positive Review == True):")
display(df[df["Positive Review"] == True].head(3)["Review"])


# ## Part 4: Define Your Project Plan
# 
# Now that you understand your data, in the markdown cell below, define your plan to implement the remaining phases of the machine learning life cycle (data preparation, modeling, evaluation) to solve your ML problem. Answer the following questions:
# 
# * Do you have a new feature list? If so, what are the features that you chose to keep and remove after inspecting the data? 
# * Explain different data preparation techniques that you will use to prepare your data for modeling.
# * What is your model (or models)?
# * Describe your plan to train your model, analyze its performance and then improve the model. That is, describe your model building, validation and selection plan to produce a model that generalizes well to new data. 

# <Double click this Markdown cell to make it editable, and record your answers here.>
# 1. Feature Selection:
#     After performing EDA, I will keep the following features:
#     - Review (text) – primary input to the model
#     - word_count – engineered numerical feature representing review length
#     No features need to be removed, as the dataset is clean and focused. However, additional derived features (e.g., sentiment polarity) may be added later to improve model performance.
# 
# 2. To prepare our data for modeling, I will apply several preprocessing techniques. First, the text in the Review column will be cleaned and normalized. This involves converting all characters to lowercase, removing punctuation and numeric characters, eliminating stopwords, and optionally applying lemmatization to reduce words to their base forms. Once the text is cleaned, I will transform it into numerical input using Term Frequency Inverse Document Frequency (TF-IDF) vectorization. This step converts the review text into a matrix of token importance scores while controlling for high-frequency common words. I may also experiment with including unigrams and bigrams in our TF-IDF vectorization, and limit the number of features to avoid sparsity. The word_count column, being numerical, will be scaled using either MinMaxScaler or StandardScaler to ensure consistent feature ranges across models that are sensitive to scale.
# 
# 3. For model selection, I will begin with a simple and interpretable baseline: logistic regression using the TF-IDF-transformed review text. This model is well-suited for sparse, high-dimensional data like text. I will also evaluate a multinomial Naive Bayes classifier, which is a strong baseline for text classification and often performs well in practice. If needed, I may explore more complex models such as random forest or gradient boosting classifiers, especially if I incorporate additional non-text features like word_count. Finally, depending on model performance and time constraints, I may explore more advanced techniques such as Support Vector Machines or deep learning models like LSTM or BERT embeddings.
# 
# 4. The training and evaluation plan will follow a structured machine learning life cycle. First, I will split the dataset into an 80% training set and a 20% test set using stratified sampling to preserve the label distribution. During training, I  will apply 5-fold cross-validation on the training set to tune hyperparameters and avoid overfitting. Once models are trained, I will evaluate them on the test set using appropriate metrics for classification tasks. These metrics include accuracy, precision, recall, F1-score, and a confusion matrix. If needed, I will also compute ROC-AUC scores to understand the trade-off between sensitivity and specificity.
#     To improve model performance, I will perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV. I may test variations of the TF-IDF vectorizer (e.g., different n-gram ranges or feature limits), experiment with including or excluding features like word_count, and test dimensionality reduction methods such as Truncated SVD for compressing the TF-IDF space. Throughout this process, I will track performance on the validation folds and test set to ensure our model generalizes well. Once I identify the best-performing model, I will retrain it on the entire training set and evaluate its performance on the test set to confirm final results.
# 

# ## Part 5: Implement Your Project Plan
# 
# <b>Task:</b> In the code cell below, import additional packages that you have used in this course that you will need to implement your project plan.

# In[23]:


# YOUR CODE HERE
# Text preprocessing and modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# <b>Task:</b> Use the rest of this notebook to carry out your project plan. 
# 
# You will:
# 
# 1. Prepare your data for your model.
# 2. Fit your model to the training data and evaluate your model.
# 3. Improve your model's performance by performing model selection and/or feature selection techniques to find best model for your problem.
# 
# Add code cells below and populate the notebook with commentary, code, analyses, results, and figures as you see fit. 

# In[24]:


# YOUR CODE HERE
#Data Preparation
X = df["Review"] # text feature
y = df["Positive Review"].astype(int) # boolean → 0/1 label

# stratified split so classes remain balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# TF–IDF vectorizer: unigrams + bigrams, drop English stop-words, limit to top 5000 features
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1,2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Baseline Modeling & Evaluation
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MultinomialNB": MultinomialNB(),
}

for name, model in models.items():
    # train
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    
    # metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"\n {name}:")
    print(f"Accuracy: {acc:.3f}   F1-score: {f1:.3f}\n")
    print(classification_report(y_test, preds, target_names=["Negative","Positive"]))
    
    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative","Positive"])
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax) 
    ax.set_title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.show()


# Hyperparameter Tuning (Logistic Regression)
param_grid = {
    "C":       [0.01, 0.1, 1, 10],
    "penalty": ["l1","l2"],
    "solver":  ["liblinear","lbfgs"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)

best = grid.best_estimator_
print("\nLogisticRegression Grid Search Results: ")
print("Best params:", grid.best_params_)
print(f"Training  F1: {grid.best_score_:.3f}")

# evaluate tuned on test set
test_preds = best.predict(X_test_tfidf)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test F1-score: {test_f1:.3f}\n")

# tuned confusion matrix
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative","Positive"])
fig, ax = plt.subplots(figsize=(4,4))
disp.plot(ax=ax)
ax.set_title("Tuned LogisticRegression Confusion Matrix")
plt.tight_layout()
plt.show()


# In[ ]:




