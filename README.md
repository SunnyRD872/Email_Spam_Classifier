
# Email Spam Classifier



## Overview

This project is an Email Spam Classifier built using Machine Learning and Natural Language Processing (NLP) techniques. It classifies emails as either spam (1) or ham (0). The project includes data cleaning, exploratory data analysis (EDA), text preprocessing, model training, evaluation, and deployment using Streamlit.
## Dataset

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
## Features

1.Data Cleaning:

• Handles null values.

• Removes duplicates.

• Drops unnecessary columns.

2.Text Preprocessing:

• Converts text to lowercase.

• Removes stopwords.

• Performs stemming and tokenization using NLTK.

3.Exploratory Data Analysis (EDA):

• Visualizes the distribution of spam and ham messages using pie charts.

• Analyzes the relationship between the number of characters, words, and sentences using pairplots.

• Creates histograms to show the distribution of sentence and word lengths.

• Generates word clouds to display the most frequent words in spam and ham messages.

4.Word Embedding:

• Utilizes CountVectorizer and TF-IDF Vectorizer for text feature extraction.

5.Model Training:

• Implements and evaluates three Naive Bayes models:

-GaussianNB

-BernoulliNB

-MultinomialNB

6.Model Evaluation:

• Compares models using accuracy_score, confusion_matrix, and precision_score.

7.Deployment:

• Saves the best-performing model and vectorizer using pickle.

• Deploys the application using Streamlit.
## Technologies Used

1.Programming Language: Python

2.Libraries:

• Data Handling: pandas, numpy

• Visualization: matplotlib, seaborn, wordcloud

• NLP: nltk (word tokenization, stopwords, stemming)

• Machine Learning: scikit-learn (CountVectorizer, TF-IDF, Naive Bayes models)

• Model Saving: pickle

• Deployment: streamlit

## Project Workflow

1.Data Loading and Cleaning:

• Load the dataset.

• Check for null values, duplicates, and unnecessary columns.

• Encode the target variable: spam as 1 and ham as 0.

2.Exploratory Data Analysis (EDA):

• Visualize the distribution of spam and ham messages using pie charts.

• Analyze the relationship between the number of characters, words, and sentences using pairplots.

• Create histograms to show the distribution of sentence and word lengths.

• Generate word clouds for spam and ham messages.

3.Text Preprocessing:

• Convert text to lowercase.

• Remove stopwords and perform stemming.

• Tokenize words and sentences using NLTK.

4.Feature Extraction:

• Use CountVectorizer and TF-IDF Vectorizer for word embedding.

5.Model Training and Evaluation:

• Split the dataset into training and testing sets.

• Train and evaluate three Naive Bayes models:

-GaussianNB

-BernoulliNB

-MultinomialNB

• Evaluate models using accuracy_score, confusion_matrix, and precision_score.

6.Model Selection:

• MultinomialNB with TF-IDF Vectorizer performs the best.

7.Saving the Model:

• Save the best model and TF-IDF vectorizer using pickle.

8.Deployment:

• Deploy the model using Streamlit for real-time spam classification.


## Results

Best Model:

- MultinomialNB with TF-IDF Vectorizer.

Evaluation Metrics:

• Accuracy: 0.9593810444874274

• Precision: 1.0

• Confusion Matrix: [[896   0]

[ 42  96]]


## Future Improvements

1.Experiment with deep learning models.

2.Enhance feature engineering for better accuracy.
## Special Thanks

This project was inspired by the tutorial from the CampusX Channel on YouTube.
Channel:https://www.youtube.com/@campusx-official
