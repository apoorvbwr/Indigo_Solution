# -*- coding: utf-8 -*-
"""Indigo-Visualization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fRtk6s8PAA5AO18Syisnf_PvVx2z9OMu
"""

pip install transformers[torch] tokenizers datasets evaluate rouge_score sentencepiece huggingface_hub --upgrade

import pandas as pd
from datasets import load_dataset

dataset = load_dataset('toughdata/quora-question-answer-dataset', split='train')
df = pd.DataFrame(dataset)

print(df.head())

print(df.info())

print(df.describe())

import matplotlib.pyplot as plt

df['question_length'] = df['question'].apply(len)

plt.hist(df['question_length'], bins=50, alpha=0.7, color='blue')
plt.title('Distribution of Question Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

df['answer_length'] = df['answer'].apply(len)

plt.hist(df['answer_length'], bins=50, alpha=0.7, color='green')
plt.title('Distribution of Answer Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()



from wordcloud import WordCloud

text = ' '.join(df['question'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

text = ' '.join(df['answer'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Assuming binary labels for simplicity
df['label'] = [1 if i % 2 == 0 else 0 for i in range(len(df))]  # Example labels

X_train, X_test, y_train, y_test = train_test_split(df['question'], df['label'], test_size=0.3, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

pipeline.fit(X_train, y_train)
importances = pipeline.named_steps['logisticregression'].coef_[0]

# Display the most important features
feature_names = pipeline.named_steps['tfidfvectorizer'].get_feature_names_out()
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

print(importance_df.head(10))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Assuming binary labels for simplicity
df['label'] = [1 if i % 2 == 0 else 0 for i in range(len(df))]  # Example labels

X_train, X_test, y_train, y_test = train_test_split(df['answer'], df['label'], test_size=0.3, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

pipeline.fit(X_train, y_train)
importances = pipeline.named_steps['logisticregression'].coef_[0]

# Display the most important features
feature_names = pipeline.named_steps['tfidfvectorizer'].get_feature_names_out()
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

print(importance_df.head(10))

import seaborn as sns

# Correlation matrix for length features
correlation_matrix = df[['question_length', 'answer_length']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Question and Answer Length')
plt.show()

print(df['question_length'].describe())
print(df['answer_length'].describe())

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['question'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

for index, topic in enumerate(lda.components_):
    print(f'Topic #{index}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['answer'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

for index, topic in enumerate(lda.components_):
    print(f'Topic #{index}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data
X = df[['question_length']]
y = df['answer_length']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate feature importance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature importance
importances = model.feature_importances_
print(f'Feature Importance: {importances}')

# Scatter plot of question length vs answer length
plt.figure(figsize=(12, 6))
sns.scatterplot(x='question_length', y='answer_length', data=df)
plt.title('Question Length vs Answer Length')
plt.xlabel('Question Length')
plt.ylabel('Answer Length')
plt.show()

# Correlation between question length and answer length
correlation = df[['question_length', 'answer_length']].corr()
print(f'Correlation matrix:\n{correlation}')

