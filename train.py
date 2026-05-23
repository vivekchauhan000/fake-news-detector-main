import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

# load data
df = pd.read_csv('../data/fake_or_real_news.csv')

# preprocess
df['fake'] = (df['label'] == 'REAL').astype(int)

X, y = df['text'], df['fake']

# split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# train
model = LinearSVC()
model.fit(x_train_vec, y_train)

# accuracy
print("Accuracy:", model.score(x_test_vec, y_test))

# save model
pickle.dump(model, open('../models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('../models/vectorizer.pkl', 'wb'))
