
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pickle
plt.style.use('ggplot')

df = pd.read_csv("Reviews.csv")

df.shape

df.rename(columns={"Text":"Review"},inplace=True)

df["Score"].value_counts()

rating_denominations = df['Score'].unique()

dataset = []
for denomination in rating_denominations:
    filtered_data = df[df['Score'] == denomination]
    data = filtered_data[:2500]
    dataset.extend(data.values.tolist())
        
dataset = pd.DataFrame(dataset,columns = df.columns)

dataset = dataset[dataset["Score"]!=3]

dataset["Score"].value_counts()

dataset["Rating"] = np.where(dataset["Score"] > 3, 1, 0)

dataset.isnull().sum()

dataset['Rating'].value_counts()

example = dataset['Review'][22]
example

import re
cleaned_text = re.sub('[^A-Za-z ]+', '', example)
print(cleaned_text)

from nltk import word_tokenize
tokenized = word_tokenize(cleaned_text)
print(tokenized)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)

tokens =[]
for word in tokenized:
    if word not in stop_words:
        tokens.append(word)
print(tokens)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized_tokens)

final_text = " ".join(lemmatized_tokens)
print(final_text)

# for i in range(0,len(dataset.index)):
#     raw_text = dataset.Review[i]
#     cleaned_text = re.sub('[^A-Za-z ]+', '', raw_text)
#     tokenized = word_tokenize(cleaned_text)
#     tokens =[]
#     for word in tokenized:
#         if word not in stop_words:
#             tokens.append(word)
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     final_text = " ".join(lemmatized_tokens)
#     print(final_text)
#     dataset['Review'] = dataset['Review'].replace(dataset.Review[i],final_text)
    

# import spacy
# nlp = spacy.load("en_core_web_sm")

# def preprocess(text):
#     doc = nlp(text)
#     filtered_tokens = []
#     for token in doc:
#         if token.is_stop or token.is_punct:
#             continue
#         filtered_tokens.append(token.lemma_)
        
#     return " ".join(filtered_tokens)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(text):
    # Tokenize the input text
    words = word_tokenize(text)
    
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Lemmatization (you can use NLTK's WordNetLemmatizer if needed)
    # Note: NLTK's lemmatizer requires POS tags for accurate lemmatization
    # You may want to explore other lemmatization options depending on your needs.
    
    return " ".join(filtered_tokens)


dataset["Review"] = dataset["Review"].apply(preprocess)

dataset["Review"][3]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset['Review'],dataset['Rating'],test_size=0.2)

y_train.value_counts()

y_test.value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer().fit(x_train)

# print("vectorizer dumping started.")
# with open("vectorizer.pkl", 'wb') as fileobj2:
#     pickle.dump(vect, fileobj2)
# print("vectorizer dumped successfully.")

len(vect.get_feature_names_out())

x_train_vectorized = vect.transform(x_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear",multi_class="ovr")
lr.fit(x_train_vectorized,y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train_vectorized, y_train)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train_vectorized.toarray(), y_train)

from sklearn.svm import SVC
sv = SVC()
sv.fit(x_train_vectorized, y_train)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train_vectorized, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion = "entropy")
rf.fit(x_train_vectorized, y_train)

x_test_vectorized = vect.transform(x_test)

lr_pred = lr.predict(x_test_vectorized)
knn_pred = knn.predict(x_test_vectorized)
nb_pred = nb.predict(x_test_vectorized.toarray())
sv_pred = sv.predict(x_test_vectorized)
dt_pred = dt.predict(x_test_vectorized)
rf_pred = rf.predict(x_test_vectorized)

from sklearn.metrics import accuracy_score

print("Train Accuracy of Logistic Regression", lr.score(x_train_vectorized, y_train)*100)
print("Accuracy score of Logistic Regression",accuracy_score(y_test, lr_pred)*100)

print("Train Accuracy of KNN", knn.score(x_train_vectorized, y_train)*100)
print("Accuracy score of KNN",accuracy_score(y_test, knn_pred)*100)

print("Train Accuracy of Naive Bayes", nb.score(x_train_vectorized.toarray(), y_train)*100)
print("Accuracy score of Naive Bayes",accuracy_score(y_test, nb_pred)*100)

print("Train Accuracy of SVM", sv.score(x_train_vectorized, y_train)*100)
print("Accuracy score of SVM",accuracy_score(y_test, sv_pred)*100)

print("Train Accuracy of Decision Tree", dt.score(x_train_vectorized, y_train)*100)
print("Accuracy score of Decision Tree",accuracy_score(y_test, dt_pred)*100)

print("Train Accuracy of Random Forest", rf.score(x_train_vectorized, y_train)*100)
print("Accuracy score of Random Forest",accuracy_score(y_test, rf_pred)*100)


predictions = lr.predict(vect.transform(x_test))

from sklearn.metrics import roc_auc_score
print("AUC", roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names_out())
sorted_coef_index = lr.coef_[0].argsort()

print("smallest coefficient",  feature_names[sorted_coef_index[:10]])
print("Largest coefficient",  feature_names[sorted_coef_index[-11:-1]])

input_data = ("The food was tasty, and the service was best")
data = [input_data]
std_data = vect.transform(data)
prediction = sv.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('Negative Rating')
else:
  print('Positive Rating')


input_data = ("hotel service is rediculous")
data = [input_data]
std_data = vect.transform(data)
prediction = sv.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('Negative Rating')
else:
  print('Positive Rating')

# print("model dumping started")
# with open("trained_model.pkl", 'wb') as fileobj:
#   pickle.dump(sv, fileobj)
# print("model sucessfully dumped")



# def analyze_sentiment(data):
#     with open("vectorizer.pkl", 'rb') as fileobj2:
#       vect = pickle.load(fileobj2)
#     std_data = vect.transform(data)
#     file1 = "trained_model.pkl"
#     fileobj1 = open(file1, 'rb')
#     model = pickle.load(fileobj1)
#     # path = os.path.join('artifacts','trained_model.pkl')
#     # model = pickle.loads(path)

#     # Make sure the number of features match
#     if std_data.shape[1] != model.support_vectors_.shape[1]:
#       raise ValueError(f"Number of features in input data ({std_data.shape[1]}) doesn't match the trained model ({model.support_vectors_.shape[1]})")
    
#     prediction=model.predict(std_data)

#     if (prediction[0] == 0):
#       return 'Negative Rating'
#     else:
#       return 'Positive Rating'
    