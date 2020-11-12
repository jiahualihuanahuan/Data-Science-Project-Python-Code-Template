# Data Class
import random
class Sentiment:
	NEGATIVE = 'NEGATIVE'
	NEUTRAL = 'NEUTRAL'
	POSITIVE = 'POSITIVE'

class Review:
	def __init__(self, text, score):
		self.text = text
		self.score = score
		self.sentiment = self.get_sentiment()

	def get_sentiment(self):
		if self.score <=2:
			return Sentiment.NEGATIVE
		elif self.score == 3:
			return Sentiment.NEUTRAL
		else:
			return Sentiment.POSITIVE

class ReviewContainer:
	def __init__(self, reviews):
		self.reviews = reviews

	def get_text(self):
		return [x.text for x in self.reviews]

	def get_sentiment(self):
		return [x.sentiment for x in self.reviews]

	def evenly_distribute(self):
		negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
		positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
		positive_shrunk = positive[:len(negative)]
		self.reviews = negative + positive_shrunk
		random.shuffle(self.reviews)

# Load Data
import json

file_name = 'E:\\Dropbox\\Machine Learning\\Data\\Amazon Review Data\\Software.json\\Software.json'

reviews = []

with open(file_name) as f:
	for line in f:
		try:
			review = json.loads(line)
			reviews.append(Review(review["reviewText"], review["overall"]))
		except KeyError:
			continue



# Prep Data
from sklearn.model_selection import train_test_split

train, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(train)

test_container = ReviewContainer(test)

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(f'number of positive reviews {train_y.count(Sentiment.POSITIVE)}')
print(f'number of negative reviews {train_y.count(Sentiment.NEGATIVE)}')

# Bag of words vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

# Classification
# Linear SVM

from sklearn import svm

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()
clf_dt.fit(train_x_vectors, train_y)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_jobs=-1)
clf_rf.fit(train_x_vectors, train_y)

# Naive Bayes xxx
#from sklearn.naive_bayes import GaussianNB

#clf_gnb = GaussianNB()
#clf_gnb.fit(train_x_vectors.toarray(), train_y)

# logistic regression
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

# neural network
from sklearn.neural_network import MLPClassifier

clf_nn = MLPClassifier(hidden_layer_sizes=(500,400,300,200,100,), random_state=1, max_iter=300)
clf_nn.fit(train_x_vectors, train_y)


# Mean Accuracy
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dt.score(test_x_vectors, test_y))
#print(clf_gnb.score(test_x_vectors, test_y))
print(clf_log.score(test_x_vectors, test_y))

# F1 Scores
from sklearn.metrics import f1_score

f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
f1_score(test_y, clf_dt.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
f1_score(test_y, clf_gnb.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])

# save model
import pickle

with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf_svm, f)


# load model
with open('./models/entiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

