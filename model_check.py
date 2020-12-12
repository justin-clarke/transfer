import sklearn
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

print("sklearn version: " + sklearn.__version__)


# abstaining classifer hotpatch function
def _predict_proba_lr_hotpatch(self, X):
	"""
	Probability estimation for OvR logistic regression.
	Positive class probabilities are computed as
	1. / (1. + np.exp(-self.decision_function(X)));
	multiclass DOES NOT NORMALIZE over all classes.
	"""
	prob = self.decision_function(X)
	expit(prob, out=prob)
	if prob.ndim == 1:
		return np.vstack([1 - prob, prob]).T
	else:
		# print("Not normalizing")
		# OvR normalization, like LibLinear's predict_probability
		#prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
		return prob


# hotpatch to stop normalization of class probabilities
sklearn.linear_model._base.LinearClassifierMixin._predict_proba_lr = _predict_proba_lr_hotpatch


print("Reading data")
data = []
labels = []
f = open("shape_data.txt")
line_count = 0
for iline in f:
	line_count += 1
	if line_count % 200000 == 0:
		print("Reading Line: " + str(line_count))
	l = iline.strip()
	l = eval(l)
	data.append(l[:-1])
	labels.append(l[-1])
f.close()

print("len(data): " + str(len(data)))
# data = data[-200000:]
# labels = labels[-200000:]

model_name = 'model_lbfgs_v041_aws_trained_shape.sav'

print("Loading model")
model = pickle.load(open(model_name, 'rb'))
print("Predicting")
predictions = model.predict(data)


# or ('pig' in labels [i] and 'pig' in predictions[i])
errors = {}
correct = 0
for i in range(len(labels)):
	if labels[i] == predictions[i]:
		correct += 1
	else:
		if str([labels[i], predictions[i]]) in errors.keys():
			errors[str([labels[i], predictions[i]])] += 1
		else:
			errors[str([labels[i], predictions[i]])] = 1


accuracy = correct / len(labels)
print("Accuracy: " + str(accuracy))


print()
print("Errors")
for i in errors:
	print(i + ": " + str(errors[i]))


probs = model.predict_proba(data)
p = [max(i) for i in probs]
print("Minimum max-class probability: " + str(min(p)))


print("Finding Thresholds")
objects = np.unique(labels)
thresholds = {}
max_probs = {}
for ob in objects:
	max_probs[ob] = []
	idx = [i for i in range(len(labels)) if labels[i] == ob]
	min_prob = 1000000000
	for i in idx:
		max_probs[ob].append(p[i])
		if p[i] < min_prob:
			min_prob = p[i]
	thresholds[ob] = min_prob


for i in thresholds:
	print("'" + i + "' : " + str(thresholds[i]) + ",")


# d = max_probs['Platform']
# d = np.sort(d)
# x = [i for i in range(len(d))]
# plt.plot(x, d, marker='o')
# plt.title("sorted probability values")
# plt.show()
# plt.close()


# for ob in max_probs:
# 	d = np.sort(max_probs[ob])
# 	y = np.arange(len(d))/float(len(d))
# 	plt.plot(d, y)
# 	plt.title(ob + " CDF")
# 	plt.savefig("CDF_" + ob)
# 	# plt.show()

# 	plt.close()






