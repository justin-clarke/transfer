# SAIL-ON object classifier training 
# trains a secondary classifier to identify the shape of novel objects

# from KDL_sklearn.sklearn.linear_model import LogisticRegression
import sklearn
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np
import pickle
import time

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

model_save_name = "model_lbfgs_v041_aws_trained_shape.sav"
print_errors = 0

print("Reading Data")
data = []
labels = []
f = open("shape_data.txt")
line_counter = 0
for iline in f:
	line_counter += 1
	if line_counter % 200000 == 0:
		print("reading line: " + str(line_counter))
# for i in range(500):
# 	iline = f.readline()
	l = iline.strip()
	l = eval(l)
	data.append(l[:-1])
	labels.append(l[-1])
f.close()

print("len(data): " + str(len(data)))

test_case_count = int(len(data) * 0.05)

train_data = data[:-test_case_count]
train_labels = labels[:-test_case_count]
test_data = data[-test_case_count:]
test_labels = labels[-test_case_count:]

print("train data size: " + str(len(train_data)))
print("test data size: " + str(len(test_data)))

train_data = np.array(train_data)
test_data = np.array(test_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


start = time.perf_counter()
print("Training")
model = LogisticRegression(random_state=8, solver='lbfgs', max_iter=10000, multi_class='ovr').fit(train_data, train_labels) 
end = time.perf_counter()
total = end - start

print("Predicting")
preds = model.predict(test_data)

correct_count = 0
for i in range(len(preds)):
	if preds[i] == test_labels[i]:
		correct_count += 1
	else:
		if print_errors:
			print("predicted: " + preds[i] + ", actual: " + test_labels[i])

accuracy = correct_count / len(test_labels)
print("Training time: " + str(total))
print("Accuracy: " + str(accuracy))

# save model
pickle.dump(model, open(model_save_name, 'wb'))

probs=model.predict_proba(data)
p = [max(i) for i in probs]
t = min(p)
print("Novelty Threshold: " + str(t))

