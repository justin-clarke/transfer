# SAIL-ON object classifier training 
# Version 0.4.1.1

import sklearn
# from sklearn import utils
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np
import pickle
import time
import random

print("sklearn version: " + sklearn.__version__)



# abstaining classifer hotpatch function
def _predict_proba_lr_hotpatch(self, X):
    """Probability estimation for OvR logistic regression.
    Positive class probabilities are computed as
    1. / (1. + np.exp(-self.decision_function(X)));
    multiclass DOES NOT NORMALIZE over all classes.
    """
    prob = self.decision_function(X)
    expit(prob, out=prob)
    if prob.ndim == 1:
        return np.vstack([1 - prob, prob]).T
    else:
        # OvR normalization, like LibLinear's predict_probability
        #prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob


# hotpatch to stop normalization of class probabilities
sklearn.linear_model._base.LinearClassifierMixin._predict_proba_lr = _predict_proba_lr_hotpatch

model_save_name = "model_lbfgs_v0411.sav"
print_errors = 1


print("Reading Data")
data = [] 
labels = []
f = open("non-novel_200levels_100samples_v0411.txt")
line_count = 0
for iline in f:
# while line_count <= 5000:
# 	iline = f.readline()
	line_count += 1
	if line_count % 250000 == 0:
		print("Reading line: " + str(line_count))
	l = eval(iline.strip())
	data.append(l[:-1])
	labels.append(l[-1])
f.close()



print("Total data size: " + str(len(data)))

num_test_points = int(len(data)*0.075)
print("Selecting " + str(num_test_points) + " test data points")
test_indices = []
while len(test_indices) < num_test_points:
	if len(test_indices) % 10000 == 0:
		print("Selecting index " + str(len(test_indices)))
	n = random.randint(0, len(data) - 1)
	if n not in test_indices:
		test_indices.append(n)
test_indices.sort()


print("Dividing data and labels into test and train sets")
data = np.array(data)
labels = np.array(labels)
print("- Slicing test data arrays")
test_data = data[test_indices]
test_labels = labels[test_indices]
print("- Selecting train data")

train_data = []
train_labels = []
data_idx = 0
test_idxs_idx = 0
test_index_val = test_indices[test_idxs_idx]
while data_idx < len(data):
	if data_idx % 250000 == 0:
		print("--  processing data row: " + str(data_idx))
	if data_idx not in test_indices:
		train_data.append(data[data_idx])
		train_labels.append(labels[data_idx])
		data_idx += 1
	else:
		data_idx += 1



	# if  data_idx != test_index_val:
	# 	train_data.append(data[data_idx])
	# 	train_labels.append(labels[data_idx])
	# 	data_idx += 1
	# else:
	# 	data_idx += 1
	# 	if test_idxs_idx == len(test_indices) - 1:
	# 		train_data.extend(data[data_idx:])
	# 		train_labels.extend(labels[data_idx:])
	# 		data_idx = len(data)
	# 	else:
	# 		test_idxs_idx += 1
	# 		test_index_val = test_indices[test_idxs_idx]

print("Train data size: " + str(len(train_data)))
print("Train label size: " + str(len(train_labels)))
print("Test data size: " + str(len(test_data)))
print("Test label size: " + str(len(test_labels)))


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



