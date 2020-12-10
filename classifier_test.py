import numpy as np
# from Object_classifier import Classifier
# from combo_classifier import Classifier
# from new_two_level_classifier import *
from Object_classifier_v0411 import *
# from Object_classifier_v0411_v1 import *
import sklearn
import sklearn.linear_model
from scipy.special import expit
import json

print("Classifier test v0.4.1.1")
print("sklearn version: " + sklearn.__version__)

def _predict_proba_lr_hotpatch(self, X):
    """Probability estimation for OvR logistic regression.
    Positive class probabilities are computed as
    1. / (1. + np.exp(-self.decision_function(X)));
    multiclass is handled by normalizing that over all classes.
    """
    prob = self.decision_function(X)
    expit(prob, out=prob)
    if prob.ndim == 1:
        return np.vstack([1 - prob, prob]).T
    else:
        # OvR normalization, like LibLinear's predict_probability
        # we don't want normalized probabilities:
        #prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob

# hotpatch to stop normalization of class probabilities
sklearn.linear_model._base.LinearClassifierMixin._predict_proba_lr = _predict_proba_lr_hotpatch

print("reading data")
f = open("non-novel_200_noisy_groundtruths_100samples_v0411.txt")
data = json.load(f)


# f = open("19_TRUE.json")
# data = [json.load(f)['objects']]  # make a one element list of the input gt dict

# data = []
# for iline in f:
#     line_count += 1
#     if line_count % 100 == 0:
#         l = eval(iline)
#         data.append(l)
#         # gt_dict = {'stuff' : 'things', 'other_stuff' : 'other_things', 'features' : l, 'more_stuff' : 'more_things'}
#         data.append(gt_dict)
# f.close()

# print("len(data): " + str(len(data)))



c = Classifier()


print("classifying")
level_count = 0
for gt in data:
    level_count += 1
    # print("Classifying level: " + str(level_count))
    c.classify(gt)
print("Done")



print("Counting novelty detections")
pig_count = 0
missing_keys = 0
novel_objects = []
novel_object_counts = []
for gt in data:
    novel_object_count = 0
    l3_detection_count = 0
    if gt['l3_detection'] == 1:
        l3_detection_count += 1
    for ob in gt['features']:
        if ob['properties']['novel'] == 1:
            novel_object_count += 1
            novel_objects.append(ob)
            for l in ['shape_novelty', 'novel_shape', 'material_novelty', 'novel_material']:
                if l not in ob['properties'].keys():
                    missing_keys += 1
        if 'pig' in ob['properties']['kdl_label']:
            pig_count += 1
    novel_object_counts.append(novel_object_count)

print("Non-zero novelty gt's: " + str(len([i for i in novel_object_counts if i > 0])))

print("L3 detections: " + str(l3_detection_count))
print("Novel objects detected: " + str(sum(novel_object_counts)))
# print("Pig count: " + str(pig_count))
print("Missing keys: " + str(missing_keys))







# with open("non-novelty_200groundtruths_v033_classified.txt", 'w') as f:
#         for i in data:
#             f.write(str(i) + '\n')

# print("Checking errors:")
# object_count = 0
# fn_count = 0
# fp_count = 0
# false_positive_objects = []
# false_negative_objects = []
# novel_object_count = 0
# non_novel_object_count = 0
# false_positive_deltas = []
# false_negative_deltas = []
# for gt in data:
#     for ob in gt['features']:
#         object_count += 1
#         if 'novel' in ob['properties']['label']: 
#             novel_object_count += 1
#             if ob['properties']['novel'] != 1:
#                 fn_count += 1
#                 false_negative_objects.append(ob['properties']['kdl_label'])
#                 false_negative_deltas.append(np.abs(ob['properties']['max_class_prob'] - c.thresholds[ob['properties']['kdl_label']]) / c.thresholds[ob['properties']['kdl_label']])
#         else:
#             non_novel_object_count += 1
#             if ob['properties']['novel'] != 0:
#                 fp_count += 1
#                 false_positive_objects.append([ob['properties']['label'], ob['properties']['kdl_label']])
#                 ratio = np.abs(ob['properties']['max_class_prob'] - c.thresholds[ob['properties']['kdl_label']]) / c.thresholds[ob['properties']['kdl_label']]
#                 false_positive_deltas.append(ratio)

# print("Fale positive rate: " + str(fp_count / non_novel_object_count))
# print("False negative rate: " + str(fn_count / novel_object_count))



# fpo = []
# for i in false_positive_objects:
#     if i not in fpo:
#         fpo.append(i)

# print()
# print("False Positives:")
# for i in fpo:
#     print(i)
# print()

# fno = []
# for i in false_negative_objects:
#     if i not in fno:
#         fno.append(i)

# print()
# print("False Negatives: ")
# for i in fno:
#     print(i)
# print()



# print("Error percentages")
# fnd = false_negative_deltas
# fpd = false_positive_deltas
# print("max(fnd): " + str(max(fnd)))
# print("min(fnd): " + str(min(fnd)))
# print("meand(fnd): " + str(np.mean(fnd)))

# print()
# print("max(fpd): " + str(max(fpd)))
# print("min(fpd): " + str(min(fpd)))
# print("meand(fpd): " + str(np.mean(fpd)))







