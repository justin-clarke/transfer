# converts ground truths to object data for classifier
import json
from gt_update import *
from get_data import *
f = open("non-novel_noisy_groundtruths_0-80_100samples_v041_aws.txt")
gt_list = json.load(f)
data = []
for gt in gt_list:
	gt_update(gt)
	d = get_data(gt)
	data.extend(d)


with open("non-novel_0-80_100samples_v041_aws.txt", 'w') as f:
        for i in data:
        	f.write(str(i) + '\n')


