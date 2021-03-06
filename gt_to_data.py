# converts ground truths to object data for classifier
import json
from gt_update import *
from get_data import *

print("Reading data")
f = open("non-novel_noisy_groundtruths_0-80_100samples_v041_aws.txt")
gt_list = json.load(f)
data = []
gt_count = 0
for gt in gt_list:
	gt_count += 1
	if gt_count % 100:
		print("Processing gt: " + str(gt_count))
	gt_update(gt)
	d = get_data(gt)
	data.extend(d)

print("Writing data")
with open("non-novel_0-80_100samples_v041_aws.txt", 'w') as f:
        for i in data:
        	f.write(str(i) + '\n')


