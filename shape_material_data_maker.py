# creates shape and material based datasets from training data

import numpy as np


data = []
labels = []
materials = []
shapes = []
material_data = []
shape_data = []
f = open("non-novel_200levels_100samples_v0411.txt")
line_count = 0
for iline in f:
	line_count += 1
	if line_count % 200000 == 0:
		print("reading line: " + str(line_count))
# for i in range(1000):
# 	iline = f.readline()
	l = iline.strip()
	l = eval(l)
	label = l[-1]
	data = l[:-1]
	if label not in ['TNT', 'Platform']:
		mat = ''
		ch = 0
		while label[ch] != '_':
			mat = mat + label[ch]
			ch += 1
		material_data.append(data[24:] + [mat])
		if mat in ['pig', 'bird']:
			shape_data.append(data[:24] + [label])
		else:
			shape_data.append(data[:24] + [label[ch+1:]])
	else:
		material_data.append(data[24:] + [label])
		shape_data.append(data[:24] + [label])

f.close()


# print("Data size: " + str(len(data)))

print("writing data")

with open("shape_data.txt", 'w') as f:
		for i in shape_data:
			f.write(str(i) + '\n')

with open("material_data.txt", 'w') as f:
		for i in material_data:
			f.write(str(i) + '\n')






