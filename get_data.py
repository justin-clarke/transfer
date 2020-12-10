# converts ground truth data into feature vectors for the classifier 
# takes a ground truth dictionary updated with classifier features from gt_update as input

def strip_labels(gt_dict):
	gt = gt_dict['features']
	for i in range(len(gt)):
		if gt[i]['properties']['label'] not in ['Platform', 'TNT', 'Ground', 'Trajectory', 'Slingshot']:
			gt[i]['properties']['label'] = gt[i]['properties']['label'][:-2]
	return



def get_data(gt):
	data = []

	strip_labels(gt)  # remove damage indicator from labels

	for i in range(len(gt['features'])):
		feature = gt['features'][i]
		if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
			# create record and append to training_data
			unit = []
			# add numeric variables
			unit.append(feature['properties']['area'])
			unit.append(feature['properties']['shape_ratio'])
			unit.append(feature['properties']['shortest_edge'])
			unit.append(feature['properties']['longest_edge'])
			unit.append(feature['properties']['mean_edge'])
			unit.append(feature['properties']['median_edge'])
			unit.append(feature['properties']['edge_sum'])
			unit.append(feature['properties']['contour_count'])
			
			# add categorical variables
			vc = feature['properties']['vertex_count']
			for c in range(4, 20):
				if c == vc:
					unit.append(1)
				else:
					unit.append(0)

			# add colormap as 256 element vector
			colormap = feature['properties']['colormap']
			colors = [int(j['color']) for j in colormap]
			vals = [float(j['percent']) for j in colormap]
			full_map = [0 for j in range(0, 256)]
			index = 0
			while index < len(colors):
				c = colors[index]
				full_map[c] = vals[index]
				index += 1

			unit.extend(full_map)

			unit.append(feature['properties']['label'])

			data.append(unit)

	return data







