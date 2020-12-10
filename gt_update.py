# takes a ground truth list and updates it to include additional traits
import numpy as np


# area of a polygon with an arbitrary number of vertices
def polygon_area(x, y):
	correction = x[-1] * y[0] - y[-1]* x[0]
	main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
	return 0.5*np.abs(main_area + correction)




def gt_update(gt):
	i = 0
	while i < len(gt['features']):
		feature = gt['features'][i]
		# print("Updating gt object" + str(i))
		if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
			
			# add area, number of vertices, and number of contours of each object
			feature['properties']['contour_count'] = len(feature['geometry']['coordinates'])
			ob_verts = feature['geometry']['coordinates'][0] # first list of vertices is the object outline
			xs = np.array([j[0] for j in ob_verts])
			ys = np.array([j[1] for j in ob_verts])
			a = polygon_area(xs, ys)
			# remainder of the contours are obj cutouts
			cutout_areas = [polygon_area(np.array([j[0] for j in c]), np.array([j[1] for j in c]))
														for c in feature['geometry']['coordinates'][1:]]
			# we subtract the areas of the cutout contours to get the object area
			# feature['properties']['area'] = a - sum(cutout_areas). # Model was not trained this way. Will retrain and fix later.
			feature['properties']['area'] = a
			feature['properties']['vertex_count'] = len(xs)

			# add ratio of height / width
			width = abs(max(xs) - min(xs))
			height = abs(max(ys) - min(ys))
			shape_ratio = height / width
			feature['properties']['shape_ratio'] = shape_ratio

			edges = []
			for j in range(-1, len(xs) - 1):
				e = np.sqrt((xs[j + 1] - xs[j])**2 + (ys[j + 1] - ys[j])**2)
				edges.append(e)
			shortest = min(edges)
			longest = max(edges)
			mean_edge = np.mean(edges)
			median_edge = np.median(edges)
			edge_sum = sum(edges)
			feature['properties']['shortest_edge'] = shortest
			feature['properties']['longest_edge'] = longest
			feature['properties']['mean_edge'] = mean_edge
			feature['properties']['median_edge'] = median_edge
			feature['properties']['edge_sum'] = edge_sum

		else:
			# assign Ground, Trajectory, and Slingshot 
			# 0 for all numeric fields, empty full_colormap, "None" label
			# ensures all keys exist for every object
			feature['properties']['area'] = 0
			feature['properties']['vertex_count'] = 0
			feature['properties']['contour_count'] = 0
			feature['properties']['shortest_edge'] = 0
			feature['properties']['longest_edge'] = 0
			feature['properties']['mean_edge'] = 0
			feature['properties']['median_edge'] = 0
			feature['properties']['edge_sum'] = 0
			feature['properties']['shape_ratio'] = 0

		i += 1
	return gt





# labels = []
# for g in ground_truths:
# 	for ob in g['features']:
# 		if ob['properties']['label'] not in labels:
# 			labels.append(ob['properties']['label'])



# vertex_counts = []
# for g in gt_list:
# 	for ob in g['features']:
# 		if 'coordinates' in ob['geometry'].keys():
# 			vc = len(ob['geometry']['coordinates'][0])
# 			if vc not in vertex_counts:
# 				vertex_counts.append(vc)


# example test
# import json
# from gt_update import *
# from get_data import *
# f = open("non-novelty_185_non-noisy_groundtruths_v036.txt")
# gt_list = json.load(f)
# data = []
# for gt in gt_list:
# 	gt_update(gt)
# 	d = get_data(gt)
# 	data.extend(d)









