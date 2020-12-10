# classifies the objects in a single ground truth list
# runs a secondary novelty classifier to identify novelty
# in shape or in material.  

import numpy as np
import pickle
import sklearn.linear_model
from scipy.special import expit
import os.path as osp

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


class Classifier(object):
	def __init__(self):

		# hotpatch to stop normalization of class probabilities
		sklearn.linear_model._base.LinearClassifierMixin._predict_proba_lr = _predict_proba_lr_hotpatch

		classifer_path = osp.join('sciencebirds', 'classifier')
		# self.model = pickle.load(open(osp.join(classifer_path, 'model_lbfgs_v0411.sav'), 'rb'))
		# self.shape_model = pickle.load(open(osp.join(classifer_path, 'model_lbfgs_v0411_shape.sav'), 'rb'))
		# self.material_model = pickle.load(open(osp.join(classifer_path, 'model_lbfgs_v0411_material.sav'), 'rb'))
		self.model = pickle.load(open('model_lbfgs_v0411.sav', 'rb'))
		self.shape_model = pickle.load(open('model_lbfgs_v0411_shape.sav', 'rb'))
		self.material_model = pickle.load(open('model_lbfgs_v0411_material.sav', 'rb'))
		# self.shape_novelty_threshold = 0.0628  # lowest maximum class-wise probability
		# self.material_novelty_threshold = 0.99 # lowest maximum class-wise probability 
		self.thresholds = {
		'Platform' : 0.9905137882651,
		'TNT' : 0.997858342901647,
		'bird_black' : 0.99879748454453,
		'bird_blue' : 0.994943154143828,
		'bird_red' : 0.979716350681756,
		'bird_white' : 0.995197859667436,
		'bird_yellow' : 0.9984648941322,
		'ice_circle' : 0.990110008003175,
		'ice_circle_small' : 0.990761954612923,
		'ice_rect_big' : 0.99208965856922,
		'ice_rect_fat' : 0.988301903373056,
		'ice_rect_medium' : 0.106888891910243,
		'ice_rect_small' : 0.2426328003766615,
		'ice_rect_tiny' : 0.971264450657221,
		'ice_square_hole' : 0.997998527005163,
		'ice_square_small' : 0.970623676505865,
		'ice_square_tiny' : 0.994897365218952,
		'ice_triang' : 0.992244352455620,
		'ice_triang_hole' : 0.995009164414971,
		'pig_basic_small' : 0.989811362931037,
		'stone_circle' : 0.984436026750760,
		'stone_circle_small' : 0.987668511679872,
		'stone_rect_big' : 0.991538114625887,
		'stone_rect_fat' : 0.956862177498844,
		'stone_rect_medium' : 0.10203630697500,
		'stone_rect_small' : 0.1159825216099624,
		'stone_rect_tiny' : 0.122532426278367,
		'stone_square_hole' : 0.996623969520941,
		'stone_square_small' : 0.985334790833571,
		'stone_square_tiny' : 0.995695721227861,
		'stone_triang' : 0.986549607856204,
		'stone_triang_hole' : 0.988628964198871,
		'wood_circle' : 0.986198033201801,
		'wood_circle_small' : 0.99047338735391,
		'wood_rect_big' : 0.99170763801300,
		'wood_rect_fat' : 0.993818230752573,
		'wood_rect_medium' : 0.2704048563908218,
		'wood_rect_small' : 0.622388470542563,
		'wood_rect_tiny' : 0.946091934099135,
		'wood_square_hole' : 0.9979672641752,
		'wood_square_small' : 0.991405154821151,
		'wood_square_tiny' : 0.99854539193297,
		'wood_triang' : 0.991152945401726,
		'wood_triang_hole' : 0.992292610809064
		}
		self.shape_thresholds = {
		'Platform' : 0.0627997090512703,
		'TNT' : 0.900288078557436,
		'bird_black' : 0.998170566030801,
		'bird_blue' : 0.985984966321337,
		'bird_red' : 0.908184378206421,
		'bird_white' : 0.994948493257766,
		'bird_yellow' : 0.99835406302888,
		'circle' : 0.99656020017687,
		'circle_small' : 0.374490722105587,
		'pig_basic_small' : 0.766876579326704,
		'rect_big' : 0.999944497362700,
		'rect_fat' : 0.0868493724289337,
		'rect_medium' : 0.164307707765943,
		'rect_small' : 0.09653562617444,
		'rect_tiny' : 0.082851701475658,
		'square_hole' : 0.999977537360161,
		'square_small' : 0.99898798498205,
		'square_tiny' : 0.999979994001707,
		'triang' : 0.999974955094864,
		'triang_hole' : 0.998649767253206
		}
		self.material_thresholds = {
		'Platform' : 0.999858052508693,
		'TNT' : 0.990378834019750,
		'bird' : 0.991645634288601,
		'ice' : 0.999065148468243,
		'pig' : 0.996423279322598,
		'stone' : 0.9986484950157,
		'wood' : 0.99614008759486
		}
		self.simba_name_dict = {
		'Ground' : 'ground',
		'Trajectory' : 'trajectory',
		'Slingshot' : 'slingshot',
		'Platform' : 'platform',
		'TNT' : 'tnt',
		'bird_black' : 'black_bird',
		'bird_blue' : 'blue_bird',
		'bird_red' : 'red_bird',
		'bird_white' : 'white_bird',
		'bird_yellow' : 'yellow_bird', 
		'ice_circle' : 'ice',
		'ice_circle_small' : 'ice',
		'ice_rect_big' : 'ice',
		'ice_rect_fat' : 'ice',
		'ice_rect_medium' : 'ice',
		'ice_rect_small' : 'ice',
		'ice_rect_tiny' : 'ice',
		'ice_square_hole' : 'ice',
		'ice_square_small' : 'ice',
		'ice_square_tiny' : 'ice',
		'ice_triang' : 'ice',
		'ice_triang_hole' : 'ice',
		'pig_basic_medium' : 'pig',
		'pig_basic_small' : 'pig',
		'stone_circle' : 'stone',
		'stone_circle_small' : 'stone',
		'stone_rect_big' : 'stone',
		'stone_rect_fat' : 'stone',
		'stone_rect_medium' : 'stone',
		'stone_rect_small' : 'stone',
		'stone_rect_tiny' : 'stone',
		'stone_square_hole' : 'stone',
		'stone_square_small' : 'stone',
		'stone_square_tiny' : 'stone',
		'stone_triang' : 'stone',
		'stone_triang_hole' : 'stone',
		'wood_circle' : 'wood',
		'wood_circle_small' : 'wood',
		'wood_rect_big' : 'wood',
		'wood_rect_fat' : 'wood',
		'wood_rect_medium' : 'wood',
		'wood_rect_small' : 'wood',
		'wood_rect_tiny' : 'wood',
		'wood_square_hole' : 'wood',
		'wood_square_small' : 'wood',
		'wood_square_tiny' : 'wood',
		'wood_triang' : 'wood',
		'wood_triang_hole' : 'wood',
		'novel' : 'novel'
		}

	
	def simba_name_converter(self, name):
		return self.simba_name_dict[name]


	# area of a polygon with an arbitrary number of vertices
	def polygon_area(self, x, y):
		correction = x[-1] * y[0] - y[-1]* x[0]
		main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
		return 0.5*np.abs(main_area + correction)


	# takes a ground truth list and updates it to include additional features
	def _gt_update(self, gt):
		i = 0
		while i < len(gt['features']):
			feature = gt['features'][i]
			if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
				
				# add area, number of vertices, and number of contours of each object
				feature['properties']['contour_count'] = len(feature['geometry']['coordinates'])
				ob_verts = feature['geometry']['coordinates'][0] # first list of vertices is the object outline
				xs = np.array([j[0] for j in ob_verts])
				ys = np.array([j[1] for j in ob_verts])
				a = self.polygon_area(xs, ys)
				# remainder of the contours are obj cutouts
				cutout_areas = [self.polygon_area(np.array([j[0] for j in c]), np.array([j[1] for j in c]))
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
				# 0 for all numeric fields
				# ensures keys exist for every object
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


	# converts ground truth data into feature vectors for the classifier 
	def _get_data(self, gt):
		self._gt_update(gt)

		# convert each object to a feature vector
		data = []
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
				# feature['properties']['full_colormap'] = full_map
				unit.extend(full_map)

				data.append(unit)
		return data


	# Level 3 Novelty Detector
	def _l3(self, gt):
		cushion = 17 # number of pixels of overlap allowed when comparing object locations
		novelty_detection_flag = 0

		slingshot = [i for i in gt['features'] if i['properties']['label'] == 'Slingshot']
		
		# ----------------------------------------------------------
		# check for multiple slingshots
		if len(slingshot) > 1:
			novelty_detection_flag = 1

		slingshot = slingshot[0]

		# ----------------------------------------------------------
		# check slingshot location (on left half of screen)
		slingshot_coords = slingshot['geometry']['coordinates'][0]
		slingshot_xs = np.array([j[0] for j in slingshot_coords])
		slingshot_ys = np.array([j[1] for j in slingshot_coords])
		if len([i for i in slingshot_xs if i > 420]) > 0:
			novelty_detection_flag = 1

		# ----------------------------------------------------------
		# check that at least one bird is classified
		birds = [i for i in gt['features'] if 'bird' in i['properties']['kdl_label']]
		if len(birds) == 0:
			novelty_detection_flag = 1

		# ----------------------------------------------------------
		# sum all colormaps, then count number of values > 0. If it is less than ~10, probably novel?
		objects = [ob for ob in gt['features'] if ob['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']]
		full_map = [0 for j in range(0, 256)]
		for ob in objects:
			colormap = ob['properties']['colormap']
			for i in colormap:
				full_map[i['color']] += i['percent']
		if len([i for i in full_map if i > 0]) < 10:
			novelty_detection_flag = 1

		return novelty_detection_flag


	# classifies all objects in a single ground truth list
	def classify(self, gt, switchoff=False):
		if switchoff:
			return gt
		if len(gt) < 1: # if gt is empty 
			return gt

		data = self._get_data(gt)
		if len(data) < 1:  # if there are only basic objects data will be empty
			return gt

		data = np.array(data) # format for sklearn
		labels = self.model.predict(data)
		
		# uncomment to classify medium pigs as pig_basic_samll
		# for i in range(len(labels)):
		# 	if labels[i] == 'pig_basic_medium':
		# 		labels[i] == 'pig_basic_small'

		probs = self.model.predict_proba(data)
		p = [np.max(i) for i in probs]  # max class probability across all classes for each unit 
		data2 = [] # data for secondary classifier
		labels2 = [] # labels for secondary classifier
		label_index = 0
		for i in range(len(gt['features'])):
			feature = gt['features'][i]
			if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot']:
				feature['properties']['kdl_label'] = labels[label_index]
				if feature['properties']['kdl_label'] in ['TNT', 'Platform']:
					mat = feature['properties']['kdl_label']
				else:
					mat = ''
					ch = 0
					while feature['properties']['kdl_label'][ch] != '_':
						mat = mat + feature['properties']['kdl_label'][ch]
						ch += 1
				feature['properties']['material'] = mat
				if mat in ['TNT', 'Platform']:
					feature['properties']['shape'] = mat
				elif mat in ['pig', 'bird']:
					feature['properties']['shape'] = feature['properties']['kdl_label']
				else:
					feature['properties']['shape'] = feature['properties']['kdl_label'][ch+1:]
				
				# detect novelty
				if p[label_index] < self.thresholds[feature['properties']['kdl_label']]:
					print(feature['properties']['kdl_label'] + " - " + str(p[label_index]) + " : " + str(self.thresholds[feature['properties']['kdl_label']]))
					feature['properties']['novel'] = 1
					data2.append(data[label_index])
					labels2.append(labels[label_index])
				else:
					feature['properties']['novel'] = 0

				label_index += 1

			else:
				feature['properties']['kdl_label'] = feature['properties']['label']
				feature['properties']['material'] = feature['properties']['label']
				feature['properties']['shape'] = feature['properties']['label']
				feature['properties']['novel'] = 0

			# set simba_type 
			if feature['properties']['novel'] == 1:
				feature['properties']['simba_type'] = 'novel'
			else:
				feature['properties']['simba_type'] = self.simba_name_converter(feature['properties']['kdl_label'])
				

		if len(data2) > 0:
			shape_data = []
			material_data = []
			for i in range(len(data2)):
				shape_data.append(data2[i][:24])
				material_data.append(data2[i][24:])
			# format for sklearn
			shape_data = np.array(shape_data) 
			material_data = np.array(material_data)

			# run secondary classification of novel objects
			material_predictions = self.material_model.predict(material_data)
			shape_predictions = self.shape_model.predict(shape_data)
			shape_probs = self.shape_model.predict_proba(shape_data)
			shape_probs = [np.max(i) for i in shape_probs]
			material_probs = self.material_model.predict_proba(material_data)
			material_probs = [np.max(i) for i in material_probs]

			label_index = 0 
			for i in range(len(gt['features'])):
				feature = gt['features'][i]
				if feature['properties']['label'] not in ['Ground', 'Trajectory', 'Slingshot'] and feature['properties']['novel'] == 1:
					# detect shape novelty
					if shape_probs[label_index] < self.shape_thresholds[feature['properties']['shape']]:
						feature['properties']['shape_novelty'] = 1
						feature['properties']['novel_shape'] = 'novel'

					else:
						feature['properties']['shape_novelty'] = 0
						feature['properties']['novel_shape'] = shape_predictions[label_index]

					# detect material novelty
					if material_probs[label_index] < self.material_thresholds[feature['properties']['material']]:
						feature['properties']['material_novelty'] = 1
						feature['properties']['novel_material'] = 'novel'
					else:
						feature['properties']['material_novelty'] = 0
						feature['properties']['novel_material'] = material_predictions[label_index]

					label_index += 1

		# detect Level 3 Novelty
		gt['l3_detection'] = self._l3(gt)

		return gt
