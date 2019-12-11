import os

os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'

import tflearn
import cv2
import pickle
import sklearn
import unittest

import tensorflow as tf

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.ensemble import RandomForestClassifier

class CnnLoader:
	def __init__(self):
		pass

	def imagetoNP(self, path, a, b):
		img = cv2.imread(path)
		img = cv2.resize(img, (a,b)) 
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_image.shape
		scaled_gray_image = gray_image/255.0
		return scaled_gray_image

	def LoadPMNA(self, path, first_layer, output_size):
		input_layer = input_data(shape=[None, first_layer[1], first_layer[2], first_layer[3]])
		conv_layer_1  = conv_2d(input_layer,
				    nb_filter=100,
				    filter_size=5,
				    activation='relu',
				    name='conv_layer_1')
		pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
		conv_layer_2 = conv_2d(pool_layer_1,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_2')
		pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
		conv_layer_3 = conv_2d(pool_layer_2,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_3')
		pool_layer_3 = max_pool_2d(conv_layer_2, 2, name='pool_layer_3')
		conv_layer_4 = conv_2d(pool_layer_3,
			       nb_filter=60,
			       filter_size=3,
			       activation='sigmoid',
			       name='conv_layer_4')
		pool_layer_4 = max_pool_2d(conv_layer_2, 2, name='pool_layer_4')
		conv_layer_5 = conv_2d(pool_layer_4,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu6',
			       name='conv_layer_5')
		pool_layer_5 = max_pool_2d(conv_layer_2, 2, name='pool_layer_5')
		conv_layer_6 = conv_2d(pool_layer_5,
			       nb_filter=60,
			       filter_size=3,
			       activation='tanh',
			       name='conv_layer_6')
		pool_layer_6 = max_pool_2d(conv_layer_2, 2, name='pool_layer_6')
		conv_layer_7 = conv_2d(pool_layer_6,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_7')
		pool_layer_7 = max_pool_2d(conv_layer_7, 2, name='pool_layer_7')
		conv_layer_8 = conv_2d(pool_layer_7,
			       nb_filter=40,
			       filter_size=3,
			       activation='softmax',
			       name='conv_layer_8')
		pool_layer_8 = max_pool_2d(conv_layer_8, 2, name='pool_layer_8')
		conv_layer_9 = conv_2d(pool_layer_8,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_9')
		pool_layer_9 = max_pool_2d(conv_layer_9, 2, name='pool_layer_9')
		conv_layer_10 = conv_2d(pool_layer_9,
			       nb_filter=40,
			       filter_size=3,
			       activation='tanh',
			       name='conv_layer_10')
		pool_layer_10 = max_pool_2d(conv_layer_10, 2, name='pool_layer_10')
		conv_layer_11 = conv_2d(pool_layer_10,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_11')
		pool_layer_11 = max_pool_2d(conv_layer_11, 2, name='pool_layer_11')
		conv_layer_12 = conv_2d(pool_layer_11,
			       nb_filter=40,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_12')
		pool_layer_12 = max_pool_2d(conv_layer_12, 2, name='pool_layer_12')
		conv_layer_13 = conv_2d(pool_layer_12,
			       nb_filter=60,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_13')
		pool_layer_13 = max_pool_2d(conv_layer_13, 2, name='pool_layer_13')
		conv_layer_14 = conv_2d(pool_layer_13,
			       nb_filter=40,
			       filter_size=3,
			       activation='relu',
			       name='conv_layer_14')
		pool_layer_14 = max_pool_2d(conv_layer_14, 2, name='pool_layer_14')
		fc_layer_1  = fully_connected(pool_layer_14, 100,
				          activation='relu',
				          name='fc_layer_1')
		fc_layer_2 = fully_connected(fc_layer_1, 2,
				         activation='softmax',
				         name='fc_layer_2')
		network = regression(fc_layer_2, optimizer='Adam',
				 loss='categorical_crossentropy',
				 learning_rate=0.001)
		model = tflearn.DNN(network)
		model.load(path, weights_only=True)
		return model

	def PredictPnma(self, model, img):
		a =  model.predict(img)
		if a[0][0] > a[0][1]:
			return (1,0)
		else:
			return (0,1)

	def pmna(self, image_path, model):
		first_layer = [-1, 80, 80, 1]
		output_size = 1
		img = self.imagetoNP(image_path, 80, 80)
		#mod = self.LoadPMNA(model_path, first_layer, output_size)
		img = img.reshape(first_layer)
		return(self.PredictPnma(model , img))

class ForestLoader():
	def __init__(self):
		pass

	def imagetoNP(self, path, a, b):
		img = cv2.imread(path)
		img = cv2.resize(img, (a,b)) 
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_image.shape
		scaled_gray_image = gray_image/255.0
		return scaled_gray_image

	def LoadModel(self, model_path):
		with open(model_path , 'rb' ) as f:
			rf = pickle.load(f)
			return rf	
	def Predict(self, img_path, model_path, a, b):
		model = self.LoadModel(model_path)
		img = self.imagetoNP(img_path, a, b)
		RTX = img.reshape(-1, a*b)
		return model.predict(RTX)

class UnitTests(unittest.TestCase):

		#If you're training a classifier, then you also need to submit a dozen or so examples (images, audio, texts, videos, etc.) on which we can run your classifier.
		## fix unit tests
##	def __init__(self, a):
##		pass

	def setUp(self):
		self.PATH_TO_FINAL_PROJ = "/root/Documents/workspace/ai/FINAL_PROJ/FinalSubmission"
		## UPDATE THE ABOVE LINE FOR THE PATH TO BUILT NETS IN YOUR SYSTEM
		self.pmna_model = self.PATH_TO_FINAL_PROJ + "/built_nets/pmna/PNEUMONIA.tfl"
		self.malaria_model = self.PATH_TO_FINAL_PROJ +"/built_nets/malaria/Malaria.cpickle"
		self.cancer_model  = self.PATH_TO_FINAL_PROJ + "/built_nets/cancer/Cancer.cpickle"
		tf.reset_default_graph()
		self.l = CnnLoader()
		self.pModel = self.l.LoadPMNA( self.pmna_model, [-1, 80, 80, 1], 1)

		

## =========== Pneumonia Tests ============##
	def test_p1(self):
		
	
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/PNEUMONIA/person1946_bacteria_4874.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(1,0))

	
	def test_p2(self):
	
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/PNEUMONIA/person1946_bacteria_4875.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(1,0))


	
	def test_p3(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/PNEUMONIA/person1947_bacteria_4876.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(1,0))


	def test_p4(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/PNEUMONIA/person1949_bacteria_4880.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(1,0))


	def test_p5(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/PNEUMONIA/person1950_bacteria_4881.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(1,0))


	def test_p6(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/PNEUMONIA/person1951_bacteria_4882.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(1,0))



	def test_p7(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/NORMAL/NORMAL2-IM-1427-0001.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(0,1))



	def test_p8(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/NORMAL/NORMAL2-IM-1430-0001.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(0,1))


	def test_p9(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/NORMAL/NORMAL2-IM-1431-0001.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(0,1))


	def test_p10(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/NORMAL/NORMAL2-IM-1436-0001.jpeg"

		self.assertEqual(self.l.pmna(image_path, self.pModel),(0,1))

	def test_p11(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/NORMAL/NORMAL2-IM-1437-0001.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(0,1))


	def test_p12(self):
		image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/pmna/NORMAL/NORMAL2-IM-1438-0001.jpeg"	

		self.assertEqual(self.l.pmna(image_path, self.pModel),(0,1))


## ========= malaria tests ===========##
	def test_m1(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/normal/C2NThinF_IMG_20150604_114730_cell_178.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[0])
		del m

	def test_m2(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/parasitized/C39P4thinF_original_IMG_20150622_105102_cell_107.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[1])
		del m

	def test_m3(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/parasitized/C39P4thinF_original_IMG_20150622_105253_cell_90.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[1])
		del m
	def test_m4(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/parasitized/C39P4thinF_original_IMG_20150622_105253_cell_91.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[1])
		del m

	def test_m5(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/parasitized/C39P4thinF_original_IMG_20150622_105253_cell_92.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[1])
		del m

	def test_m6(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/parasitized/C39P4thinF_original_IMG_20150622_105253_cell_93.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[1])
		del m

	def test_m7(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/normal/C2NThinF_IMG_20150604_114730_cell_173.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[0])
		del m

	def test_m8(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/normal/C2NThinF_IMG_20150604_114730_cell_183.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[0])
		del m

	def test_m9(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/normal/C2NThinF_IMG_20150604_114751_cell_38.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[0])
		del m

	def test_m10(self):
		m = ForestLoader()
			
		img_path = self.PATH_TO_FINAL_PROJ + "/predict_images/mal/normal/C2NThinF_IMG_20150604_114751_cell_38.png"

		self.assertEqual(m.Predict(img_path, self.malaria_model, 80, 80),[0])
		del m

	def test_m11(self):
		pass

	def test_m12(self):
		pass


## ========= cancer tests ===========##
	def test_c1(self):
		# figure out what's wrong with cancer?
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0029322.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c

	def test_c2(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0029323.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c
	
	def test_c3(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/normal/ISIC_0029324.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[0])
		del c

	def test_c4(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0029325.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c

	def test_c5(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/normal/ISIC_0029326.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[0])
		del c


	def test_c6(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/normal/ISIC_0024309.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[0])
		del c

	def test_c7(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0024310.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c

	def test_c8(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0024311.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c

	def test_c9(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0024312.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c

	def test_c10(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0024313.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c

	def test_c11(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/normal/ISIC_0025792.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[0])
		del c


	def test_c12(self):
		c = ForestLoader()
		
		c_image_path = self.PATH_TO_FINAL_PROJ + "/predict_images/cancer/diseased/ISIC_0025793.jpg"
		self.assertEqual(c.Predict(c_image_path, self.cancer_model, 75, 100),[1])
		del c



def main():
	unittest.main()



if __name__ == "__main__":
	main()
	








