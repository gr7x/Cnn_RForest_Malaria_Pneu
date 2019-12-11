from sklearn import tree, metrics
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'

import tflearn
import pickle
import threading
import sklearn
#import cPickle

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification

#tflearn.logging.set_verbosity(tflearn.logging.ERROR)

##function to pull images from the folders
def getData(pathy, output, a, b):
    imgsX  = []
    imgsY  = []
    print(pathy)
    print("Importing images please wait..")
    for p, s, f in os.walk(pathy):
          for name in f:
                 if name != '.DS_Store':
                        img = cv2.imread(os.path.join(p, name))
                        if img is not None:
                            imgsX.append(imagetoNP(img, a, b))
                            imgsY.append(output)
	        
    return imgsX, imgsY

## function to preprocess images
def imagetoNP(img, a, b):
    img = cv2.resize(img, (a,b)) 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image.shape
    scaled_gray_image = gray_image/255.0
    return scaled_gray_image

##model for pneumonia
def modular_net(first_layer, output_size): ##pmna's friend
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
    return model



def Build_Modular_Net(net_name, folder_array, save_path, sizeX, sizeY):
        ##build neural network
        NUM_EPOCHS = 12  
        BATCH_SIZE = 10 
        first_layer = [-1, sizeX, sizeY, 1]	
        ## prepare the data
        print(net_name)
        trainX, trainY = getData(folder_array[0][0], folder_array[0][1], sizeX, sizeY)
        noX, noY = getData(folder_array[1][0], folder_array[1][1], sizeX, sizeY)
        testX, testY = getData(folder_array[2][0], folder_array[2][1], sizeX, sizeY)
        notX, notY = getData(folder_array[3][0], folder_array[3][1], sizeX, sizeY)    
        
        X = np.concatenate((trainX, noX), axis = 0)
        Y = np.concatenate((trainY, noY), axis = 0)

        X = X.flatten().reshape(first_layer)
             
        testX = np.concatenate((testX, notX), axis = 0)
        testX = testX.reshape([-1, sizeX, sizeY, 1])

        testY = np.concatenate((testY, notY), axis = 0)

        MODEL = modular_net(first_layer, len(folder_array[0][1]))
        ## pass to the model
        MODEL.fit(X, Y, n_epoch=NUM_EPOCHS,
		  shuffle=True,
		  validation_set=(testX, testY),
		  show_metric=True,
		  batch_size=BATCH_SIZE,
		  run_id= net_name)

        MODEL.save(save_path+net_name+'.tfl')

def Build_Modular(net_name, folder_array, save_path, sizeX, sizeY):
        ## build random forrests to classify images with 30 decision trees
        clf = RandomForestClassifier(random_state=0, n_estimators = 30)
        NUM_EPOCHS = 4  
        BATCH_SIZE = 12 
        first_layer = [-1, sizeX, sizeY, 1]	

        print(net_name)
        trainX, trainY = getData(folder_array[0][0], folder_array[0][1], sizeX, sizeY)

        noX, noY = getData(folder_array[1][0], folder_array[1][1], sizeX, sizeY)

        testX, testY = getData(folder_array[2][0], folder_array[2][1], sizeX, sizeY)
        notX, notY = getData(folder_array[3][0], folder_array[3][1], sizeX, sizeY)
     
        X = np.concatenate((trainX, noX), axis = 0)
        Y = np.concatenate((trainY, noY), axis = 0)

        RX = X.reshape(-1, sizeX*sizeY)
        RY = Y 
        print("Fitting with Random Forest, please wait..")
        clf.fit(RX, RY)

        print(first_layer)
        X = X.flatten().reshape(first_layer)
             
        testX = np.concatenate((testX, notX), axis = 0)
        RTX = testX.reshape(-1, sizeX*sizeY)
        testY = np.concatenate((testY, notY), axis = 0)

        predictions = clf.predict(RTX)
        cm = sklearn.metrics.confusion_matrix(testY, predictions)
        print(cm)
        with open( save_path+net_name+'.cpickle', 'wb' ) as f:
              pickle.dump(clf, f)
          


def malaria():
## the structure of this object is [[path to infected images], [path to normal images], [path to unfected test images], [path to normal test images]]. The int after is the desired output from the model.
    malaria_folders = [['/root/Documents/workspace/ai/FINAL_PROJ/malaria/cell-images-for-detecting-malaria/cell_images/Parasitized/', 1], [ '/root/Documents/workspace/ai/FINAL_PROJ/malaria/cell-images-for-detecting-malaria/cell_images/Uninfected/' , 0], [ '/root/Documents/workspace/ai/FINAL_PROJ/malaria/cell-images-for-detecting-malaria/cell_images/Parasitized/' , 1],[ '/root/Documents/workspace/ai/FINAL_PROJ/malaria/cell-images-for-detecting-malaria/cell_images/Uninfected/' , 0]]
    sizeX = 80
    sizeY = 80

## this method takes the name of the test, the object above, the path to save to, and the size for all the images to be resized to
    Build_Modular('Malaria', malaria_folders, '/root/Documents/workspace/ai/FINAL_PROJ/built_nets/malaria/', sizeX, sizeY)




def pmna():
## the structure of this object is [[path to infected images], [path to normal images], [path to unfected test images], [path to normal test images]]. The tuple after is the desired output from the model.
    pmna_folders = [['/root/Documents/workspace/ai/FINAL_PROJ/pmna/chest-xray-pneumonia/chest_xray/train/PNEUMONIA', (1,0)], [ '/root/Documents/workspace/ai/FINAL_PROJ/pmna/chest-xray-pneumonia/chest_xray/train/NORMAL/' ,(0,1)], [ '/root/Documents/workspace/ai/FINAL_PROJ/pmna/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/' ,(1,0)],[ '/root/Documents/workspace/ai/FINAL_PROJ/pmna/chest-xray-pneumonia/chest_xray/test/NORMAL/' ,(0,1)]]
    sizeX = 80
    sizeY = 80

    Build_Modular_Net('PNEUMONIA', pmna_folders, '/root/Documents/workspace/ai/FINAL_PROJ/built_nets/pmna/', sizeX, sizeY)
   


def cancer():
## the structure of this object is [[path to infected images], [path to normal images], [path to unfected test images], [path to normal test images]]. The int after is the desired output from the model.
    cancer_folders = [['/root/Documents/workspace/ai/FINAL_PROJ/skin_cancer/skin-cancer-mnist-ham10000/HAM10000_images_part_1/', 1], [ '/root/Documents/workspace/ai/FINAL_PROJ/skin_cancer/skin-cancer-mnist-ham10000/HAM10000_images_part_2/' ,0], [ '/root/Documents/workspace/ai/FINAL_PROJ/skin_cancer/skin-cancer-mnist-ham10000/ham10000_images_part_1/' ,1],[ '/root/Documents/workspace/ai/FINAL_PROJ/skin_cancer/skin-cancer-mnist-ham10000/ham10000_images_part_2/' ,0]]
    sizeX = 100
    sizeY = 75

    Build_Modular('Cancer', cancer_folders, '/root/Documents/workspace/ai/FINAL_PROJ/built_nets/cancer/', sizeX, sizeY)

def main():
   ## start the three different models building
   malaria()
   pmna()
   cancer()
   

if __name__ == "__main__":
    main()

