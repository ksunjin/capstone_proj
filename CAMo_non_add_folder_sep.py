#-*- coding:utf-8 -*-
import face_recognition
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from os import listdir
import os
import cv2
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image

#setx OPENCV_VIDEOIO_PRIORITY_MSMF 0 

BASE_DIR = "facerec1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = glob.glob(os.path.join("facerec1/*.jpg"))

#detector
detector = MTCNN()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#_var
PT = []
oks = dict()
crop_num = 0


def agree():
    imgnum = 0
    i = 0
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('agree', frame)
        for i in range(10):
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.imwrite("facerec1/agreed_people_" + str(imgnum) + ".jpg", frame)
                imgnum += 1
                i += 1
        if imgnum == 11:
            break
    video_capture.release()
    cv2.destroyAllWindows()

#recognize and crop face
def face_rec_crop():
    #store_path_var
    num = 0
    for p in image_path:
        image = face_recognition.load_image_file(p)
        results = detector.detect_faces(image)
        
        if len(results) == 0 :
            print("No faces detected.")
        elif len(results) > 0:

            print("Number of faces detected: {} 명".format(len(results)))  
            
            for result in results:
                
                global bounding_box
                global keypoints

                bounding_box = result['box']
                keypoints = result['keypoints']

                face_position = [int(x) for x in bounding_box]
                x = face_position[0]
                y = face_position[1]
                w = face_position[2]
                h = face_position[3]

                if (y - int(h / 4))> 0:
                    cropped = image[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
                    train_data = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    #cv2.imshow('c', cropped)
                    #k = cv2.waitKey(0)
                    save_path = os.path.join("train/ok_" + str(num)+ ".jpg")
                    cv2.imwrite(save_path, train_data)
                    num += 1
                    global crop_num
                    crop_num += 1
                    global PT
                    PT.append(save_path)
                else:
                    print("cannot")
                crop_num = crop_num *10

def preTreat(cropped_path):
    i = 0
    
    train_aug_gen = ImageDataGenerator(rotation_range = 40, brightness_range=[0.5, 1.5], width_shift_range=0.2, height_shift_range=0.2, rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
    for img_path in cropped_path:
        image_ = load_img(img_path)
        x = img_to_array(image_)
        x = x.reshape((1,)+x.shape)
        
        for batch in enumerate(train_aug_gen.flow(x, batch_size = 1, save_to_dir = "train", save_prefix= "ok", save_format = 'jpg')):
            i += 1
            if i > 300:
                break

if __name__ == '__main__':
    print("\n [INFO] Cheeze! Please look at the camera for a moment")
    agree()
    print("\n [INFO] Processing ...")
    face_rec_crop()
    print("\n [INFO] Image Processing ...")
    preTreat(PT)
    print("\n [INFO] Image Data Generating finish")

def load_model():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    from keras.models import model_from_json
    model.load_weights("vgg_face_weights.h5")

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    return vgg_face_descriptor

print("\n [INFO] model training...")
model = load_model()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)
    return img

def predict():
    ok_pictures = "train"

    for file in listdir(ok_pictures):
        ok, extension = file.split(".")
        oks[ok] = model.predict(preprocess_image("train/%s.jpg" % (ok)), steps=1)[0,:]
    print("\n [INFO] It's almost over..")

predict()

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def livestream():
    color = (67, 67, 67)
    video_capture = cv2.VideoCapture(0)
    i = 0
    while True:
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        faces = faceCascade.detectMultiScale(frame, 1.3, 5)

        for(x,y,w,h) in faces:

            if w > 130:
                detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
                
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                #img_pixels /= 255
                #employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                img_pixels /= 127.5
                img_pixels -= 1
                
                captured_representation = model.predict(img_pixels)[0,:]
                
                found = 0
                for i in oks:
                    ok_name = i
                    representation = oks[i]

                    similarity = findCosineSimilarity(representation, captured_representation)
                    if(similarity < 0.30):
                        cv2.putText(frame, ok_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        found = 1
                        break
                    else:
                        cv2.putText(frame, 'passerby', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        passerby = frame[int(y):int(y+h), int(x):int(x+w)]
                        kernel = np.ones((5,5), np.float32)/25
                        blur = cv2.filter2D(passerby, -1, kernel)
                        frame[int(y):int(y+h), int(x):int(x+w)] = blur
                        #passerby = frame[int(y):int(y+h), int(x):int(x+w)]
                        #passerby = cv2.GaussianBlur(passerby, (99, 99), 30)
                        #frame[int(y):int(y+h), int(x):int(x+w)] = passerby

        
                cv2.line(frame,(int((x+x+w)/2),y+15),(x+w,y-20),color,1)
                cv2.line(frame,(x+w,y-20),(x+w+10,y-20),color,1)
                    
        cv2.imshow('livestream',frame)   

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

print("\n [INFO] Start live straming...")
livestream()