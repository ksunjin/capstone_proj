#-*- coding:utf-8 -*-
#-------------------------------------------------------------------------
#출처: github etc.. 모두 감사합니다..

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
import threading
import time
import dlib
import timeit
#-------------------------------------------------------------------------

#기본설정
BASE_DIR = "C:\\Users\\user\\Desktop\\facerec1\\"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
image_path = glob.glob(os.path.join(BASE_DIR + "capture\\*.jpg"))

#탐지 모델
detector = MTCNN()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#변수
PT = []
ok_name = dict()
oks = dict()
crop_num = 0

#동의 받기 후 사진 촬영(10번)
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
                cv2.imwrite(BASE_DIR + "capture\\agreed_people_" + str(imgnum) + ".jpg", frame)
                imgnum += 1
                i += 1
        if imgnum == 11:
            break
    video_capture.release()
    cv2.destroyAllWindows()

#얼굴 탐지 후 크롭
def face_rec_crop():
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
                    save_path = os.path.join(BASE_DIR + "train\\ok_" + str(num)+ ".jpg")
                    cv2.imwrite(save_path, train_data)
                    num += 1
                    global crop_num
                    crop_num += 1
                    global PT
                    PT.append(save_path)
                else:
                    print("cannot")
                crop_num = crop_num *10

#사진 전처리
def preTreat(cropped_path):
    
    train_aug_gen = ImageDataGenerator(rotation_range = 40, brightness_range=[0.5, 1.5], width_shift_range=0.2, height_shift_range=0.2, rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
    for img_path in cropped_path:
        i = 0
        image_ = load_img(img_path)
        x = img_to_array(image_)
        x = x.reshape((1,)+x.shape)
        
        for batch in enumerate(train_aug_gen.flow(x, batch_size = 16, save_to_dir =  BASE_DIR + "train", save_prefix= "ok", save_format = 'jpg', shuffle=False)):
            i += 1
            if i > 30: #better than 10
                break

#모델 학습
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

#사진 배열 전처리
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)
    return img

#예측
def predict():
    ok_pictures = "C:\\Users\\user\\Desktop\\facerec1\\train"

    for file in listdir(ok_pictures):
        ok, extension = file.split(".")
        oks[ok] = model.predict(preprocess_image("C:\\Users\\user\\Desktop\\facerec1\\train\\%s.jpg" % (ok)), steps=1)[0,:]

#유사도 측정
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

#이름 설정 및 저장
def naming(ok_name, fid):
    time.sleep(2)
    ok_name[ fid ] = "Agreed" + str(fid)
    
#생방송 스트리밍 시작
def livestream():
    color = (67, 67, 67)
    rectangleColor = (0,165,255)
    
    video_capture = cv2.VideoCapture(0)

    frameCounter = 0
    currentFaceID = 0

    faceTrackers = {}
    cv2.startWindowThread()

    while True:
        _, frame = video_capture.read()
        baseImage = cv2.flip(frame, 1)

        frameCounter += 1 
        fidsToDelete = []
        
        for fid in faceTrackers.keys():
            trackingQuality = faceTrackers[ fid ].update( baseImage )

            if trackingQuality < 7:
                fidsToDelete.append( fid )

        for fid in fidsToDelete:
            print("Removing fid " + str(fid) + " from list of trackers")
            faceTrackers.pop( fid , None )

        #20 프레임씩 측정
        if (frameCounter % 20) == 0:
            #faces = faceCascade.detectMultiScale(frame, 1.3, 5)
            #harr 보다 위에서 썼던 mtcnn을 사용하는 게 탐지 하는 데에 정확함
            faces = detector.detect_faces(baseImage)
                
            for face in faces:
                global bounding_box
                global keypoints

                bounding_box = face['box']
                keypoints = face['keypoints']

                face_position = [int(x) for x in bounding_box]
                x = face_position[0]
                y = face_position[1]
                w = face_position[2]
                h = face_position[3]

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h


                matchedFid = None

                for fid in faceTrackers.keys():
                    tracked_position =  faceTrackers[fid].get_position()

                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                            ( t_y <= y_bar   <= (t_y + t_h)) and 
                            ( x   <= t_x_bar <= (x   + w  )) and 
                            ( y   <= t_y_bar <= (y   + h  ))):
                        matchedFid = fid

                if matchedFid is None:

                    print("Creating new tracker " + str(currentFaceID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(baseImage,
                                        dlib.rectangle( x-10,
                                                        y-20,
                                                        x+w+10,
                                                        y+h+20))

                    faceTrackers[ currentFaceID ] = tracker
                    
                    t = threading.Thread(target = naming,
                                            args=(ok_name, currentFaceID))
                    t.start()
                    
                    currentFaceID += 1

        for fid in faceTrackers.keys():
            tracked_position =  faceTrackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            cv2.rectangle(baseImage, (t_x, t_y), (t_x + t_w , t_y + t_h),rectangleColor ,2)

            if fid in ok_name.keys():
                if w > 130:
                    detected_face = baseImage[y:y+h, x:x+w]
                    detected_face = cv2.resize(detected_face, (224, 224))
                    
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    img_pixels /= 127.5
                    img_pixels -= 1

                    captured_representation = model.predict(img_pixels)[0,:]
                    
                    found = 0
                    i = 0
                    
                    for i in oks:
                        representation = oks[i]

                        similarity = findCosineSimilarity(representation, captured_representation)
                        
                        if(similarity < 0.20):
                            cv2.putText(baseImage, ok_name[fid], (int(t_x + t_w/2), int(t_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            found = 1
                            break
                        
                        else:
                            cv2.putText(baseImage, 'Passerby', (int(t_x + t_w/2), int(t_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            passerby = baseImage[y:y+h, x: x+w]
                            kernel = np.ones((5,5), np.float32)/25
                            blur = cv2.filter2D(passerby, -1, kernel)
                            baseImage[y:y+h, x: x+w] = blur
                            
            else:
                cv2.putText(baseImage, "Detecting..." , 
                            (int(t_x + t_w/2), int(t_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                
        baseImage = cv2.resize(baseImage,(775,600))

        cv2.imshow("result-image", baseImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    print("\n [INFO] Cheeze! Please look at the camera for a moment")
    agree()
    print("\n [INFO] Processing ...")
    face_rec_crop()
    print("\n [INFO] Image Processing ...")
    preTreat(PT)
    print("\n [INFO] Image Data Generating finish")

    start = timeit.default_timer()
    print("\n [INFO] model loading...")
    model = load_model()
    stop = timeit.default_timer()
    print(stop - start)
    print("\n [INFO] model predicting...")
    predict()
    print("\n [INFO] It's almost over..")

    print("\n [INFO] Start live straming...")
    livestream()
    


