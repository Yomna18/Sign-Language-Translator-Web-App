from flask import Flask, render_template, Response
import time
import os
import jyserver.Flask as jsf
import cv2
import mediapipe as mp
import math
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop

# Class hands
class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        #blank_image = np.zeros((1080,1080,3), np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
#                     cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
#                                   (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
#                                   (255, 0, 255), 2)
#                     cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
#                                 2, (255, 0, 255), 2)
        if draw:
            return allHands,img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info



#Camera file
train = ImageDataGenerator(rescale = 1./255
)
validation = ImageDataGenerator(rescale = 1./255)
train_dataset = train.flow_from_directory('./data/trainSet',
                                         target_size = (350,350),
                                         batch_size = 32,
                                         class_mode = "categorical")
validation_dataset = validation.flow_from_directory('./data/validSet',
                                                   target_size = (350,350),
                                                   batch_size = 32,
                                                   class_mode = "categorical")
train_dataset.class_indices
dic = list(train_dataset.class_indices)


offset = 10
imgSize = 350
frame_rate = 5
counter = 0
result = 0
detector = HandDetector(maxHands=1)
model = load_model("./models/VGG16_Augmented1.h5")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    
    def get_frame_original(self):
        ret, img = self.video.read()
        ret, jpeg = cv2.imencode('.jpg',img)
        return jpeg.tobytes()

    def get_frame(self, time_elapsed):
        ret, img = self.video.read()
        imgOutput = img.copy()

        #Here we will manipulate the frame to pass it to the model
        hands, img = detector.findHands(img)

        if hands:

            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            imgWhiteCopy = imgWhite.copy()
            imgWhite = img_to_array(imgWhite)
            imgWhite = imgWhite.reshape((1, imgWhite.shape[0], imgWhite.shape[1], imgWhite.shape[2]))
            imgWhite = preprocess_input(imgWhite)
            if time_elapsed > 1./frame_rate:
                prev = time.time()
                result = model.predict(imgWhite)
                word = dic[result.argmax()]
                oldWord = word
                result = result * 100000000
                maxVal = (result[0].max()/sum(result[0])) * 100
                if(maxVal > 99.9999999):
                    cv2.putText(imgOutput, dic[result.argmax()],(x,y-20), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                    # cv2.imshow(f"ImageCrop", imgCrop)
                else:
                    cv2.putText(imgOutput, "",(x,y-20), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
                    # cv2.imshow(f"ImageCrop", imgCrop)
            else:
                cv2.putText(imgOutput, word,(x,y-20), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

            result = model.predict(imgWhite)

            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)
            

            cv2.putText(imgOutput, dic[result.argmax()],(x,y-20), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)           
            # cv2.imshow(f"ImageCrop", imgCrop)
            # cv2.imshow(f"imgWhite", imgWhiteCopy)


        ret, jpeg = cv2.imencode('.jpg',imgOutput)
        return jpeg.tobytes()


app = Flask(__name__)
prev = 0



@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        time_elapsed = time.time() - prev
        try:
            frame = camera.get_frame(time_elapsed)
            yield (b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n\r\n' + frame 
                + b'\r\n\r\n')
        except:
            frame = camera.get_frame_original()
            yield (b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n\r\n' + frame 
                + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__== '__main__':
    app.run(debug=True)