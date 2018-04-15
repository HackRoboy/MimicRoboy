import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import dlib
import os
import re
from models.mtcnn.align_dlib import AlignDlib
from models.mtcnn import detect_face
from scipy import misc
#from ros_publisher import WebsocketROSPublisher
import json
#Web= WebsocketROSPublisher('192.168.91.1',9091)
import socket

TCP_IP = '127.0.0.1' #192.168.91.1  127.0.0.1
TCP_PORT = 9091
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))


align = AlignDlib('models/dlib/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

def detect_face_dlib(img):
    bbs = detector(img, 1)
    tuples = []
    for r in bbs:
        tuples.append((r.left(), r.top(), r.right(), r.bottom()))
    return tuples

EXPECT_SIZE = 160
def align_face_dlib(image, face_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE):
    assert isinstance(face_box, tuple)
    face_rect = dlib.rectangle(*face_box)
    landmarks = align.findLandmarks(image, face_rect)
    alignedFace = align.align(EXPECT_SIZE, image, face_rect,
                              landmarks=landmarks,
                              landmarkIndices=landmarkIndices)
    return alignedFace, landmarks

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def detect_face_and_landmarks_mtcnn(img):
    img = img[:,:,0:3]
    bbs, lms = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    boxes = []
    landmarks = []
    face_index = 0
    for r in bbs:
        r = r.astype(int)
        points = []
        for i in range(5):
            points.append((lms[i][face_index] , lms[i+5][face_index]))
        landmarks.append(points)
        boxes.append((r[0] , r[1] , r[2] , r[3]))
        #boxes.append(r[:4].astype(int).tolist())
        face_index += 1
    return boxes, landmarks

EXPECT_SIZE = 160
def align_face_mtcnn(img, bb, landmarks):
    assert isinstance(bb, tuple)
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    scaled = misc.imresize(cropped, (EXPECT_SIZE, EXPECT_SIZE), interp='bilinear')
    return scaled


def draw_rects(image, rects):
    result = image.copy()
    for left, top, right, bottom in rects:
        cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)
    return result

def draw_landmarks(image, points):
    result = image.copy()
    for point in points:
        cv2.circle(result, point, 3, (0, 255, 0), -1 )
    return result


# as proof: https://pomax.github.io/bezierinfo/

from numpy import array, linalg, matrix
from scipy.misc import comb as nOk
Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
bezierM = lambda ts: matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])
def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points


def beziertransformation(a,b,c,d):
    V = array
    E, W, N, S = V(a), V(b), V(c), V(d)
    cw = 0.1
    ch = 0.1
    cpb = V(a)
    cpe = V(d)
    xys = [cpb, cpb + ch * N + E * cw / 8, cpe + ch * N + E * cw / 8, cpe]

    ts = V(range(11)) / 10
    M = bezierM(ts)
    points = M * xys  # produces the points on the bezier curve at t in ts
    return lsqfit(points, M)

def roboy_trans(mat,factor, c):
    scale = factor*mat
    moved = scale + c
    return moved

gesamt = []
for i in range (10):
    camera = cv2.VideoCapture(0)
    return_value,frame = camera.read()
    camera.release()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

   # plt.imshow(img)
    bbs, lm = detect_face_and_landmarks_mtcnn(img)

    aligned_face, lm = align_face_dlib(img, bbs[0], AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    bbs, lm = detect_face_and_landmarks_mtcnn(aligned_face)
    aligned_face, lm = align_face_dlib(aligned_face, bbs[0], AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    k = lm[49:55]
    k = np.array(k)
    print(k)
    upper_bezier_points = beziertransformation(lm[49],lm[51], lm[53], lm[55])
    lower_bezier_points = beziertransformation(lm[49],lm[59], lm[57], lm[55])
    plt.imshow(draw_landmarks(aligned_face, lm[49:61]))
    plt.plot(upper_bezier_points[:,0],upper_bezier_points[:,1],'ro')
    move = roboy_trans(upper_bezier_points, 0.5, -30)
    s.send(upper_bezier_points, lower_bezier_points)
    #Web.publish('roboy.communication_middleware/Trajectory')
    plt.show()


s.close()
#    gesamt.append(lm)
# with open('landmarks.csv','w') as csvfile:
#     writer = csv.writer(csvfile)
#     for i in gesamt:
#         writer.writerow(i)