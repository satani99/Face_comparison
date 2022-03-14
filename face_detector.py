import cv2
import os, sys
import time
import scipy
import numpy as np
from deepface import DeepFace
from scipy import spatial

from face_embedding import FaceEmbedding

trained_face_data = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_1 = cv2.imread('ae_1.jpg')
img_2 = cv2.imread('ae_2.jpg')

face_1 = trained_face_data.detectMultiScale(img_1)
face_2 = trained_face_data.detectMultiScale(img_2)

face_a = face_1.tolist()
face_b = face_2.tolist()

def getting_face_co(list):
    list_w = {}
    for i in range(len(list)):
        list_w[i] = list[i][-1]
    max_w = max(list_w.values())
    i = [key for key in list_w.keys()][[value for value in list_w.values()].index(max_w)]
    x, y, w, h = list[i]
    return x, y, w, h

x_1, y_1, w_1, h_1 = getting_face_co(face_a)
x_2, y_2, w_2, h_2 = getting_face_co(face_b)

cv2.rectangle(img_1, (x_1, y_1), (x_1+w_1, y_1+h_1), (0, 255, 0), 10)
cv2.rectangle(img_2, (x_2, y_2), (x_2+w_2, y_2+h_2), (0, 255, 0), 10)

cropped_image_1 = img_1[y_1:y_1+h_1, x_1:x_1+w_1]
cropped_image_2 = img_2[y_2:y_2+h_2, x_2:x_2+w_2]

cv2.imwrite('Cropped_ae_1.png', cropped_image_1)
cv2.imwrite('Cropped_ae_2.png', cropped_image_2)

def main():
    print("In main")
    idFace = cv2.imread('Cropped_ae_1.png')
    selfie_face = cv2.imread('Cropped_ae_2.png')

    embeddingModel = "openface_nn4.small2.v1.t7"

    faceEmbeddingVec = FaceEmbedding(idFace, embeddingModel)
    embeddingVectorId = faceEmbeddingVec.get_face_embedding()

    faceEmbeddingVec = FaceEmbedding(selfie_face, embeddingModel)
    embeddingVectorSelfie = faceEmbeddingVec.get_face_embedding()

    similarity_dist = spatial.distance.cosine(embeddingVectorId, embeddingVectorSelfie)
    print("Similarity: ", (1 - similarity_dist)*100, '%')

    obj = DeepFace.analyze(img_path='Cropped_ae_2.png', actions=['gender'])
    print(obj['gender'])
    
if __name__ == '__main__':
    main()
    




