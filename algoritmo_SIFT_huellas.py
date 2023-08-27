# -*- coding: utf-8 -*-
"""
Creado el domingo :)

@author: RSS
"""

import cv2
import numpy as np
import os

test_original = cv2.imread("huella_desconocida_b.tif")
cv2.imshow("Huella persona desconocida", cv2.resize(test_original, None, fx=1, fy=1))


for file in [file for file in os.listdir("base_de_huellas_conocidas")]:
    
    fingerprint_database_image = cv2.imread("./base_de_huellas_conocidas/"+file)
    
    sift = cv2.xfeatures2d.SIFT_create()  # Creación de instancia de SIFT
    
    #Descriptores de la huella desconocida y la imagen conocida 
    keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)
    
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), #Coincidencias
          dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    
    match_points = [] #Lista de coincidencias

    for p, q in matches:
        if p.distance < 0.1*q.distance:
            match_points.append(p)

        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
           keypoints = len(keypoints_1)            
        else:
           keypoints = len(keypoints_2)
         #porcentaje de coincidencia, pueden variarlo a su consideranción
        if (len(match_points) / keypoints)>0.95:
            
           print("% Coincidencia: ", len(match_points) / keypoints * 100) 
           print("ID de la huella desconocida: " + str(file)) 
           #Comparación y dibujado de coincidencias en las 2 huellas
           result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, 
                                    keypoints_2, match_points, None) 
           result = cv2.resize(result, None, fx=1.0, fy=1.0)
           cv2.imshow("Resultados de coincidencias", result)
           cv2.waitKey(0)
           cv2.destroyAllWindows()
           break



