import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

#Krok 0: wczytaj obraz w skali szarości

filename = "images_gz2/images/8500.jpg"
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

#Krok 1: Normalizacja
norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

#Krok 2: Filtracja (póki co filtr bilateral o losowych parametrach XD)

filtered = cv.bilateralFilter(norm,7,75,75)

#Krok 3: Progowanie

ret, thresh = cv.threshold(filtered, 16, 256, cv.THRESH_BINARY)

#i druga opcja, progowanie adaptacyjne
thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,3,2)

#Krok 3.5: Wyświetl kolejne kroki od 0 do 3 w celu sprawdzenia poprawności wykonania (opcjonalne)

titles = ['Original Image', 'After normalization',
            'After filtering', 'After thresholding']
images = [img, norm, filtered, thresh]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

#Krok 4: Detekcja krawędzi 

# Głupi Canny
edges_canny = cv.Canny(img, 20,100)

cv.imshow("Canny",edges_canny)
k = cv.waitKey(0)

#HED
W,H = 424
blob = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)

#JUTRO:
#POBIERZ POTRZEBNE PLIKI Z NETA
#PRZETESTUJ WSZYSTKIE TRZY DETEKTORY, KTÓRE POWINNAŚ
net = cv.dnn.readNetFromCaffe("path to prototxt file", "path to model weights file")