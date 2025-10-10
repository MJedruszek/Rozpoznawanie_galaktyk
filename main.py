import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

#Krok 0: wczytaj obraz w skali szarości

filename = "images_gz2/images/8500.jpg"
img_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
#ten sam, ale kolorowy dla heda
img_color = cv.imread(filename)

#Krok 1: Normalizacja
# norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

#Krok 2: Filtracja (póki co filtr bilateral o losowych parametrach XD)

# filtered = cv.bilateralFilter(norm,7,75,75)

# #Krok 3: Progowanie

# ret, thresh = cv.threshold(filtered, 16, 256, cv.THRESH_BINARY)

#i druga opcja, progowanie adaptacyjne
# thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,3,2)

#Krok 3.5: Wyświetl kolejne kroki od 0 do 3 w celu sprawdzenia poprawności wykonania (opcjonalne)

# titles = ['Original Image', 'After normalization',
#             'After filtering', 'After thresholding']
# images = [img, norm, filtered, thresh]

# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

#Krok 4: Detekcja krawędzi 

# Głupi Canny
edges_canny = cv.Canny(img_gray, 20,100)

#HED
W = 424
H = 424
blob = cv.dnn.blobFromImage(img_color, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)

net = cv.dnn.readNetFromCaffe("HED/deploy.prototxt", "HED/hed_pretrained_bsds.caffemodel")

net.setInput(blob)
hed = net.forward()
hed = cv.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")
cv.imshow("Input", img_gray)
cv.imshow("HED", hed)
cv.imshow("Canny", edges_canny)
cv.waitKey(0)