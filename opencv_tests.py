# Krok 1: testy OpenCV
# Potrzebne:
# +(potencjalnie) zmiana rozmiaru obrazu, niepotrzebne, każdy obraz ma 424x424
# +normalizacja (jeśli część zakresu jest nieużywana, użyjmy jej! automat robi za nas)
# -filtracja (typ do wyboru później, wiedzieć tylko jak to się robi)
# -progowanie (podobnie, typ do wyboru później, na razie tylko jak)
# -detekcja krawędzi (j.w.)

import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
#######################################################
# Otwieranie, wyświetlanie i zapisywanie obrazu       #
#######################################################

filename = "images_gz2/images/8500.jpg"
# Otwórz plik pod konkretną nazwą/ścieżką
img = cv.imread(filename)

# Czy plik istnieje?
if img is None:
    sys.exit("Could not read the image.")
 
# Wyświetl plik, poczekaj aż użytkownik kliknie
cv.imshow("All colors", img)
k = cv.waitKey(0)

# Aby odczytać coś od razu jako grayscale, dodać flagę cv.IMREAD_GRAYSCALE

#######################################################
# Zmiana wartości pikseli, właściwości obrazu         #
#######################################################

# pobierz i wyświetl wartości BGR danego piksela
# format [ B G R]
# dla czarnobiałych wychodzi tylko jasność piksela

px = img[100,100]
print( px )

# pobierz i wyświetl intensywność niebieskiego piksela

blue = img[100,100,0]
print(blue)

# # zmień wartość piksela na zadaną
# img[100,100] = [255,255,255]

# cv.imshow("Display window", img)
# k = cv.waitKey(0)
 
# # Zapisz, jeśli użytkownik kliknął "s"
# if k == ord("s"):
#     cv.imwrite("results/galaxy.png", img)

# ^^^ tak warto tylko jeden zmienić, do większych operacji raczej nie

# odbierz i pokaż wymiary obrazka (x,y, liczba kanałów)

#print(img.shape)

# skopiuj i wklej jakiś obszar w konkretne miejsce

piece = img[180:240, 180:240]
#img[100:160, 100:160] = piece

# pobierz konkretny kolor
# r = img[:,:,2]

# podziel obraz na trzy oddzielne kanały (wolniejsze od powyższego)
b,g,r = cv.split(img)

# cv.imshow("Only red", r)
# k = cv.waitKey(0)

# cv.imshow("Only green", g)
# k = cv.waitKey(0)

# cv.imshow("Only blue", b)
# k = cv.waitKey(0)

#######################################################
# Zmiana skali kolorów z BGR na skalę szarości        #
#######################################################

# Zmień ten obraz na skalę szarości
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# alternatywnie:
gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.imshow("Grayscale", gray)
k = cv.waitKey(0)

# Zapisz, jeśli użytkownik kliknął "s"
if k == ord("s"):
    cv.imwrite("results/galaxy.png", gray)

#######################################################
# Progowanie na różne sposoby                         #
#######################################################

# progowanie na chama: jedna wartość progu, wszystko traktowane równo
# cv.threshold(obraz_skala_szarosci, próg, max_value_piksela, flaga)
#Flaga: cv.THRESH_BINARY 
ret,th1 = cv.threshold(gray, 32, 256, cv.THRESH_BINARY)

# progowanie mądre: 
# cv.adapriveThreshold(obraz_skala_szarości, max_value, typ_adaptywny, \
# typ_progowania rozmiar sąsiedztwa, stała)
th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,5,2)
th3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,5,2)

titles = ['Original Image', 'Global Thresholding (v = 32)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [gray, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

#######################################################
# Funkcja do zrobienia histogramu i pobrania detali   #
#######################################################

# cv.calvHist(obraz, kanał, maska, max_wart, range)

hist = cv.calcHist([gray], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.plot(hist)
plt.title('Pixel Intensity Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')

# Puste i pełne kubełki jako czerwone kropki
plt.subplot(1, 2, 2)
empty_mask = (hist.flatten() == 0)
plt.plot(empty_mask, 'ro', markersize=2)
plt.title('Empty Bins (Red Dots)')
plt.xlabel('Bin Index')
plt.ylabel('Empty (1) / Non-empty (0)')
plt.yticks([0, 1])

plt.tight_layout()
plt.show()

#których konkretnie brakuje? Wychodzi na to, że powyżej 227 (włącznie) nie ma żadnych pixeli
missing_intensities = np.where(hist.flatten() == 0)[0]
print(f"Missing intensity values: {missing_intensities}")

#######################################################
# NORMALIZACJA OBRAZKA                                #
#######################################################

#bardziej na późniejszy check, niepotrzebne
max_intensity = np.max(gray)
min_intensity = np.min(gray)

#cv.normalize(obraz, drugi obraz, min, max, typ normalizacji)
normalized_gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)

cv.imshow("Normalized grayscale", normalized_gray)
k = cv.waitKey(0)