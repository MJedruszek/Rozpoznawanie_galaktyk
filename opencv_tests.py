# Krok 1: testy OpenCV
# Potrzebne:
# -(potencjalnie) zmiana rozmiaru obrazu
# -normalizacja
# -filtracja (typ do wyboru później, wiedzieć tylko jak to się robi)
# -progowanie (podobnie, typ do wyboru później, na razie tylko jak)
# -detekcja krawędzi (j.w.)

import numpy as np
import cv2 as cv
import sys
#######################################################
# Otwieranie, wyświetlanie i zapisywanie obrazu       #
#######################################################

# Otwórz plik pod konkretną nazwą/ścieżką
img = cv.imread("images_gz2/images/8500.jpg")

# Czy plik istnieje?
if img is None:
    sys.exit("Could not read the image.")
 
# Wyświetl plik, poczekaj aż użytkownik kliknie
cv.imshow("All colors", img)
k = cv.waitKey(0)


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

cv.imshow("Only red", r)
k = cv.waitKey(0)

cv.imshow("Only green", g)
k = cv.waitKey(0)

cv.imshow("Only blue", b)
k = cv.waitKey(0)

#######################################################
# Zmiana skali kolorów z BGR na skalę szarości        #
#######################################################

# Zmień ten obraz na skalę szarości
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("Grayscale", gray)
k = cv.waitKey(0)

# Zapisz, jeśli użytkownik kliknął "s"
if k == ord("s"):
    cv.imwrite("results/galaxy.png", gray)