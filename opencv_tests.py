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

# Otwieranie, wyświetlanie i zapisywanie obrazu

# Otwórz plik pod konkretną nazwą/ścieżką
img = cv.imread("images_gz2/images/10000.jpg")

# Czy plik istnieje?
if img is None:
    sys.exit("Could not read the image.")
 
# Wyświetl plik, poczekaj aż użytkownik kliknie
cv.imshow("Display window", img)
k = cv.waitKey(0)
 
# Zapisz, jeśli użytkownik kliknął "s"
if k == ord("s"):
    cv.imwrite("results/galaxy.png", img)