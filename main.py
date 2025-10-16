import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

from RCF.models import RCF
from CATS.models import Network


filename = "images_gz2/images/8600.jpg"
img_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
#ten sam, ale kolorowy dla heda
img_color = cv.imread(filename)

# Zwykły detektor Canny
edges_canny = cv.Canny(img_gray, 20,100)

#HED
W = 424
H = 424

#stwórz sieć HED
net_hed = cv.dnn.readNetFromCaffe("HED/deploy.prototxt", "HED/hed_pretrained_bsds.caffemodel")

#stwórz blob ze zdjęcia
blob = cv.dnn.blobFromImage(img_color, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
#przetwórz i zwróć
net_hed.setInput(blob)
hed_raw = net_hed.forward()
hed_raw = cv.resize(hed_raw[0, 0], (W, H))
hed_raw = (255 * hed_raw).astype("uint8")
hed = cv.normalize(hed_raw, None, 0, 255, cv.NORM_MINMAX)


#RCF

def load_rcf_model():
    # Inicjalizacja modelu
    model = RCF()
    
    # załaduj strukturę i wagi
    checkpoint = torch.load("RCF/bsds500_pascal_model.pth", map_location='cpu')
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

#Właściwa detekcja
def rcf_edge_detection(image_path):
    # Załaduj model

    model = load_rcf_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #pobierz obraz w formacie rgb (CV robi to w formacie BGR)
    image = Image.open(image_path).convert('RGB')
    original_size = image.size[::-1]  # obróć obraz
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Zmień rozmiar
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Pobieramy tylko ostatnie z wyjść
        if isinstance(outputs, (list, tuple)):
            edge_map = outputs[-1] 
        else:
            edge_map = outputs
        
        edge_map = torch.sigmoid(edge_map[0, 0]).cpu().numpy()
    
    # Zwróć formę w tym samym rozmiarze, co oryginał
    edge_map = cv.resize(edge_map, (original_size[1], original_size[0]))
    edge_map = (edge_map * 255).astype(np.uint8)
    
    return edge_map

rcf_raw = rcf_edge_detection(filename)

rcf = cv.normalize(rcf_raw, None, 0, 255, cv.NORM_MINMAX)


#cats

#Konfiguracja dla CATS
class Config:
    def __init__(self):
        self.pretrained = "CATS/vgg16.pth"  #Ścieżka do wytrenowanego modelu
        self.resume = "CATS/bsds.pth"  #Ścieżka do wag
        self.gpu = False  
        self.num_classes = 1

def load_cats_model(pth_path):
    
    config = Config()
    config.resume = pth_path
    
    # Inicjalizacja modelu wraz z wagami
    model = Network(config)
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

#Właściwa detekcja krawędzi
def cats_edge_detection(model):
    device = torch.device('cpu')
    model.to(device)
    
    # Załaduj obrazek
    image = Image.open(filename).convert('RGB')
    
    # Znormalizuj go
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Używamy tylko ostatniego wyniku
        if isinstance(outputs, list) and len(outputs) > 0:
            fuse_output = outputs[-1]
            edge_map = fuse_output[0, 0].cpu().numpy()
        else:
            edge_map = outputs[0, 0].cpu().numpy()
    
    edge_map = (edge_map * 255).astype(np.uint8)
    
    return edge_map

cats_model = load_cats_model("CATS/bsds.pth")

cats_raw = cats_edge_detection(cats_model)
cats = cv.normalize(cats_raw, None, 0, 255, cv.NORM_MINMAX)

#Wyświetl wyniki, najpierw oryginał i trzy zaawansowane, później Canny oraz sprogowane trzy zaawansowane
cv.imshow("Input", img_gray)

cv.imshow("HED", hed)
cv.imshow("RCF", rcf)
cv.imshow("CATS", cats)
cv.waitKey(0)

_, hed_binary = cv.threshold(hed, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
_, rcf_binary = cv.threshold(rcf, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
_, cats_binary = cv.threshold(cats, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("HED binary", hed_binary)
cv.imshow("RCF binary", rcf_binary)
cv.imshow("CATS binary", cats_binary)
cv.imshow("Canny", edges_canny)
cv.waitKey(0)