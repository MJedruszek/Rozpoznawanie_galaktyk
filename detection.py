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

# filename = "test.png"
filename = "images_gz2/images/2137.jpg"
img_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
img_gray = cv.normalize(img_gray, None, 0, 255, cv.NORM_MINMAX)
img_gray = cv.filter2D(img_gray, -1, np.ones((5,5),np.float32)/25)
#ten sam, ale kolorowy dla heda
img_color = cv.imread(filename)
img_color = cv.filter2D(img_color, -1, np.ones((5,5),np.float32)/25)

# Zwykły detektor Canny
def use_canny(sigma, img):
    # v = np.median(img)
    # # apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    upper, thresh_im = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lower = sigma*upper
    edges_canny = cv.Canny(img, lower, upper)
    return edges_canny

#HED

def init_hed():
    net_hed = cv.dnn.readNetFromCaffe("HED/deploy.prototxt", "HED/hed_pretrained_bsds.caffemodel")
    # Pobierz nazwy wszystkich warstw, potrzebne do wybrania konkretnej
    layer_names = net_hed.getLayerNames()
    try:
        # OpenCV 4.x
        output_layers = [layer_names[i - 1] for i in net_hed.getUnconnectedOutLayers()]
    except:
        # OpenCV 3.x
        output_layers = [layer_names[i[0] - 1] for i in net_hed.getUnconnectedOutLayers()]
    
    return net_hed, output_layers

def use_hed(net, layers, img):
    H, W, channels = img.shape
    img_resized = cv.resize(img, (512, 512))
    blob = cv.dnn.blobFromImage(img_resized, scalefactor=1.0, size=(512, 512),mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
    net.setInput(blob)

    hed_outputs = net.forward(layers)

    # Przetwórz każdy output
    hed_return = []
    for i, output in enumerate(hed_outputs):    
        # Zmień rozmiar
        hed_raw = output[0,0]  # Get the first channel of first image in batch
        hed_raw = cv.resize(hed_raw, (W, H), interpolation=cv.INTER_LINEAR)
        hed_raw = (255 * hed_raw).astype("uint8")
        hed_normalized = cv.normalize(hed_raw, None, 0, 255, cv.NORM_MINMAX)
        
        hed_return.append(hed_normalized)

    return hed_return

#stwórz sieć HED
# net_hed = cv.dnn.readNetFromCaffe("HED/deploy.prototxt", "HED/hed_pretrained_bsds.caffemodel")
# #stwórz blob ze zdjęcia
# blob = cv.dnn.blobFromImage(img_color, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
# #przetwórz i zwróć
# net_hed.setInput(blob)
# hed_raw = net_hed.forward()
# hed_raw = cv.resize(hed_raw[0, 0], (W, H))
# hed_raw = (255 * hed_raw).astype("uint8")
# hed = cv.normalize(hed_raw, None, 0, 255, cv.NORM_MINMAX)

#RCF

def init_rcf():
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
def use_rcf(model, img_name):
    # Załaduj model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #pobierz obraz w formacie rgb (CV robi to w formacie BGR)
    image = Image.open(img_name).convert('RGB')
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
        
        # Pobieramy tylko najlepsze z wyjść
        if isinstance(outputs, (list, tuple)):
            edge_map = outputs[1] 
        else:
            edge_map = outputs
        
        edge_map = torch.sigmoid(edge_map[0, 0]).cpu().numpy()
    
    # Zwróć formę w tym samym rozmiarze, co oryginał
    edge_map = cv.resize(edge_map, (original_size[1], original_size[0]))
    edge_map = (edge_map * 255).astype(np.uint8)
    edge_map = cv.normalize(edge_map, None, 0, 255, cv.NORM_MINMAX)
    
    return edge_map


#cats

#Konfiguracja dla CATS
class Config:
    def __init__(self):
        self.pretrained = "CATS/vgg16.pth"  #Ścieżka do wytrenowanego modelu
        self.resume = "CATS/bsds.pth"  #Ścieżka do wag
        self.gpu = False  
        self.num_classes = 1

def init_cats():
    
    config = Config()
    
    # Inicjalizacja modelu wraz z wagami
    model = Network(config)
    checkpoint = torch.load("CATS/bsds.pth", map_location='cpu')
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

#Właściwa detekcja krawędzi
def use_cats(model, img_name):
    device = torch.device('cpu')
    model.to(device)
    
    # Załaduj obrazek
    image = Image.open(img_name).convert('RGB')
    
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
            fuse_output = outputs[0]
            edge_map = fuse_output[0, 0].cpu().numpy()
        else:
            edge_map = outputs[0, 0].cpu().numpy()
    
    edge_map = (edge_map * 255).astype(np.uint8)
    edge_map = cv.normalize(edge_map, None, 0, 255, cv.NORM_MINMAX)
    return edge_map

# hed_model, hed_layers = init_hed()
# rcf_model = init_rcf()
# cats_model = init_cats()

# hed = []
# hed = use_hed(hed_model, hed_layers, img_color)
# rcf = use_rcf(rcf_model, filename)
# cats = use_cats(cats_model, filename)
canny = use_canny(0.5, img_gray)

#Wyświetl wyniki, najpierw oryginał i trzy zaawansowane, później Canny oraz sprogowane trzy zaawansowane
cv.imshow("Input", img_gray)

# cv.imshow("HED 0", hed[0])
# cv.imshow("HED 1", hed[1])
# cv.imshow("HED 2", hed[2])
# cv.imshow("HED 3", hed[3])
# cv.imshow("HED 4", hed[4])
# cv.imshow("HED 5", hed[5])

# cv.imshow("RCF", rcf)
# cv.imshow("CATS", cats)
# cv.waitKey(0)

# _, hed_binary_0 = cv.threshold(hed[0], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# _, hed_binary_1 = cv.threshold(hed[1], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# _, hed_binary_2 = cv.threshold(hed[2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# _, hed_binary_3 = cv.threshold(hed[3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# _, hed_binary_4 = cv.threshold(hed[4], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# _, hed_binary_5 = cv.threshold(hed[5], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# _, rcf_binary = cv.threshold(rcf, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# _, cats_binary = cv.threshold(cats, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow("HED binary 0", hed_binary_0)
# cv.imshow("HED binary 1", hed_binary_1)
# cv.imshow("HED binary 2", hed_binary_2)
# cv.imshow("HED binary 3", hed_binary_3)
# cv.imshow("HED binary 4", hed_binary_4)
# cv.imshow("HED binary 5", hed_binary_5)

# cv.imshow("RCF binary", rcf_binary)
# cv.imshow("CATS binary", cats_binary)
cv.imshow("Canny", canny)
cv.waitKey(0)

# cv.imwrite("results/HED0.png", hed_binary_0)
# cv.imwrite("results/HED1.png", hed_binary_1)
# cv.imwrite("results/HED2.png", hed_binary_2)
# cv.imwrite("results/HED3.png", hed_binary_3)
# cv.imwrite("results/HED4.png", hed_binary_4)
# cv.imwrite("results/HED5.png", hed_binary_5)
# cv.imwrite("results/RCF5.png", rcf_binary)
# cv.imwrite("results/CATS5.png", cats_binary)
# cv.imwrite("results/CANNY.png", edges_canny)



