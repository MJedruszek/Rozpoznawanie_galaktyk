import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

from RCF.models import RCF
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

net_hed = cv.dnn.readNetFromCaffe("HED/deploy.prototxt", "HED/hed_pretrained_bsds.caffemodel")

net_hed.setInput(blob)
hed_raw = net_hed.forward()
hed_raw = cv.resize(hed_raw[0, 0], (W, H))
hed_raw = (255 * hed_raw).astype("uint8")

hed = cv.normalize(hed_raw, None, 0, 255, cv.NORM_MINMAX)
#RCF

def load_rcf_model():
    # Initialize model
    model = RCF()
    
    # Load weights - .pth file should be in the RCF folder
    checkpoint = torch.load("RCF/bsds500_pascal_model.pth", map_location='cpu')
    
    # Load weights based on checkpoint structure
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def rcf_edge_detection(image_path):
    # Load model
    model = load_rcf_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size[::-1]  # (width, height) -> (height, width)
    
    # Preprocessing - check the repository for exact requirements
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Common size for RCF
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # RCF usually returns multiple side outputs - take the final one
        if isinstance(outputs, (list, tuple)):
            edge_map = outputs[-1]  # Final output
        else:
            edge_map = outputs
        
        # Apply sigmoid and convert to numpy
        edge_map = torch.sigmoid(edge_map[0, 0]).cpu().numpy()
    
    # Resize back to original dimensions
    edge_map = cv.resize(edge_map, (original_size[1], original_size[0]))
    
    # Convert to 0-255
    edge_map = (edge_map * 255).astype(np.uint8)
    
    return edge_map

rcf_raw = rcf_edge_detection(filename)

rcf = cv.normalize(rcf_raw, None, 0, 255, cv.NORM_MINMAX)

cv.imshow("Input", img_gray)
cv.imshow("HED", hed)
cv.imshow("Canny", edges_canny)
cv.imshow("RCF", rcf)
cv.waitKey(0)