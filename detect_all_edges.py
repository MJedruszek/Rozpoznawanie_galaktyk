import numpy as np
import pandas as pd
import cv2  as cv
# size = 20
photo_df = pd.read_csv('gz2_hart16_cropped.csv', usecols=['dr7objid', 't04_spiral_a08_spiral_flag', 't04_spiral_a09_no_spiral_flag', 't08_odd_feature_a20_lens_or_arc_flag', 't08_odd_feature_a22_irregular_flag'])
size = photo_df.shape[0]

def use_canny(sigma, filename):
    # v = np.median(img)
    # # apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img = cv.filter2D(img, -1, np.ones((5,5),np.float32)/25)
    upper, thresh_im = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    upper -= 7
    # print("upper: " + str(upper))
    lower = sigma*upper
    edges_canny = cv.Canny(img, lower, upper)
    return edges_canny

def save_photo(name, type):
    filename1 = "images_gz2/images/" + str(name) + ".jpg"
    if(type):
        filename2 = "test_data/canny/" + str(name) + ".jpg"
    else:
        filename2 = "train_data/canny/" + str(name) + ".jpg"
    try:
        img = use_canny(0.01, filename1)
        cv.imwrite(filename2, img)
    except:
        print("Couldn't find file: " + filename1)
        print("Couldn't save file: " + filename2)


spirals = True
elipses = True
lens = True
irregular = True

for i in range(0,size):
    row = photo_df.iloc[i]
    if(row.iloc[1]==1 and row.iloc[3] == 0 and row.iloc[4]==0):
        save_photo(row.iloc[0], spirals)
        spirals = not spirals
    elif(row.iloc[2]==1 and row.iloc[3]==0 and row.iloc[4] == 0):
        save_photo(row.iloc[0], elipses)
        elipses = not elipses
    elif(row.iloc[2]==1 and row.iloc[3]==1 and row.iloc[4]==0):
        save_photo(row.iloc[0], lens)
        lens = not lens
    elif(row.iloc[4]==1):
        save_photo(row.iloc[0], irregular)
        irregular = not irregular
