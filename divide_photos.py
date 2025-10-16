import pandas as pd
import cv2 as cv

#funkcja wczytuje obraz o nazwie "name" z folderu ze zdjęciami, a następnie zapisuje go w odpowiednim
#folderze, dla type=1 do testów, dla type=0 do train
def save_photo(name, type):
    filename1 = "images_gz2/images/" + str(name) + ".jpg"
    if(type):
        filename2 = "test_data/originals/" + str(name) + ".jpg"
    else:
        filename2 = "train_data/originals/" + str(name) + ".jpg"
    try:
        img = cv.imread(filename1)
        cv.imwrite(filename2, img)
    except:
        print("Couldn't find file: " + filename1)
        print("Couldn't save file: " + filename2)

#pobierz bazę informacji o zdjęciach, które nas interesują
# size = 20

photo_df = pd.read_csv('gz2_hart16_cropped.csv', usecols=['dr7objid', 't04_spiral_a08_spiral_flag', 't04_spiral_a09_no_spiral_flag', 't08_odd_feature_a20_lens_or_arc_flag', 't08_odd_feature_a22_irregular_flag'])
size = photo_df.shape[0]

spirals = True
elipses = True
lens = True
irregular = True

#dla każdego ze zdjęć, pobierz i zapisz w odpowiednim folderze (naprzemiennie do train i test)

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

print("done")