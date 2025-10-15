import pandas as pd
import io

# size = 10000
hart16 = pd.read_csv('gz2_hart16.csv', usecols=['dr7objid', 't04_spiral_a08_spiral_flag', 't04_spiral_a09_no_spiral_flag', 't08_odd_feature_a20_lens_or_arc_flag', 't08_odd_feature_a22_irregular_flag']).dropna()
size = hart16.shape[0]
# print(size)

row = hart16.iloc[0]
new_hart16 = pd.DataFrame(columns=['dr7objid', 't04_spiral_a08_spiral_flag', 't04_spiral_a09_no_spiral_flag', 't08_odd_feature_a20_lens_or_arc_flag', 't08_odd_feature_a22_irregular_flag'])
name_map = pd.read_csv('gz2_filename_mapping.csv', usecols=['objid', 'asset_id']).dropna()

spirals = 0
elipses = 0
lens = 0
irregular = 0
none = 0

#naszym celem jest około 4k spiralnych i eliptycznych oraz około 3k nieregularnych i wszystkie soczewkowate

for i in range(0,size):
    row = hart16.iloc[i]
    if(row.iloc[1]==1 and row.iloc[3] == 0 and row.iloc[4]==0):
        spirals += 1
        if spirals % 25 == 0:
            new_hart16.loc[len(new_hart16)] = row
    elif(row.iloc[2]==1 and row.iloc[3]==0 and row.iloc[4] == 0):
        elipses += 1
        if elipses % 22 == 0:
            new_hart16.loc[len(new_hart16)] = row
    elif(row.iloc[2]==1 and row.iloc[3]==1 and row.iloc[4]==0):
        lens+=1
        new_hart16.loc[len(new_hart16)] = row
    elif(row.iloc[4]==1):
        irregular+=1
        if irregular % 2 == 0:
            new_hart16.loc[len(new_hart16)] = row
    else:
        none += 1
        

print("Spirals in sample: " + str(spirals))
print("Elipses in sample: " + str(elipses))
print("Lenticulars in sample: " + str(lens))
print("Irregulars in sample: " + str(irregular))
print("Other types in sample: " + str(none))

merged = new_hart16.merge(name_map,
                          left_on='dr7objid',
                          right_on='objid',
                          how='left')

merged['dr7objid']=merged['asset_id']
merged = merged.drop(['asset_id', 'objid'], axis=1)
new_hart16 = merged



new_hart16.to_csv('gz2_hart16_cropped.csv')

