import numpy as np
import pandas as pd

preds = np.load('data/attribute_classifier/white_black_asian_hispanic_other.npy')
preds = pd.DataFrame(np.argmax(preds, axis = 1), columns=['class'])
preds.index += 1



id_map = pd.read_csv('data/attribute_classifier/identity_filtered.txt', header=None, sep=' ', names=['name', 'id']).sort_values('id')
id_map = id_map.join(preds, on='id')
# print(ee.shape)
# ee = ee.loc[~ee.id.isin(remove_id)]
# print(ee.shape)

for i, race in enumerate(['white', 'black', 'asian', 'hispanic']):
    np.savetxt(f'../recognition/data/attribute_classifier/datasets/race_v2/{race}_list.txt', id_map[id_map['class'] == i].name, fmt='%s')
    # id_map['class'].value_counts()
    
    
# for i in range(4):
#     print(i)
#     n = id_map[id_map['class'] == i].name.values
#     for j in range(30):
#         plt.imshow(plt.imread(f'/sdh/data/celebA/all_imgs/{n[j]}'))
#         plt.show()
#         