import pickle
import os


folder_path = './../cylinder_data/2clss_25rect_65kpts_5rect/pcd_adjusted/'

train_fileames = []

for filename in os.listdir(folder_path):
    if filename.endswith('.pcd'):
        filepath = os.path.join(folder_path, filename)
        train_fileames.append(filepath)

print(len(train_fileames))
with open('./train_filenames_25.pkl', 'wb') as file:
    pickle.dump(train_fileames, file)

print('done')