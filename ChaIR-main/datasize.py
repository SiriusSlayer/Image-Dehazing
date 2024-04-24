import cv2
import os
data_path='new/hazy'
file_list=os.listdir(data_path)
for file in file_list:
    img_path = os.path.join(data_path, file)
    img = cv2.imread(img_path)
    print(f"Shape of {file}: {img.shape}")
# print(file_list)