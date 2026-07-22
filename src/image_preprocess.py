import os
from PIL import Image, ImageFile
import glob
import seaborn as sns 
import imghdr 
import cv2 
from tqdm import tqdm
import numpy as np
import pandas as pd 


directory = "preprocessed_dataset"
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Make preprocessed directory
if(not os.path.exists(directory)):
    mode = 0o66666
    os.mkdir(directory,mode)
    
    train_directory = "train"
    test_directory = "test"
    
    train_path = os.path.join(directory, train_directory)
    test_path = os.path.join(directory, test_directory)
    
    os.mkdir(train_path, mode)
    os.mkdir(test_path,mode)
    
    fake = "fake"
    real = "real"
    
    train_fake_path = os.path.join(train_path, fake)
    train_real_path = os.path.join(train_path, real)
    test_fake_path = os.path.join(test_path, fake)
    test_real_path = os.path.join(test_path, real)
    
    os.mkdir(train_fake_path, mode)
    os.mkdir(train_real_path, mode)
    os.mkdir(test_fake_path,mode)
    os.mkdir(test_real_path, mode)


#Resize all images to 300 by 300
train_path = "dataset/train"
test_path = "dataset/test"

save_train_path = "preprocessed_dataset/train"
save_test_path = "preprocessed_dataset/test"




def image_preprocessing(input_path: str, output_path: str, csv_name: str, ):
    
    img_width = 600
    img_height = 600 
    
    for class_name in os.listdir(train_path):
        image_directory_path = os.path.join(input_path, class_name)
        save_image_directory_path = os.path.join(output_path, class_name)
        
        for img_file in os.listdir(image_directory_path):
            
            
            #Skip if the the image was already preprocessed in the directory
            if os.path.exists(os.path.join(save_image_directory_path, img_file)):
                continue
            
            img = Image.open(os.path.join(image_directory_path, img_file))
            if img is None:
                continue
            
            new_img = img.convert("L")
            new_img = new_img.resize((img_width, img_height))
            
            new_img.save(os.path.join(save_image_directory_path, img_file))


image_preprocessing(input_path = train_path, output_path=save_train_path, csv_name="training_data.csv")
image_preprocessing(input_path = test_path, output_path=save_test_path, csv_name="testing_data.csv")