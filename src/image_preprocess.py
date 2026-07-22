import os
from PIL import Image, ImageFile
import glob
import seaborn as sns 
import imghdr 
import cv2 

img_width = 300
img_height = 300 

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


#Resize all images to 192 by 192, grayscale 
train_path = "dataset/train"
test_path = "dataset/test"

save_train_path = "preprocessed_dataset/train"
save_test_path = "preprocessed_dataset/test"

sorted_list = ["real", "fake"]
for path in sorted_list:
    image_directory_path = os.path.join(train_path, path)
    save_image_directory_path = os.path.join(save_train_path, path)
    print("preprocessing " + path)
    for i in range(24000):
        img_name = str(i+1)
        
        if(i+1 < 1000):
            img_name = "0" + img_name
        if(i+1 < 100):
            img_name = "0" + img_name
        if(i+1 < 10):
            img_name = "0" + img_name 
        
        img_name = img_name
        
        #Skip if the the image was already preprocessed in the directory
        if os.path.exists(os.path.join(save_image_directory_path, img_name + ".jpg")):
            continue
        
        #Find out what type the image is
        if os.path.exists(os.path.join(image_directory_path, img_name + ".jpg")):
            image_type = ".jpg"
            
        elif os.path.exists(os.path.join(image_directory_path, img_name + ".png")):
            image_type = ".png"
        
        else:
            continue
        
        img = Image.open(os.path.join(image_directory_path, img_name + image_type))
        
        new_img = img.convert("RGB")
        new_img = new_img.resize((300,300))
        
        new_img.save(save_image_directory_path + "/" + img_name + ".jpg", "JPEG")

for path in sorted_list:
    image_directory_path = os.path.join(test_path, path)
    save_image_directory_path = os.path.join(save_test_path, path)
    for i in range(6000):
        img_name = str(i+1)
        
        if(i+1 < 1000):
            img_name = "0" + img_name
        if(i+1 < 100):
            img_name = "0" + img_name
        if(i+1 < 10):
            img_name = "0" + img_name
        
        
        #Skip if the the image was already preprocessed in the directory
        if os.path.exists(os.path.join(save_image_directory_path, img_name + ".jpg")):
            continue
        
        #Find out what type the image is
        if os.path.exists(os.path.join(image_directory_path, img_name + ".jpg")):
            image_type = ".jpg"
            
        elif os.path.exists(os.path.join(image_directory_path, img_name + ".png")):
            image_type = ".png"
        
        else:
            continue
        
        
        
        img = Image.open(os.path.join(image_directory_path, img_name + image_type))
        
        new_img = img.convert("RGB")
        new_img = new_img.resize((300,300))
        
        new_img.save(save_image_directory_path + "/" + img_name + ".jpg", "JPEG")
        