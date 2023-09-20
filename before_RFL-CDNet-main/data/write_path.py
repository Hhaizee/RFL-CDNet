import os
from PIL import Image
import random

file_train = open('../train.txt',"w")
path_train = '/home/hehaibin/ch/before_RFL-CDNet-main/samples/'
path_train_list = sorted(os.listdir(path_train + 'A/'))
random.shuffle(path_train_list)
for file_A in path_train_list:
    path_file_A = path_train+"A/"+file_A
    path_file_B = path_train+"B/"+file_A
    path_file_label = path_train+"OUT/"+file_A
    # img = Image.open(path_file_label)
    # if not img.getbbox():
    #     print(path_file_label)
    #     continue
    file_train.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
file_train.close()
print("ok_train")

file_val = open('../val.txt',"w")
path_val = '/home/hehaibin/ch/before_RFL-CDNet-main/samples/'
path_val_list = sorted(os.listdir(path_val + 'A/'))
random.shuffle(path_val_list)
for file_A in path_val_list:
    path_file_A = path_val+"A/"+file_A
    path_file_B = path_val+"B/"+file_A
    path_file_label = path_val+"OUT/"+file_A
    # img = Image.open(path_file_label)
    # if not img.getbbox():
    #     print(path_file_label)
    #     continue
    file_val.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
file_val.close()
print("ok_val")

file_test = open('../test.txt',"w")
path_test = '/home/hehaibin/ch/before_RFL-CDNet-main/samples/'
path_test_list = sorted(os.listdir(path_test+'A/'))
random.shuffle(path_test_list)
for file_A in path_test_list:
    path_file_A = path_test+"A/"+file_A
    path_file_B = path_test+"B/"+file_A
    path_file_label = path_test+"OUT/"+file_A
    # img = Image.open(path_file_label)
    # if not img.getbbox():
    #     print(path_file_label)
    #     continue
    file_test.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
file_test.close()
print("ok_test")