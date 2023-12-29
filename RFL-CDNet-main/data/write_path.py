import os
from PIL import Image
import random

# file_train = open('train-before.txt',"w")
# out = open("train.txt","w")
# n=0
# path_train = '/dat02/liujia/Datasets/WHU-building/train_nooverlap/'
# path_train_list = sorted(os.listdir(path_train+'A/'))
# random.shuffle(path_train_list)
# for file_A in path_train_list:
#     path_file_A = path_train+"A/"+file_A
#     path_file_B = path_train+"B/"+file_A
#     path_file_label = path_train+"OUT/"+file_A
#     # img = Image.open(path_file_label)
#     # if not img.getbbox():
#     #     print(path_file_label)
#     #     continue
#     n+=1
#     file_train.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
# print(n)
# path_train_another = '/dat02/liujia/Datasets/WHU-building/train_around_instance/'
# path_train_list = sorted(os.listdir(path_train_another+'A/'))
# random.shuffle(path_train_list)
# for file_A in path_train_list:
#     path_file_A = path_train_another+"A/"+file_A
#     path_file_B = path_train_another+"B/"+file_A
#     path_file_label = path_train_another+"OUT/"+file_A
#     img = Image.open(path_file_label)
#     if not img.getbbox():
#         print(path_file_label)
#         continue
#     n+=1
#     file_train.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
# file_train.close()
# print(n)
# with open('train-before.txt', "r") as f1:
#     lines = f1.read().splitlines()
# random.shuffle(lines)
# for line in lines:
#     out.write(line+'\n')
# print("ok_train")

file_train = open('train.txt',"w")
#path_train = '/project/jhliu4/ch/pp/datasets/shengteng_256/train/'
path_train ='/project/jhliu4/ch/Datasets/subset-refine/train/'
#path_train = '/project/jhliu4/ch/Datasets/WHUBuildingDataset/train_nooverlap/'
path_train_list = sorted(os.listdir(path_train + 'A/'))
random.shuffle(path_train_list)
for file_A in path_train_list:
    path_file_A = path_train+"A/"+file_A
    path_file_B = path_train+"B/"+file_A
    path_file_label = path_train+"OUT/"+file_A[0:-4]+'.jpg'
    file_train.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
file_train.close()
print("ok_train")

file_val = open('val.txt',"w")
path_val = '/project/jhliu4/ch/Datasets/subset-refine/val/'
path_val_list = sorted(os.listdir(path_val + 'A/'))
random.shuffle(path_val_list)
for file_A in path_val_list:
    path_file_A = path_val+"A/"+file_A
    path_file_B = path_val+"B/"+file_A
    path_file_label = path_val+"OUT/"+file_A[0:-4]+'.jpg'
    file_val.write(path_file_A + ' ' + path_file_B + ' ' + path_file_label + '\n')
file_val.close()
print("ok_val")

file_test = open('test.txt',"w")
path_test = '/project/jhliu4/ch/Datasets/subset-refine/val/'
path_test_list = sorted(os.listdir(path_test+'A/'))
random.shuffle(path_test_list)
for file_A in path_test_list:
    path_file_A = path_test+"A/"+file_A
    path_file_B = path_test+"B/"+file_A
    path_file_label = path_test+"OUT/"+file_A[0:-4]+'.jpg'
    file_test.write(path_file_A + ' ' + path_file_B + '\n')
file_test.close()
print("ok_test")