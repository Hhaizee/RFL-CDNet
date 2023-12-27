'''
This file is used to save the output image
'''

import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics, load_model_test
import os
from tqdm import tqdm
import cv2

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt, batch_size=1)
#test_loader = get_val_loaders(opt, batch_size=1)

model = load_model_test(opt, dev)
#path = './tmp/checkpoint_cd_epoch_best.pt'   # the path of the model
path = '/home/hehaibin/ch/before_RFL-CDNet-main/tmp_whu_best/checkpoint_cd_epoch_best.pt'
model_weights = torch.load(path, map_location="cpu")
model_weights = remove_module_prefix(model_weights)
model.load_state_dict(model_weights)

model.eval()
index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # cd_preds = model(batch_img1, batch_img2)
#        [cd_preds] = model(batch_img1, batch_img2)
        [cd_preds_1, cd_preds_2, cd_preds_3,cd_preds_4, cd_preds_5, cd_preds] = model(batch_img1, batch_img2)

        cd_preds_1 = cd_preds_1[-1]
        _, cd_preds_1 = torch.max(cd_preds_1, 1)
        cd_preds_1 = cd_preds_1.data.cpu().numpy()
        cd_preds_1 = cd_preds_1.squeeze() * 255

        cd_preds_2 = cd_preds_2[-1]
        _, cd_preds_2 = torch.max(cd_preds_2, 1)
        cd_preds_2 = cd_preds_2.data.cpu().numpy()
        cd_preds_2 = cd_preds_2.squeeze() * 255

        cd_preds_3 = cd_preds_3[-1]
        _, cd_preds_3 = torch.max(cd_preds_3, 1)
        cd_preds_3 = cd_preds_3.data.cpu().numpy()
        cd_preds_3 = cd_preds_3.squeeze() * 255

        cd_preds_4 = cd_preds_4[-1]
        _, cd_preds_4 = torch.max(cd_preds_4, 1)
        cd_preds_4 = cd_preds_4.data.cpu().numpy()
        cd_preds_4 = cd_preds_4.squeeze() * 255
        
#        cd_preds_5 = cd_preds_5[-1]
#        _, cd_preds_5 = torch.max(cd_preds_5, 1)
#        cd_preds_5 = cd_preds_5.data.cpu().numpy()
#        cd_preds_5 = cd_preds_5.squeeze() * 255

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        file_path = './output_img/' + str(index_img).zfill(5) + '_1'
#        file_path = './output_img/' + str(name[0])
#        cv2.imwrite(file_path + '_stage_1' + '.png', cd_preds_1)
#        cv2.imwrite(file_path + '_stage_2' + '.png', cd_preds_2)
#        cv2.imwrite(file_path + '_stage_3' + '.png', cd_preds_3)
#        cv2.imwrite(file_path + '_stage_4' + '.png', cd_preds_4)
#        cv2.imwrite(file_path + '_stage_e' + '.png', cd_preds_5)
        cv2.imwrite(file_path + '_final' + '.png', cd_preds)



        index_img += 1

