import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics, load_model_test
import os
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

model = load_model_test(opt, dev)
path = './tmp/checkpoint_cd_epoch_best.pt'   # the path of the model"_epoch_best.pt"
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

        [cd_preds, cd_preds1, cd_preds2, cd_preds3, cd_preds4, cd_preds5, cd_preds6, cd_preds7, cd_preds8] = model(batch_img1, batch_img2)

        cd_preds = cd_preds[-1]
        cd_preds1 = cd_preds1[-1]
        cd_preds2 = cd_preds2[-1]
        cd_preds3 = cd_preds3[-1]
        cd_preds4 = cd_preds4[-1]
        cd_preds5 = cd_preds5[-1]
        cd_preds6 = cd_preds6[-1]
        cd_preds7 = cd_preds7[-1]
        cd_preds8 = cd_preds8[-1]
        print("Shape of cd_preds:", cd_preds.shape)

        change_channel = cd_preds[1, :, :]  # 提取第二个通道
        change_channel1 = cd_preds1[1, :, :]
        change_channel2 = cd_preds2[1, :, :]
        change_channel3 = cd_preds3[1, :, :]
        change_channel4 = cd_preds4[1, :, :]
        change_channel5 = cd_preds5[1, :, :]
        change_channel6 = cd_preds6[1, :, :]
        change_channel7 = cd_preds7[1, :, :]
        change_channel8 = cd_preds8[1, :, :]
        
        
        change_channel_np = change_channel.detach().cpu().numpy()
        change_channel_np1 = change_channel1.detach().cpu().numpy()
        change_channel_np2 = change_channel2.detach().cpu().numpy()
        change_channel_np3 = change_channel3.detach().cpu().numpy()
        change_channel_np4 = change_channel4.detach().cpu().numpy()
        change_channel_np5 = change_channel5.detach().cpu().numpy()
        change_channel_np6 = change_channel6.detach().cpu().numpy()
        change_channel_np7 = change_channel7.detach().cpu().numpy()
        change_channel_np8 = change_channel8.detach().cpu().numpy()
        
        min_value = change_channel_np.min()
        min_value1 = change_channel_np1.min()
        min_value2 = change_channel_np2.min()
        min_value3 = change_channel_np3.min()
        min_value4 = change_channel_np4.min()
        min_value5 = change_channel_np5.min()
        min_value6 = change_channel_np6.min()
        min_value7 = change_channel_np7.min()
        min_value8 = change_channel_np8.min()
        
        max_value = change_channel_np.max()
        max_value1 = change_channel_np1.max()
        max_value2 = change_channel_np2.max()
        max_value3 = change_channel_np3.max()
        max_value4 = change_channel_np4.max()
        max_value5 = change_channel_np5.max()
        max_value6 = change_channel_np6.max()
        max_value7 = change_channel_np7.max()
        max_value8 = change_channel_np8.max()
        
        normalized_change_channel = (change_channel_np - min_value) / (max_value - min_value)
        normalized_change_channel1 = (change_channel_np1 - min_value1) / (max_value1 - min_value1)
        normalized_change_channel2 = (change_channel_np2 - min_value2) / (max_value2 - min_value2)
        normalized_change_channel3 = (change_channel_np3 - min_value3) / (max_value3 - min_value3)
        normalized_change_channel4 = (change_channel_np4 - min_value4) / (max_value4 - min_value4)
        normalized_change_channel5 = (change_channel_np5 - min_value5) / (max_value5 - min_value5)
        normalized_change_channel6 = (change_channel_np6 - min_value6) / (max_value6 - min_value6)
        normalized_change_channel7 = (change_channel_np7 - min_value7) / (max_value7 - min_value7)
        normalized_change_channel8 = (change_channel_np8 - min_value8) / (max_value8 - min_value8)

        sample_index = 0

        plt.imshow(normalized_change_channel[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_1'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel1[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_2'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel2[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_3'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel3[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_4'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel4[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_5'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel5[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_6'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel6[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_7'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel7[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_8'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        plt.imshow(normalized_change_channel8[:, :], cmap='gray')
        plt.colorbar()
        plt.title("Visualization of Change Channel")
        file_path = './output_img/' + str(index_img).zfill(5) + '_9'
        plt.savefig(file_path + '.png', dpi = 300)
        plt.close()  # 关闭当前图像，释放资源
        
        
        
        index_img += 1