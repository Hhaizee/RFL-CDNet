import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, load_model_test
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from thop import profile
import os
from utils.metrics import Evaluator

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
# if torch.cuda.is_available():
#     dev = torch.device('cuda')
#     opt.cuda = True
# else:
#     dev = torch.device('cpu')
#     opt.cuda = False
#
# if opt.cuda:
#     try:
#         opt.gpu_ids = [int(s) for s in opt.gpu_ids.split(',')]
#     except ValueError:
#         raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
#
# num_gpus = len(opt.gpu_ids)
# opt.distributed = num_gpus>1
#
# if opt.distributed:
#     torch.cuda.set_device(opt.local_rank)
#     torch.distributed.init_process_group(
#         backend="nccl", init_method="env://")
#     device_ids = opt.gpu_ids
#     ngpus_per_node = len(device_ids)
#     opt.batch_size = int(opt.batch_size/ngpus_per_node)


test_loader = get_test_loaders(opt, batch_size=1)

model = load_model_test(opt, dev)
path = './tmp/checkpoint_cd_epoch_best.pt'   # the path of the model
model_weights = torch.load(path, map_location="cuda")
model_weights = remove_module_prefix(model_weights)
model.load_state_dict(model_weights)

#input = torch.randn(1, 2, 512, 512)
#flops, params = profile(model, inputs=(input,input))

#print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
#print("Params=", str(params/1e6)+'{}'.format("M"))

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()
evaluator = Evaluator(opt.num_class)

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        [cd_preds_1, cd_preds_2, cd_preds_3, cd_preds_4, cd_preds_5, cd_preds] = model(batch_img1, batch_img2)
        
        cd_preds = cd_preds[-1]

           
        _, cd_preds = torch.max(cd_preds, 1)
        evaluator.add_batch(labels, cd_preds)


mIoU = evaluator.Mean_Intersection_over_Union()
Precision = evaluator.Precision().data.cpu()
Recall = evaluator.Recall().data.cpu()
F1 = evaluator.F1().data.cpu()

print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(Precision, Recall, F1, mIoU))
