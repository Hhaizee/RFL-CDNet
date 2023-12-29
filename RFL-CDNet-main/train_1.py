import datetime
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
from utils.metrics import Evaluator
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import time

def add_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = 'module.' + key
        new_state_dict[new_key] = value
    return new_state_dict

if __name__ == '__main__':

    """
    Initialize Parser and define arguments
    """
    parser, metadata = get_parser_with_args()
    opt = parser.parse_args()

    """
    Initialize experiments log
    """
    logging.basicConfig(level=logging.INFO)
    os.makedirs(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/', exist_ok=True)
    path = opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    writer = SummaryWriter(path)


    """
    Set up environment: define paths, download data, and set device
    """
    #here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        opt.cuda = True
    else:
        dev = torch.device('cpu')
        opt.cuda = False
    # dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

    # ##############################################################################################
    if opt.cuda:
        try:
            opt.gpu_ids = [int(s) for s in opt.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    num_gpus = len(opt.gpu_ids)
    opt.distributed = num_gpus>1

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        device_ids = opt.gpu_ids
        ngpus_per_node=len(device_ids)
        opt.batch_size = int(opt.batch_size/ngpus_per_node)

    if opt.sync_bn is None:
        if opt.cuda and len(opt.gpu_ids) > 1:
            opt.sync_bn = True
        else:
            opt.sync_bn = False
    # ################################################################################################


    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(seed=777)

    train_loader, train_sampler, val_loader = get_loaders(opt)

    """
    Load Model then define other aspects of the model
    """
    logging.info('LOADING Model')
    model = load_model(opt, dev)
    opt.start_epoch = 0

    criterion = get_criterion(opt)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)  # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    # scheduler = LR_Scheduler(opt.lr_scheduler, opt.learning_rate, opt.epochs, len(train_loader))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70,80,90,100,105], 0.5)

    if opt.resume is not None:
        if not os.path.isfile(opt.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume,map_location='cuda:0')
        opt.start_epoch = int(opt.resume.split('.')[0].split('/')[-1].split('_')[-1]) + 1
        if opt.cuda:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}' (epoch {})" .format(opt.resume, opt.start_epoch))

    model_dict = model.state_dict()
    pretrained_dict = torch.load(opt.resume_cd,map_location='cpu')
    pretrained_dict = add_module_prefix(pretrained_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)      

    """
     Set starting values
    """
    best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1, 'cd_miou':-1}
    logging.info('STARTING training')
    total_step = -1

    for epoch in range(opt.start_epoch, opt.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = initialize_metrics()
        val_metrics = initialize_metrics()
        evaluator = Evaluator(opt.num_class)

        """
        Begin Training
        """
        model.train()
        logging.info('SET model mode to train!')
        confusion_matrix = torch.zeros(opt.num_class, opt.num_class)
        batch_iter = 0
        train_loss = 0.0
        tbar = tqdm(train_loader)
        loss_print = []
        for i, [batch_img1, batch_img2, labels] in enumerate(tbar):
            # time_start = time.time()
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            total_step += 1
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss, backprop
            # time_model_start = time.time()
            # here!!!
            [cd_preds, cd_preds1, cd_preds2, cd_preds3, cd_preds4, cd_preds5] = model(batch_img1, batch_img2)
            # time_model_complete = time.time()
            # time_model = time_model_complete - time_model_start
            # print("time_model=",time_model)

            # time_loss_start = time.time()
            #here
            cd_loss = criterion(cd_preds, labels)+criterion(cd_preds1, labels)+criterion(cd_preds2, labels)+criterion(cd_preds3, labels)+criterion(cd_preds4, labels)+criterion(cd_preds5, labels)
            #cd_loss = criterion(cd_preds, labels)+criterion(cd_preds1, labels)+criterion(cd_preds2, labels)
            # print("time_loss_cal=", time_loss_cal)
            # loss = cd_loss

            # backward_start = time.time()
            cd_loss.backward()
            optimizer.step()
            loss_print.append(cd_loss.data.cpu().numpy())
            # time_backward = time.time() - backward_start
            # print("time_backward=", time_backward)

            # train_loss += cd_loss.item()
            # tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            # time_loss_complete = time.time()
            # time_loss_back = time_loss_complete - time_loss_calculate
            # print("time_loss_back=",time_loss_back)

            # cd_preds = cd_preds[-1]
            # _, cd_preds = torch.max(cd_preds, 1)
            # #
            # # time_metrics_start = time.time()
            # # # Calculate and log other batch metrics
            # cd_corrects = (100 *
            #                (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
            #                (labels.size()[0] * (opt.patch_size**2)))
            # #
            # time_eval_start = time.time()
            # gt_image = labels
            # pre_image = cd_preds
            # # evaluator.add_batch(gt_image, pre_image)
            # confusion_matrix = evaluator.generate_matrix(gt_image,pre_image)
            # mIoU = evaluator.Mean_Intersection_over_Union(confusion_matrix)
            # precision = evaluator.Precision(confusion_matrix)
            # recall = evaluator.Recall(confusion_matrix)
            # f1_score = evaluator.F1(confusion_matrix)
            # cd_train_report = [mIoU, precision, recall, f1_score]
            #
            #
            # # cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
            # #                        cd_preds.data.cpu().numpy().flatten(),
            # #                        average='binary',
            # #                        pos_label=1)
            # time_eval = time.time() - time_eval_start
            # print("time_eval:", time_eval)
            # #
            # train_metrics = set_metrics(train_metrics,
            #                             cd_loss,
            #                             cd_corrects,
            #                             cd_train_report,
            #                             scheduler.get_last_lr())
            # #
            # # # log the batch mean metrics
            # mean_train_metrics = get_mean_metrics(train_metrics)
            #
            # for k, v in mean_train_metrics.items():
            #     writer.add_scalars(str(k), {'train': v}, total_step)
            #     print(k)
            #     print(v)
            #     # print('\n')

            # time_metrics_complete = time.time()
            # time_metrics = time_metrics_complete - time_metrics_start
            # print("time_metrics=",time_metrics)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        scheduler.step()
        loss_mean = np.mean(loss_print)
        print("train_loss:", loss_mean)
        # logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """
        Begin Validation
        """
        total_step = -1
        batch_iter = 0
        test_loss = 0.0
        model.eval()
        evaluator.reset()
        val_loss_list = []
        tbar = tqdm(val_loader, desc='\r')
        with torch.no_grad():
            for batch_img1, batch_img2, labels in tbar:
                # Set variables for training
                tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + opt.batch_size))
                batch_iter = batch_iter + opt.batch_size
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                labels = labels.long().to(dev)

                # Get predictions and calculate loss
                #here!!!
                [cd_preds, cd_preds1, cd_preds2, cd_preds3, cd_preds4, cd_preds5] = model(batch_img1, batch_img2)

                # here!!!
                cd_loss = criterion(cd_preds, labels)+criterion(cd_preds1, labels)+criterion(cd_preds2, labels)+criterion(cd_preds3, labels)+criterion(cd_preds4, labels)+criterion(cd_preds5, labels)
                val_loss_list.append(cd_loss.data.cpu().numpy())

                cd_preds5 = cd_preds5[-1]
                _, cd_preds5 = torch.max(cd_preds5, 1)

                evaluator.add_batch(labels, cd_preds5)

                # Calculate and log other batch metrics
                # cd_corrects = (100 *
                #                (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                #                (labels.size()[0] * (opt.patch_size**2)))
                #
                # # cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                # #                      cd_preds.data.cpu().numpy().flatten(),
                # #                      average='binary',
                # #                      pos_label=1,zero_division="warn")
                #
                # cd_val_report = []
                # mIoU = evaluator.Mean_Intersection_over_Union()
                # Precision = evaluator.Precision()
                # Recall = evaluator.Recall()
                # F1 = evaluator.F1(Precision, Recall)
                # cd_val_report.append(Precision)
                # cd_val_report.append(Recall)
                # cd_val_report.append(F1)
                # cd_val_report.append(mIoU)
                #
                # val_metrics = set_metrics(val_metrics,
                #                           cd_loss,
                #                           cd_corrects,
                #                           cd_val_report,
                #                           scheduler.get_last_lr())
                #
                # # log the batch mean metrics
                # mean_val_metrics = get_mean_metrics(val_metrics)
                #
                # for k, v in mean_val_metrics.items():
                #     writer.add_scalars(str(k), {'val': v}, total_step)
                #
                # # clear batch variables from memory
                # del batch_img1, batch_img2, labels
                # evaluator.reset()

            mIoU = evaluator.Mean_Intersection_over_Union()
            # FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            Precision = evaluator.Precision().data.cpu()
            Recall = evaluator.Recall().data.cpu()
            F1 = evaluator.F1().data.cpu()
            val_loss = np.mean(val_loss_list)

            mean_val_metrics={}
            mean_val_metrics['val_loss'] = val_loss
            mean_val_metrics['cd_precisions'] = Precision
            mean_val_metrics['cd_recalls'] = Recall
            mean_val_metrics['cd_f1scores'] = F1
            mean_val_metrics['cd_miou'] = mIoU

            logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

            """
            Store the weights of good epochs based on validation results
            """
            if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                    or
                    (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                    or
                    (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

                # Insert training and epoch information to metadata dictionary
                logging.info('updata the model')
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists('./tmp'):
                    os.makedirs('./tmp', exist_ok=True)
                with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(str(metadata), fout)

                if opt.local_rank == 0:
                    torch.save(model.module.state_dict(), './tmp/checkpoint_epoch_'+str(epoch)+'.pt')

                # comet.log_asset(upload_metadata_file_path)
                if mean_val_metrics['cd_miou'] > best_metrics['cd_miou']:
                    torch.save(model.state_dict(), './tmp/checkpoint_cd_epoch_'+'best'+'.pt')
                    with open('./tmp/metadata_epoch_' + 'best' + '.json', 'w') as fout:
                        json.dump(str(metadata), fout)
                    best_metrics = mean_val_metrics

            print('An epoch finished.')
    writer.close()  # close tensor board
    print('Done!')