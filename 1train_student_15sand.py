# our method
from __future__ import print_function

import os
import argparse
import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from util import accuracy, AverageMeter
from cls_eval import data_test, data_test_complex

import numpy as np
import copy
from Models.dataloader.data_utils import *
import random
from uncertainty_loss import edl_digamma_loss, edl_mse_loss, edl_log_loss, Prior_edl_mse_loss
import torch.nn.functional as F

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    # device = torch.device("cpu")
    return device

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs for few-shot dataset')
    parser.add_argument('--sample_times', type=int, default=100, help='number of sampling from MSTAR data to compose a few-shot dataset')
    parser.add_argument('--loss_option', type=str, default='EDL',
                        choices=['CE', 'EDL', 'Prior-EDL'],
                        help='the choice of loss function, if CE in loss_option the parameter ''use_uncertainty'' must be false ')

    #uncertainty
    parser.add_argument('--use_uncertainty', type=bool, default=True, help='use uncertainty estimation')
    parser.add_argument('--uncertainty_loss', type=str, default='mse',
                        choices=['mse', 'log', 'digamma'], help='use loss function for classification uncertainty estimation')
    # optimization for SGD
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='MSTAR', choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs', 'MSTAR'])
    parser.add_argument('--sub_dataset', type=str, default='sand_15', choices=['MSTAR_fdf8_EOC_15', 'Grass_EOC_17', 'MSTAR_fdf8', 'Grass_fdf8_3C', 'sand_15', 'grass_15'])
    parser.add_argument('--data_path', type=str, default='./data/MSTAR_REAL/fdf8-EOC/15')
    parser.add_argument('--model_path', type=str, default='./Pretrained_models/', help='path for pretrained teacher model')
    parser.add_argument('--model_name', type=str, default='15_sand.pth', help='model name for pretrained teacher model')
    parser.add_argument('--result_root', type=str, default='./results/15_sand/', help='txt path for writting test results')
    parser.add_argument('--result_path', type=str, default=None, help='txt path for writting test results')

    parser.add_argument('--sample_flag', type=bool, default=True, help='sample from dataset')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=5, help='temperature for KD distillation')
    parser.add_argument('--kd_EDL', type=bool, default=False, help='use EDL probability to distillation')

    # setting for few-shot learning
    parser.add_argument('--n_ways', type=int, default=4,
                        help='Number of classes for doing each classification run')#parameter for support set
    parser.add_argument('--k_shots', type=int, default=5,
                        help='Number of shots in test')#parameter for support set

    opt = parser.parse_args()

    # set the path according to the environment

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def load_teacher(model_path, model_name, numclasses):
    """load the teacher model"""
    print('==> loading teacher model')
    print(model_name)
    model = torch.load(os.path.join(model_path, model_name))
    print('==> done')
    return model

def one_hot_embedding(labels, num_classes=3):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def y_embedding(y_hat,y):
    # Convert to One Hot Encoding
    y.append(y_hat)
    return y

def relu_evidence(y):
    return F.relu(y)

def train(train_loader, model_s, model_t, criterion_cls, optimizer, opt, epoch, device):
    """One epoch training"""
    model_s.train()
    model_t.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for i, (input, label, path) in enumerate(pbar):
            data_shot = input.to(device)
            label = label.to(device)
            logit_s = model_s(data_shot)

            with torch.no_grad():
                logit_t = model_t(data_shot)

            if opt.use_uncertainty:
                y = one_hot_embedding(label, opt.n_ways)#one_hot vector
                y = y.to(device)
                loss1 = criterion_cls(logit_s, y.float(), epoch, opt.n_ways, 5, device)
            else:
                loss1 = criterion_cls(logit_s, label)
            if opt.loss_option == 'CE' or opt.loss_option == 'EDL':
                loss = loss1
            elif opt.loss_option == 'Prior-EDL':
                y_prior = one_hot_embedding(label, opt.n_ways)
                target = one_hot_embedding(label, opt.n_ways)
                topk = (1,)
                maxk = max(topk)
                _, pred_s = logit_s.topk(maxk, 1, True, True)
                _, pred_t = logit_t.topk(maxk, 1, True, True)
                for indx, val in enumerate(pred_s.eq(pred_t)):
                    if val.item() == True:#预测结果相同且正确
                        if pred_s[indx] == label[indx]:
                            y_prior[indx] = F.softmax(logit_t[indx] / opt.kd_T, dim=0)
                loss = Prior_edl_mse_loss(logit_s, y_prior.float(), target, epoch, opt.n_ways, 5, device)#移除正确预测上的所有样本


            acc1, acc2 = accuracy(logit_s, label, topk=(1, 2))
            top1.update(acc1[0], input.size(0))
            losses.update(loss.item(), input.size(0))
            # ===================backward=====================
            optimizer.zero_grad()
            # loss_all = loss
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy()),
                              "Loss": '{0:.2f}'.format(losses.avg, 2),
                              })

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def main(opt):
    device = get_device()

    acc_list_all_test = []
    acc_list_test_15 = []
    acc_list_test_17 = []
    acc_list_test_30 = []
    acc_list_test_45 = []
    seed_list = [i for i in range(1000)]
    random.seed(1)
    random.shuffle(seed_list)

    # loss function
    if opt.use_uncertainty:
        print('The loss contains EDL loss')
        if opt.uncertainty_loss == 'digamma':
            criterion_cls = edl_digamma_loss
        elif opt.uncertainty_loss == 'log':
            criterion_cls = edl_log_loss
        elif opt.uncertainty_loss == 'mse':
            criterion_cls = edl_mse_loss
        else:
            opt.error("--uncertainty requires --mse, --log or --digamma.")
    else:
        criterion_cls = nn.CrossEntropyLoss()
        print('the loss contains cross Entropy')

    count_list = [i for i in range(opt.sample_times)]
    print('the count list length:', len(count_list))
    for count in count_list:
        # dataloader
        # train
        Dataset = set_up_datasets(opt)
        trainset = Dataset('all-train', seed_list[count], opt)
        setup_seed(50)
        train_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        # iid test the angle of test set is the same as train set
        testset = Dataset('all-test', 1, opt)
        iid_test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)

        iid_testset_17 = Dataset('test_17', 1, opt)
        iid_testset_17_loader = DataLoader(dataset=iid_testset_17, batch_size=1,
                                           shuffle=False)
        iid_testset_30 = Dataset('test_30', 1, opt)
        iid_testset_30_loader = DataLoader(dataset=iid_testset_30, batch_size=1,
                                           shuffle=False)
        iid_testset_45 = Dataset('test_45', 1, opt)
        iid_testset_45_loader = DataLoader(dataset=iid_testset_45, batch_size=1,
                                           shuffle=False)

        #ood test
        ood_testset_15 = Dataset('uncertainty_oodtest_15', 1, opt)
        ood_testset_15_loader = DataLoader(dataset=ood_testset_15, batch_size=1, shuffle=False)

        ood_testset_17 = Dataset('uncertainty_oodtest_17', 1, opt)
        ood_testset_17_loader = DataLoader(dataset=ood_testset_17, batch_size=1,
                                           shuffle=False)

        # model
        model_t = load_teacher(opt.model_path, opt.model_name, opt.n_ways)
        model_s = copy.deepcopy(model_t)

        optimizer = optim.SGD(model_s.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

        model_t = model_t.to(device)
        model_s = model_s.to(device)
        test_acc_all_test = 0
        best_acc_test_17 = 0
        best_acc_test_30 = 0
        best_acc_test_45 = 0
        # routine: supervised model distillation
        for epoch in range(1, opt.epochs + 1):
            print("==> training...")
            time1 = time.time()
            train_acc, train_loss = train(train_loader, model_s, model_t, criterion_cls,
                                          optimizer, opt, epoch, device)
            time2 = time.time()
            print('epoch {}, total time {:.2f}, train accuracy {}, train loss {}'.format(epoch, time2 - time1, train_acc, train_loss))
            if epoch % 3 == 0:
                all_test_acc = data_test(iid_test_loader, model_s, device)
                if test_acc_all_test < all_test_acc and epoch > 45:
                    test_acc_all_test = all_test_acc
                    best_model_all_test = copy.deepcopy(model_s)

                test_17_acc = data_test(iid_testset_17_loader, model_s, device)
                if best_acc_test_17 < test_17_acc and epoch > 45:
                    best_acc_test_17 = test_17_acc
                    best_model_test_17 = copy.deepcopy(model_s)

                test_30_acc = data_test(iid_testset_30_loader, model_s, device)
                if best_acc_test_30 < test_30_acc and epoch > 45:
                    best_acc_test_30 = test_30_acc
                    best_model_test_30 = copy.deepcopy(model_s)

                test_45_acc = data_test(iid_testset_45_loader, model_s, device)
                if best_acc_test_45 < test_45_acc and epoch > 45:
                    best_acc_test_45 = test_45_acc
                    best_model_test_45 = copy.deepcopy(model_s)

        all_test_acc = data_test(iid_test_loader, best_model_all_test, device)
        test_17_acc = data_test(iid_testset_17_loader, best_model_test_17, device)
        test_30_acc = data_test(iid_testset_30_loader, best_model_test_30, device)
        test_45_acc = data_test(iid_testset_45_loader, best_model_test_45, device)

        acc_list_all_test.append(all_test_acc.cpu())
        acc_list_test_17.append(test_17_acc.cpu())
        acc_list_test_30.append(test_30_acc.cpu())
        acc_list_test_45.append(test_45_acc.cpu())

        #Prediction on the optimal convergence model with a depression angle of 15°
        data_test_complex(iid_test_loader, best_model_all_test, device, opt, title='id15_15sand_15')
        data_test_complex(ood_testset_15_loader, best_model_all_test, device, opt, title='ood15_15sand_15')
        data_test_complex(ood_testset_17_loader, best_model_all_test, device, opt,  title='ood17_15sand_15')

        #Prediction on the optimal convergence model with a depression angle of 17°
        data_test_complex(iid_testset_17_loader, best_model_test_17, device, opt, title='id17_15sand_17')
        data_test_complex(ood_testset_15_loader, best_model_test_17, device, opt, title='ood15_15sand_17')
        data_test_complex(ood_testset_17_loader, best_model_test_17, device, opt, title='ood17_15sand_17')

        # Prediction on the optimal convergence model with a depression angle of 30°
        data_test_complex(iid_testset_30_loader, best_model_test_30, device, opt, title='id30_15sand_30')
        data_test_complex(ood_testset_15_loader, best_model_test_30, device, opt, title='ood15_15sand_30')
        data_test_complex(ood_testset_17_loader, best_model_test_30, device, opt, title='ood17_15sand_30')

        # #Prediction on the optimal convergence model with a depression angle of 45°
        data_test_complex(iid_testset_45_loader, best_model_test_45, device, opt, title='id45_15sand_45')
        data_test_complex(ood_testset_15_loader, best_model_test_45, device, opt, title='ood15_15sand_45')
        data_test_complex(ood_testset_17_loader, best_model_test_45, device, opt, title='ood17_15sand_45')

if __name__ == '__main__':
    k_list = [1, 2, 5, 10, 20]
    bs_list = [1, 1, 2, 5, 8]
    opt = parse_option()
    print('the loss function is: ', opt.loss_option)
    print('the temperature is: ', opt.kd_T)
    for i in range(len(k_list)):
        opt.k_shots = k_list[i]
        opt.batch_size = bs_list[i]
        opt.result_path = os.path.join(opt.result_root, str(opt.n_ways)+'way-'+str(opt.k_shots)+'shot')
        print('opt.result_path', opt.result_path)
        if not os.path.exists(opt.result_path):
            os.mkdir(opt.result_path)
        main(opt)


