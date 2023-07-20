from __future__ import print_function

import argparse
import torch
from torch.utils.data import DataLoader
from cls_eval import mcdropout_test
from Models.dataloader.data_utils import *


#WA方法
if __name__ == '__main__':

    parser = argparse.ArgumentParser('argument for WA testing')
    parser.add_argument('--data_path', type=str, default='/home/user/zxy/zxy22/02FSDA/data/MSTAR_REAL/fdf8-EOC/15')
    parser.add_argument('--dataset', type=str, default='MSTAR', choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs', 'MSTAR'])
    parser.add_argument('--sample_flag', type=bool, default=True, help='sample from dataset')
    parser.add_argument('--result_path', type=str, default='./results/WA/', help='txt path for writting test results')

    opt = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_path = './Pretrained_models/15_grass.pth'
    model = torch.load(model_path)
    model = model.to(device)




    Dataset = set_up_datasets(opt)
    iid_testset_15 = Dataset('test_15', 1, opt)
    iid_testset_15_loader = DataLoader(dataset=iid_testset_15, batch_size=1,
                                       shuffle=False)

    iid_testset_17 = Dataset('test_17', 1, opt)
    iid_testset_17_loader = DataLoader(dataset=iid_testset_17, batch_size=1,
                                       shuffle=False)
    iid_testset_30 = Dataset('test_30', 1, opt)
    iid_testset_30_loader = DataLoader(dataset=iid_testset_30, batch_size=1,
                                       shuffle=False)
    iid_testset_45 = Dataset('test_45', 1, opt)
    iid_testset_45_loader = DataLoader(dataset=iid_testset_45, batch_size=1,
                                       shuffle=False)

    # ood test
    ood_testset_15 = Dataset('uncertainty_oodtest_15', 1, opt)
    ood_testset_15_loader = DataLoader(dataset=ood_testset_15, batch_size=1, shuffle=False)

    ood_testset_17 = Dataset('uncertainty_oodtest_17', 1, opt)
    ood_testset_17_loader = DataLoader(dataset=ood_testset_17, batch_size=1,
                                       shuffle=False)
    all_test_15 = mcdropout_test(iid_testset_15_loader, model, device, opt, title='id15_15grass')
    all_test_17 = mcdropout_test(iid_testset_17_loader, model, device, opt, title='id17_15grass')
    all_test_30 = mcdropout_test(iid_testset_30_loader, model, device, opt, title='id30_15grass')
    all_test_45 = mcdropout_test(iid_testset_45_loader, model, device, opt, title='id45_15grass')
    mcdropout_test(ood_testset_15_loader, model, device, opt, title='ood15_15grass_15')
    mcdropout_test(ood_testset_17_loader, model, device, opt, title='ood17_15grass_17')

