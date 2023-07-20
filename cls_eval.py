from __future__ import print_function

import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def relu_evidence(y):
    return F.relu(y)

#利用EDL进行不确定性估计
def uncertainty_estimation(model, img_variable, opt):
    output = model(img_variable)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    uncertainty = opt.n_ways / torch.sum(alpha, dim=1, keepdim=True)
    return uncertainty

def validate(val_loader, model, criterion, opt,device):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader)) as pbar:
            end = time.time()
            for idx, (input, target) in enumerate(pbar):
                if(opt.simclr):
                    input = input[0].float()
                else:
                    input = input.float()
                    
                if torch.cuda.is_available():
                    input = input.to(device)
                    target = target.to(device)

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc2 = accuracy(output, target, topk=(1, 2))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top2.update(acc2[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                                  "Acc@2":'{0:.2f}'.format(top2.avg.cpu().numpy(),2),
                                  "Loss" :'{0:.2f}'.format(losses.avg,2), 
                                 })
#                 if idx % opt.print_freq == 0:
#                     print('Test: [{0}/{1}]\t'
#                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                            idx, len(val_loader), batch_time=batch_time, loss=losses,
#                            top1=top1, top5=top5))

            print('Val_Acc@1 {top1.avg:.3f} Val_Acc@2 {top2.avg:.3f}'
                  .format(top1=top1, top2=top2))

    return top1.avg, top2.avg, losses.avg

def write_file(filepath, string):
    with open(filepath, 'a') as af:
        af.write(string)
        af.write('\n')

def data_test_complex(data_loader, model, device, opt, title):
    path = opt.result_path
    filename = opt.loss_option + '_' + title+'.txt'
    file_path = os.path.join(path, filename)
    write_file(file_path, '#'*15+title)
    """One epoch validation"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            end = time.time()
            for idx, (input, true_label, path) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)

                # compute output
                logits = model(input)#logits
                topk = (1,)
                maxk = max(topk)
                _, pred_label = logits.topk(maxk, 1, True, True)
                prob = F.softmax((logits), dim=1)

                evidence = relu_evidence(logits)
                alpha = evidence + 1
                uncertainty = opt.n_ways / torch.sum(alpha, dim=1, keepdim=True)
                prob_edl = alpha / torch.sum(alpha, dim=1, keepdim=True)
                # measure accuracy and record loss
                acc1, acc2 = accuracy(logits, true_label, topk=(1,2))
                top1.update(acc1[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy())})
                write_file(file_path, '*'*10)
                write_file(file_path, 'image_path:  ' + str(path))
                write_file(file_path, 'logits:   ' + str(logits))
                write_file(file_path, 'prob:   ' + str(prob))
                write_file(file_path, 'prob_edl:   ' + str(prob_edl))
                write_file(file_path, 'pred_label:   ' + str(pred_label))
                write_file(file_path, 'true_label:   ' + str(true_label))
                write_file(file_path, 'alpha:   ' + str(alpha))
                write_file(file_path, 'uncertainty:   ' + str(uncertainty))
            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))

def data_test(data_loader, model, device):
    """One epoch validation"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            end = time.time()
            for idx, (input, true_label, path) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)

                # compute output
                logits = model(input)#logits
                # measure accuracy and record loss
                acc1, acc2 = accuracy(logits, true_label, topk=(1,2))
                top1.update(acc1[0], input.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy())})
            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def uncertainty_test(uncertainty_test_loader, model, wandb, opt,count_uncertainty,device):
    """One epoch validation"""
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with tqdm(uncertainty_test_loader, total=len(uncertainty_test_loader)) as pbar:
            for idx, (input, target) in enumerate(pbar):
                count_uncertainty = count_uncertainty + 1
                if torch.cuda.is_available():
                    input = input.to(device)
                output = model(input)
                evidence = relu_evidence(output)
                alpha = evidence + 1
                uncertainty = opt.n_ways / torch.sum(alpha, dim=1, keepdim=True)
                prob_edl = alpha / torch.sum(alpha, dim=1, keepdim=True)
                prob_edl = np.array(prob_edl[0].cpu())
                percentage = F.softmax((output), dim=1)
                percentage = np.array(percentage[0].cpu())
                percentage_list = percentage.tolist()
                wandb.log({
                    'count_uncertainty': count_uncertainty,
                    'uncertainty': uncertainty,
                    'prediction probability of EDL': max(prob_edl),
                    'prediction probability': max(percentage),
                    'prediction label': percentage_list.index(max(percentage)),
                    'true label': target})
    return count_uncertainty

def deep_ensemble_test(data_loader, model, device, opt, title, fw=True):#, loss_re = False
    if fw:
        path = opt.result_path
        filename = title+'.txt'
        file_path = os.path.join(path, filename)
        write_file(file_path, '#'*15+title)
    """One epoch validation"""
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()#测试的时候也打开dropout
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            for idx, (input, true_label, path) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)
                # compute output
                logits = model(input)  # logits
                topk = (1,)
                maxk = max(topk)
                _, pred_label = logits.topk(maxk, 1, True, True)
                # measure accuracy and record loss
                acc1, acc2 = accuracy(logits, true_label, topk=(1, 2))
                top1.update(acc1[0], input.size(0))

                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy())})
                if fw:
                    write_file(file_path, '*'*10)
                    write_file(file_path, 'image_path:  ' + str(path))
                    write_file(file_path, 'logits:   ' + str(logits))
                    write_file(file_path, 'pred_label:   ' + str(pred_label))
                    write_file(file_path, 'true_label:   ' + str(true_label))

            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))
    if fw:
        pass
    else:
        return top1.avg



def mcdropout_test(data_loader, model, device, opt, title, fw=True):#, loss_re = False
    if fw:
        path = opt.result_path
        filename = title+'.txt'
        file_path = os.path.join(path, filename)
        write_file(file_path, '#'*15+title)
    """One epoch validation"""
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()#Dropout is also turned on when testingt
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            for idx, (input, true_label, path) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)
                output_list = []
                p_list = []
                for i in range(opt.T):
                    # compute output
                    logits = model(input)#logits
                    p_list.append(F.softmax((logits), dim=1).cpu().numpy()[0])
                    output_list.append(logits)
                output_mean = torch.cat(output_list, 0).mean(0)
                output_mean = torch.unsqueeze(output_mean, -2)#Expand the tensor dimension，tensor([-0.4710, -0.2836, -0.0831, -0.6003]) --》tensor([[-0.4710, -0.2836, -0.0831, -0.6003]])
                topk = (1,)
                maxk = max(topk)
                _, pred_label = output_mean.topk(maxk, 1, True, True)
                prob_mean = F.softmax((output_mean), dim=1)
                acc1, acc2 = accuracy(output_mean, true_label, topk=(1,2))
                top1.update(acc1[0], input.size(0))
                # loss = torch.nn.CrossEntropyLoss(output_mean, true_label)
                # CrossEntropyLoss += loss.item()

                # print('image_path:  ', path)
                # print('prob_list:  ', p_list)
                # print('logits_mean:  ', output_mean)
                # print('prob_mean:  ', prob_mean)
                # print('pred_label:  ', pred_label)
                # print('true_label:  ', true_label)
                prob = []
                for p in p_list:
                    prob.append(p[pred_label.cpu().numpy()])
                var_p = np.var(prob)

                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy())})
                if fw:
                    write_file(file_path, '*'*10)
                    write_file(file_path, 'image_path:  ' + str(path))
                    write_file(file_path, 'prob_list:  ' + str(p_list))
                    write_file(file_path, 'logits_mean:   ' + str(output_mean))
                    write_file(file_path, 'prob_mean:   ' + str(prob_mean))
                    write_file(file_path, 'pred_label:   ' + str(pred_label))
                    write_file(file_path, 'true_label:   ' + str(true_label))
                    write_file(file_path, 'uncertainty (various of p_max):   ' + str(var_p))#计算predict label上的概率方差

            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))
    if fw:
        pass
    # if loss_re:
    #     return top1.avg, CrossEntropyLoss
    else:
        return top1.avg

