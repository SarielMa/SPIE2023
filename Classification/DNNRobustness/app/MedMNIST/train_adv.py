
from MedMNIST_Dataset import get_dataloader
from MedMNIST_CNN import main, get_filename
import sys
sys.path.append('../../core')
#%%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
from RobustDNN_ours import our_loss, check_margin, update_margin

#%%
random_seed=0
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(random_seed)
#%%
loader_check=get_dataloader(batch_size = 1024, return_idx=(True, False, False))
loader_check=loader_check[0]#training set
#%%
#%%
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    if epoch+1 == 60:
        update_lr(optimizer, 0.09)
    if epoch+1 == 90:
        update_lr(optimizer, 0.03)
    if epoch+1 == 120:
        update_lr(optimizer, 0.009)
def train(model, device, optimizer, dataloader, epoch, train_arg):
    if arg['optimizer'] == 'SGD':
        adjust_learning_rate(optimizer, epoch)
    model.train()
    norm_type=train_arg['norm_type']
    noise=train_arg['noise']
    beta=train_arg['beta']
    beta_position=train_arg['beta_position']
    epoch_refine=train_arg['epoch_refine']
    E=train_arg['E']
    delta=train_arg['delta']
    max_iter1=train_arg['max_iter1']
    max_iter2=train_arg['max_iter2']
    alpha1=train_arg['alpha1']
    alpha2=train_arg['alpha2']
    pgd_loss_fn1=train_arg['pgd_loss_fn1']
    pgd_loss_fn2=train_arg['pgd_loss_fn2']
    model_eval_attack=train_arg['model_eval_attack']
    model_eval_Xn=train_arg['model_eval_Xn']
    model_Xn_advc_p=train_arg['model_Xn_advc_p']
    Xn1_equal_X=arg['Xn1_equal_X']
    Xn2_equal_Xn=arg['Xn2_equal_Xn']
    #--------------
    refine_Xn_max_iter=train_arg['refine_Xn_max_iter']
    stop=arg['stop']
    stop_near_boundary=False
    stop_if_label_change=False
    stop_if_label_change_next_step=False
    if stop==1:
        stop_near_boundary=True
    elif stop==2:
        stop_if_label_change=True
    elif stop==3:
        stop_if_label_change_next_step=True
    #-------------
    print('noise', noise, 'epoch', epoch, 'delta', delta, 'stop', stop,
          'max_iter1', max_iter1, 'alpha1', alpha1, 'max_iter2', max_iter2, 'alpha2', alpha2)
    #--------------
    loss_train=0
    loss1_train=0
    loss2_train=0
    loss3_train=0
    acc1_train =0
    acc2_train =0
    sample1_count=0
    sample2_count=0
    for batch_idx, (X, Y, Idx) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #----------------------------
        model.zero_grad()
        #----------------------------
        rand_init_norm=torch.clamp(E[Idx]-delta, min=delta).to(device)
        margin=E[Idx].to(device)
        step1=alpha1*margin/max_iter1
        loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n = our_loss(model, X, Y,
                                                                       norm_type=norm_type,
                                                                       rand_init_norm=rand_init_norm,
                                                                       margin=margin,
                                                                       max_iter=max_iter1,
                                                                       step=step1,
                                                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                                                       Xn1_equal_X=Xn1_equal_X,
                                                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                                                       stop_near_boundary=stop_near_boundary,
                                                                       stop_if_label_change=stop_if_label_change,
                                                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                                                       beta=beta, beta_position=beta_position,
                                                                       use_optimizer=False,
                                                                       pgd_loss_fn=pgd_loss_fn1,
                                                                       model_eval_attack=model_eval_attack,
                                                                       model_eval_Xn=model_eval_Xn,
                                                                       model_Xn_advc_p=model_Xn_advc_p)
        loss.backward()
        optimizer.step()
        #---------------------------
        loss_train+=loss.item()
        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        loss3_train+=loss3.item()
        acc1_train+= torch.sum(Yp==Y).item()
        acc2_train+= torch.sum(Ypn==Y[idx_n]).item()
        sample1_count+=X.size(0)
        sample2_count+=Xn.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), loss.item(), loss1.item(), loss2.item(), loss3.item()))
    #---------------------------
    loss_train/=len(dataloader)
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    acc1_train/=sample1_count
    acc2_train/=sample2_count
    print('acc1_train', acc1_train, 'acc2_train', acc2_train)
    #---------------------------
    print('IMA_check_margin: begin', 'max_iter', max_iter2, 'alpha2', alpha2)
    step2=alpha2*E/max_iter2
    print('noise_e', delta*(epoch+1), 'step2', 'max=', step2.max().item(), 'min=', step2.min().item())
    flag1, flag2, E_new =check_margin(model, device, loader_check,
                                          margin=E, norm_type=norm_type,
                                          max_iter=max_iter2,
                                          step=step2,
                                          rand_init_norm=E-delta,
                                          refine_Xn_max_iter=refine_Xn_max_iter,
                                          stop_near_boundary=True,
                                          use_optimizer=False,
                                          pgd_loss_fn=pgd_loss_fn2)
    update_margin(E, delta, noise, flag1, flag2, E_new)
    #---------------------------
    fig, ax = plt.subplots()
    ax.hist(E.cpu().numpy(), 100)
    display.display(fig)
    filename=get_filename(arg['net_name'], arg['loss_name'], epoch)
    fig.savefig(filename+'_histE.png')
    plt.close(fig)
    #---------------------------
    return loss_train, acc1_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--noise', default=3.0, type=float) # 0.3Linf, 5.0L2.0
    parser.add_argument('--norm_type', default=2, type=float)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_refine', default=150, type=int)
    parser.add_argument('--epoch_end', default=150, type=int)
    parser.add_argument('--max_iter1', default=20, type=int)
    parser.add_argument('--max_iter2', default=20, type=int)
    parser.add_argument('--alpha1', default=4, type=float)
    parser.add_argument('--alpha2', default=4, type=float)
    parser.add_argument('--stop', default=1, type=int)
    parser.add_argument('--refine_Xn_max_iter', default=10, type=int)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--beta_position', default=1, type=int)
    parser.add_argument('--pgd_loss_fn1', default='slm', type=str)
    parser.add_argument('--pgd_loss_fn2', default='slm', type=str)
    parser.add_argument('--model_eval_attack', default=0, type=int)
    parser.add_argument('--model_eval_Xn', default=0, type=int)
    parser.add_argument('--model_Xn_advc_p', default=1, type=int)
    parser.add_argument('--Xn1_equal_X', default=1, type=int)
    parser.add_argument('--Xn2_equal_Xn', default=1, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--cuda_id', default=1, type=int)
    parser.add_argument('--net_name', default='resnet18', type=str)
    arg = parser.parse_args()
    print(arg)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.cuda_id)
    #-------------------------------------------
    sample_count_train=89996
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    #delta=arg.noise/arg.epoch_refine
    delta = 0.05
    loss_name=(str(arg.beta)+'ours'+str(arg.noise)+'L'+str(arg.norm_type)
               +'_'+str(arg.max_iter1)+'a'+str(arg.alpha1)
               +'_'+str(arg.max_iter2)+'a'+str(arg.alpha2)+'pgd'
               +'_s'+str(arg.stop)
               +'r0'
               +'b'+str(arg.beta_position)
               +'e'+str(arg.epoch_refine)
               +'i1'
               +'d'+str(delta)
               +'m'+str(arg.refine_Xn_max_iter)
               +str(bool(arg.model_eval_attack))[0]
               +str(bool(arg.model_eval_Xn))[0]
               +str(bool(arg.model_Xn_advc_p))[0]
               +'_'+arg.pgd_loss_fn1
               +'_'+arg.pgd_loss_fn2
               +'_'+arg.optimizer)
    if random_seed >0:
        loss_name+='_rs'+str(random_seed)
    #-------------------------------------------
    #stop=0 if every is False
    #stop=1 if stop_near_boundary=True
    #stop=2 if stop_if_label_change=True
    #stop=3 if stop_if_label_change_next_step=True
    #-------------------------------------------
    arg=vars(arg)
    arg['delta']=delta
    arg['E']=delta*torch.ones(sample_count_train, dtype=torch.float32)
    arg['return_idx']=(True, False, False)
    arg['loss_name']=loss_name
    arg['device'] = device
    main(epoch_start=arg['epoch_start'], epoch_end=arg['epoch_end'],
         train=train, arg=arg, evaluate_model=True)
#%%
