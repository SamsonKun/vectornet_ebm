# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from modeling.vectornet import HGNN
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from dataset import GraphDataset
from torch_geometric.data import DataLoader
from utils.eval import get_eval_metric_results_ebm
from tqdm import tqdm
import time
from typing import List

# %%
from modeling.lbebm import LBEBM,calculate_loss
import argparse
import torch.nn as nn
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument("--data_scale", default=1.86, type=float)
    parser.add_argument("--dec_size", default=[1024, 512, 1024], type=list) 
    parser.add_argument("--enc_dest_size", default=[256, 128], type=list) 
    parser.add_argument("--enc_latent_size", default=[256, 512], type=list) 
    parser.add_argument("--enc_past_size", default=[512, 256], type=list)
    parser.add_argument("--predictor_hidden_size", default=[1024, 512, 256], type=list) 
    parser.add_argument("--non_local_theta_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_phi_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_g_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_dim", default=128, type=int)
    # parser.add_argument("--fdim", default=16, type=int)
    parser.add_argument("--fdim", default=64, type=int)
    parser.add_argument("--future_length", default=30, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--kld_coeff", default=0.6, type=float)
    parser.add_argument("--future_loss_coeff", default=1, type=float)
    parser.add_argument("--dest_loss_coeff", default=2, type=float)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--lr_decay_size", default=0.5, type=float)
    parser.add_argument("--lr_decay_schedule", default=[120, 150, 180, 210, 240, 270, 300], type=list)

    parser.add_argument("--mu", default=0, type=float)
    parser.add_argument("--n_values", default=6, type=int)
    parser.add_argument("--nonlocal_pools", default=3, type=int)
    parser.add_argument("--num_epochs", default=400, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--past_length", default=20, type=int)
    parser.add_argument("--sigma", default=1.3, type=float)
    # parser.add_argument("--zdim", default=16, type=int)
    parser.add_argument("--zdim", default=64, type=int)
    parser.add_argument("--print_log", default=60, type=int)
    parser.add_argument("--sub_goal_indexes", default=[2, 10, 18, 26], type=list)
     


    parser.add_argument('--e_prior_sig', type=float, default=2, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=2, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_steps_pcd', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')
    parser.add_argument('--e_lr', default=0.00003, type=float)
    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--e_max_norm', type=float, default=25, help='max norm allowed')
    parser.add_argument('--e_decay', default=1e-4, help='weight decay for ebm')
    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--e_beta1', default=0.9, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)
    parser.add_argument('--memory_size', default=200000, type=int)
    parser.add_argument('--patience_epoch', default=20, type=int)
    parser.add_argument('--lr_threshold', default=0.000000003, type=float)


    parser.add_argument('--dataset_name', type=str, default='argoverse_interm_data')
    parser.add_argument('--dataset_folder', type=str, default='/home/zk/gitcode/dataset')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--batch_size',type=int,default=700)

    parser.add_argument('--ny', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='/home/zk/gitcode/lbebm/saved_models/lbebm_argoverse.pt')
    parser.add_argument('--competition_output_path', type=str, default='/home/zk/gitcode/dataset/argoverse/competition_files')

    return parser.parse_args()

# %%
TRAIN_DIR = os.path.join('/home/zk/gitcode/dataset/argoverse_interm_data_vector', 'train_intermediate')
VAL_DIR = os.path.join('/home/zk/gitcode/dataset/argoverse_interm_data_vector', 'val_intermediate')
SEED = 13
epochs = 50
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
batch_size = 64
decay_lr_factor = 0.3
decay_lr_every = 10
lr = 0.001
in_channels, out_channels = 8, 60
show_every = 1200
val_every = 1
small_dataset = False
end_epoch = 0
save_dir = 'trained_params_ebm'
best_minade = float('inf')
date = f"200630.epochs{epochs}.lr_decay{decay_lr_factor}.decay_every{decay_lr_every}.lr{lr}"
global_step = 0
# checkpoint_dir = os.path.join('trained_params', 'epoch_6.valminade_3.796.pth')
checkpoint_dir = None
# eval related
max_n_guesses = 1
horizon = 30
miss_threshold = 2.0


#%%
#%%
def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade, date):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'end_epoch' : end_epoch,
        'val_minade': val_minade
        }
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.{date}.{"xkhuang"}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


#%%
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # hyper parameters
    
    args = parse_args()

    train_data = GraphDataset(TRAIN_DIR).shuffle()
    val_data = GraphDataset(VAL_DIR)
    if small_dataset:
        train_loader = DataLoader(train_data[:1000], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data[:200], batch_size=batch_size)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

    # model = HGNN(in_channels, out_channels).to(device)
    model = LBEBM(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    if checkpoint_dir:
        load_checkpoint(checkpoint_dir, model, optimizer)

    # overfit the small dataset
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        print(f"start training at epoch:{epoch}")
        acc_loss = .0
        num_samples = 1
        start_tic = time.time()
        for data in train_loader:
            if epoch < end_epoch: break
            if isinstance(data, List):
                y = torch.cat([i.y for i in data], 0).view(-1, out_channels).to(device)
            else:
                data = data.to(device)
                y = data.y.view(-1, out_channels)
            future = y.view(y.size(0),-1,2).cumsum(axis=1).detach().clone()
            dest = future[:, args.sub_goal_indexes, :].detach().clone().view(y.size(0), -1) 
            future = future.view(y.size(0), -1)

            optimizer.zero_grad()
            # out = model(data)
            # loss = F.mse_loss(out, y)
            dest_recon, mu, var, interpolated_future, cd, en_pos, en_neg, pcd = model.forward(data,dest,global_step)
            dest_loss, future_loss, kld, subgoal_reg = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, args.sub_goal_indexes)
            loss = args.dest_loss_coeff * dest_loss + args.future_loss_coeff * future_loss + args.kld_coeff * kld  + cd + subgoal_reg
            loss.backward()
            acc_loss += batch_size * loss.item() #!!!!
            num_samples += y.shape[0] #!!!!
            optimizer.step()
            global_step += 1
            if (global_step + 1) % show_every == 0:
                # print(f"epoch{epoch},step{global_step}: dest_loss={dest_loss.item():8.6f}, future_loss={future_loss.item():8.6f}, kld={kld.item():8.6f}, cd={cd.item():8.6f}, en_pos={en_pos.item():8.6f}, en_neg={en_neg.item():8.6f}, pcd={pcd}, subgoal_reg={subgoal_reg.detach().cpu().numpy()}, time:{time.time() - start_tic: 4f}sec ")
                print(f"epoch{epoch}, step{global_step}: dest_loss={dest_loss.item():8.6f}, future_loss={future_loss.item():8.6f}, total_loss = {loss.item():8.6f}")
                print(f"        kld={kld.item():8.6f}, cd={cd.item():8.6f}, en_pos={en_pos.item():8.6f}, en_neg={en_neg.item():8.6f}, pcd={pcd}, subgoal_reg={subgoal_reg.detach().cpu().numpy()}, time:{time.time() - start_tic: 4f}sec ")
                # print( f"epoch {epoch} step {global_step}： loss:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
                # # print( f"epoch {epoch} step {global_step}： loss:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        scheduler.step()
        print(
            f"finished epoch {epoch}: loss:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        
        if (epoch+1) % val_every == 0 and (not epoch < end_epoch):
            print(f"eval as epoch:{epoch}")
            metrics = get_eval_metric_results_ebm(model, val_loader, device, out_channels, max_n_guesses, horizon, miss_threshold)
            curr_minade = metrics["minADE"]
            print(f"minADE:{metrics['minADE']:3f}, minFDE:{metrics['minFDE']:3f}, MissRate:{metrics['MR']:3f}")

            if curr_minade < best_minade:
                best_minade = curr_minade
                save_checkpoint(save_dir, model, optimizer, epoch, best_minade, date)
            model.train()
                
    # eval result on the identity dataset
    metrics = get_eval_metric_results_ebm(model, val_loader, device, out_channels, max_n_guesses, horizon, miss_threshold)
    curr_minade = metrics["minADE"]
    if curr_minade < best_minade:
        best_minade = curr_minade
        save_checkpoint(save_dir, model, optimizer, -1, best_minade, date)


# %%
