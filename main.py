import os

from sa_convlstm import SAConvLSTM
from convlstm import ConvLSTM
from utils import *

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys
import pickle
from tqdm import tqdm
import numpy as np
import math
import argparse
import json

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('-clstm',
                    '--convlstm',
                    action='store_true')
parser.add_argument('-saclstm',
                    '--saconvlstm',
                    action='store_true')
parser.add_argument('-pcrps',
                    '--prob_crps',
                    action='store_true')
parser.add_argument('-orth',
                    '--kernel_orth',
                    action='store_true')

parser.add_argument('--kernel_size',
                    default="(3, 3)",
                    type=str)
parser.add_argument('--bias',
                    default=True,
                    type=bool)
parser.add_argument('--hidden_dim',
                    default="(128, 128, 128, 128)",
                    type=str)
parser.add_argument('--attn_dim',
                    default=32,
                    type=int)
parser.add_argument('--dropout',
                    default=0.,
                    type=float)
parser.add_argument('--ssr_decay_rate',
                    default=0.8e-4,
                    type=float)

parser.add_argument('--input_dim',
                    default=5,
                    type=tuple)
parser.add_argument('--output_dim',
                    default=5,
                    type=bool)
parser.add_argument('--input_length',
                    default=7,
                    type=tuple)
parser.add_argument('--output_length',
                    default=5,
                    type=int)

parser.add_argument('--batch_size',
                    default=1,
                    type=int)
parser.add_argument('--learn_rate',
                    default=0.001,
                    type=float)
parser.add_argument('--weight_decay',
                    default=0,
                    type=float)
parser.add_argument('--num_epochs',
                    default=50,
                    type=int)
parser.add_argument('--early_stopping',
                    default=True,
                    type=bool)
parser.add_argument('--patience',
                    default=3,
                    type=int)
parser.add_argument('--gradient_clipping',
                    default=True,
                    type=bool)
parser.add_argument('--clipping_threshold',
                    default=1,
                    type=int)

parser.add_argument('--lambda_orth',
                    default=0.001,
                    type=float)
parser.add_argument('--orth_decay',
                    default=0,
                    type=int)

parser.add_argument('--data',
                    default="./WeatherBenchData/wthrbnch_air_gpt_sh_u_v_5.625deg_24.npy",
                    type=str)

parser.add_argument('--name',
                    default="minion",
                    type=str)

args = parser.parse_args()
args.kernel_size = eval(args.kernel_size)
args.hidden_dim = eval(args.hidden_dim)

# random_seed = 1234
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)

save_dir = './save_models/'
if args.convlstm: save_dir+="clstm_"
if args.saconvlstm: save_dir+="saclstm_"
if args.prob_crps: save_dir+="pcrps_"
if args.kernel_orth: save_dir+="orth_" + str(args.orth_decay) + "_"
if args.dropout>0: save_dir+="dropout_" + str(args.dropout) + "_"
save_dir+="epoch_"+str(args.num_epochs)+"_"
save_dir+= args.name + "/"

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
with open(save_dir+'args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

trainFolder = wb_dataset(root=args.data, dataset_type="train", frames_input=args.input_length,
                              frames_output=args.output_length, prob = args.prob_crps)

validFolder = wb_dataset(root=args.data, dataset_type="eval", frames_input=args.input_length,
                              frames_output=args.output_length, prob = args.prob_crps)

trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=True)

validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if args.convlstm and not args.prob_crps:
    network = ConvLSTM(args.input_dim, args.hidden_dim, args.output_dim,
                             args.kernel_size, device, dropout=args.dropout).to(device)
elif args.convlstm and args.prob_crps:
    network = ConvLSTM(2*args.input_dim, args.hidden_dim, 2*args.output_dim,
                       args.kernel_size, device, dropout=args.dropout).to(device)
elif args.saconvlstm and not args.prob_crps:
    network = SAConvLSTM(args.input_dim, args.hidden_dim, args.output_dim, args.attn_dim,
                         args.kernel_size, device, dropout=args.dropout).to(device)
else:
    network = SAConvLSTM(2*args.input_dim, args.hidden_dim, 2*args.output_dim, args.attn_dim,
                         args.kernel_size, device, dropout=args.dropout).to(device)


optimizer = torch.optim.Adam(network.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)    
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=0, verbose=True, min_lr=0.0001)

def train():
    cur_epoch = 0
    epoch_eval_loss = []
    epoch_eval_wb_loss = []
    count = 0
    best = math.inf
    ssr_ratio = 1
    for i in range(cur_epoch, args.num_epochs):
        print('\nepoch: {0}'.format(i))
        network.train()
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for j, mc in enumerate(t):
            if ssr_ratio > 0:
                ssr_ratio = max(ssr_ratio - args.ssr_decay_rate, 0)
                
            mc_pred = network(mc.float(), teacher_forcing=True, scheduled_sampling_ratio=ssr_ratio, train=True)
            optimizer.zero_grad()
            
            if args.prob_crps:
                loss = loss_prob(mc_pred, mc[:, 1:].to(device), args.output_dim) 
            else:
                loss = loss_mc(mc_pred, mc[:, 1:].to(device))
            
            if args.kernel_orth and args.saconvlstm:
                diff = orth_dist(network.layers[0].sa.conv_h.weight) + orth_dist(network.layers[0].sa.conv_m.weight) + orth_dist(network.layers[0].sa.conv_z.weight)
                diff += deconv_orth_dist(network.layers[0].sa.conv_output.weight, stride = 1)
                diff += orth_dist(network.layers[1].sa.conv_h.weight) + orth_dist(network.layers[1].sa.conv_m.weight) + orth_dist(network.layers[1].sa.conv_z.weight)
                diff += deconv_orth_dist(network.layers[1].sa.conv_output.weight, stride = 1)
                diff += orth_dist(network.layers[2].sa.conv_h.weight) + orth_dist(network.layers[2].sa.conv_m.weight) + orth_dist(network.layers[2].sa.conv_z.weight)
                diff += deconv_orth_dist(network.layers[2].sa.conv_output.weight, stride = 1)
                diff += orth_dist(network.layers[3].sa.conv_h.weight) + orth_dist(network.layers[3].sa.conv_m.weight) + orth_dist(network.layers[3].sa.conv_z.weight)
                diff += deconv_orth_dist(network.layers[3].sa.conv_output.weight, stride = 1)
                loss += args.lambda_orth*diff#/np.exp(args.orth_decay*i) 
            
            loss.backward()                                                                                             
            if args.gradient_clipping:
                nn.utils.clip_grad_norm_(network.parameters(), args.clipping_threshold)
            optimizer.step()

            if j % 2500 == 0:
                print('batch training loss: {:.5f}, ssr ratio: {:.4f}'.format(loss, ssr_ratio))

        # evaluation
        loss_mc_eval = infer(validLoader, args.input_length, network, args.output_dim, args.prob_crps)
        epoch_eval_loss.append(loss_mc_eval)
        loss_wb_eval = infer_WB(validFolder, validLoader, args.input_length, network, args.output_dim, args.prob_crps)
        epoch_eval_wb_loss.append(loss_wb_eval.detach().cpu().numpy())
        print('epoch eval loss:\nmc loss: {:.5f}'.format(loss_mc_eval))
        lr_scheduler.step(loss_mc_eval)
        if loss_mc_eval >= best:
            count += 1
            print('eval loss is not improved for {} epoch'.format(count))
        else:
            count = 0
            print('eval loss is improved from {:.5f} to {:.5f}, saving model'.format(best, loss_mc_eval))
            save_model(save_dir + str(i) + "_checkpoint.chk")
            best = loss_mc_eval

        if count == args.patience:
            print('early stopping reached, best loss is {:5f}'.format(best))
            break
    np.save(save_dir + "eval_loss.npy", np.array(epoch_eval_loss))
    np.save(save_dir + "eval_wb_loss.npy", np.array(epoch_eval_wb_loss))

def save_model(path):
    torch.save({'net': network.state_dict(),
                'optimizer': optimizer.state_dict()}, path)


if __name__ == "__main__":
    train()
