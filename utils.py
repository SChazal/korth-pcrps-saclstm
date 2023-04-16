import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

class wb_dataset(Dataset):
    def __init__(self, root, dataset_type, frames_input, frames_output, prob = False):
        super().__init__()
        self.data = np.load(root)
        self.dataset_type = dataset_type
        
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.frames_total = self.frames_input + self.frames_output
        
        self.days_total = self.data.shape[0]
        self.days_eval = 366 #2016
        self.days_test = 365 + 365 #2017 and 2018
        self.days_train = self.days_total-self.days_eval-self.days_test
        
        self.train_start = 0
        self.eval_start = self.days_train 
        self.test_start = self.days_train + self.days_eval
        
        self.long_mean = np.mean(self.data[:self.eval_start], axis = 0)
        self.long_std = np.std(self.data[:self.eval_start], axis = 0)
        self.data = (self.data-self.long_mean)/self.long_std
        
        self.prob = prob

    def __len__(self):
        if self.dataset_type=="train":
            return self.days_train - self.frames_total
        elif self.dataset_type=="eval":
            return self.days_eval - self.frames_total
        else:
            return self.days_test - self.frames_total

    def __getitem__(self, idx):
        if self.dataset_type=="train":
            item = self.data[idx: (idx + self.frames_total), ...]
        elif self.dataset_type=="eval":
            item = self.data[self.eval_start + idx: (self.eval_start + idx + self.frames_total), ...]
        else:
            item = self.data[self.test_start + idx: (self.test_start + idx + self.frames_total), ...]
        
        if self.prob: item = np.concatenate((item, np.zeros(item.shape)), axis=1) 
        return item
        
def loss_mc(y_pred, y_true):
    # y_pred/y_true (N, 5+7-1, 5, 32, 64)
    rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3, 4])
    rmse = torch.sum(rmse.sqrt().mean(dim=0))
    return rmse

def loss_prob(y_pred, y_true, output_channels):
    # y_pred/y_true (N, 5+7-1, 5+5, 32, 64)
    dist = Normal(loc=0, scale=1)
    x = y_true[:, :, :output_channels, ...]
    mu = y_pred[:, :, :output_channels, ...]
    sig = torch.sqrt(y_pred[:, :, output_channels:, ...]**2)
#         sig = torch.sqrt(torch.exp(y_pred[:, :, 5:, ...]))
    sx = (x - mu) / sig
    pdf = torch.exp(dist.log_prob(sx))
    cdf = dist.cdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    crps = torch.mean(crps, dim=[2, 3, 4])
    crps = torch.sum(crps.mean(dim=0))
    return crps

def loss_WB(y_pred, y_true):
    # y_pred/y_true (N, 5+7-1, 5, 32, 64)
    lat_weights = np.cos(np.linspace(-90+5.625/2, 90-5.625/2, 32)*np.pi/180)
    lat_weights = lat_weights/np.mean(lat_weights)
    lat_weights = torch.from_numpy(lat_weights).to(y_pred.device)
    wse = (y_pred - y_true)**2 * lat_weights[:, None]
    rmse = torch.mean(wse, dim=[3, 4]).sqrt().mean(dim=0)
    return rmse

def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).to(mat.device))
        
def deconv_orth_dist(kernel, stride = 1, padding = 1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).to(kernel.device)
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).to(kernel.device)
    return torch.norm( output - target )

def predictions(dataloader, input_length, network, output_channels, prob=False):
    mc_pred = []
    with torch.no_grad():
        for mc in dataloader:
            mc = network(mc[:, :input_length, ...].float(), train=False)
            if prob: mc_pred.append(mc[:, :, :output_channels, ...])
            else: mc_pred.append(mc)
    return torch.cat(mc_pred, dim=0)

def infer(dataloader, input_length, network, output_channels, prob=False):
    network.eval()
    with torch.no_grad():
        mc_pred = predictions(dataloader, input_length, network, output_channels, prob)
        mc_true = []
        for mc in dataloader:
            if prob: mc_true.append(mc[:, input_length:, :output_channels, ...].to(network.device))
            else: mc_true.append(mc[:, input_length:, ...].to(network.device))
        mc_true = torch.cat(mc_true, dim=0)
    return loss_mc(mc_pred, mc_true).item()
    
def infer_WB(dataset, dataloader, input_length, network, output_channels, prob=False):
    network.eval()
    with torch.no_grad():
        mc_pred = predictions(dataloader, input_length, network, output_channels, prob)
        mc_true = []
        for mc in dataloader:
            if prob: mc_true.append(mc[:, input_length:, :output_channels, ...].to(network.device))
            else: mc_true.append(mc[:, input_length:, ...].to(network.device))
        mc_true = torch.cat(mc_true, dim=0)
        long_std = torch.from_numpy(dataset.long_std).to(network.device)
        long_mean = torch.from_numpy(dataset.long_mean).to(network.device)
        mc_pred = mc_pred*long_std + long_mean
        mc_true = mc_true*long_std + long_mean
    return loss_WB(mc_pred, mc_true)

