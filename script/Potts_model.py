__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/13 23:43:38"

import numpy as np
import pickle
import torch
import torch.nn as nn
from scipy import optimize
from sys import exit
import sys
import timeit
import argparse
import subprocess

parser = argparse.ArgumentParser(description = "Learn a Potts model using Multiple Sequence Alignment data.")
parser.add_argument("--input_dir",
                    help = "input directory where the files seq_msa_binary.pkl, seq_msa.pkl, seq_weight.pkl are.")
parser.add_argument("--max_iter",
                    help = "The maximum num of iteratioins in L-BFGS optimization.",
                    type = int)
parser.add_argument("--weight_decay",
                    help = "weight decay factor of L2 penalty",
                    type = float)
parser.add_argument("--output_dir",
                    help = "output directory for saving the model")
args = parser.parse_args()

## read msa
msa_file_name = args.input_dir + "/seq_msa_binary.pkl"
with open(msa_file_name, 'rb') as input_file_handle:
    seq_msa_binary = pickle.load(input_file_handle)
seq_msa_binary = seq_msa_binary.astype(np.float32)

msa_file_name = args.input_dir + "/seq_msa.pkl"
with open(msa_file_name, 'rb') as input_file_handle:
    seq_msa = pickle.load(input_file_handle)
seq_msa = seq_msa.astype(np.float32)

weight_file_name = args.input_dir + "/seq_weight.pkl"
with open(weight_file_name, 'rb') as input_file_handle:
    seq_weight = pickle.load(input_file_handle)
seq_weight = seq_weight.astype(np.float32)

seq_msa_binary = torch.from_numpy(seq_msa_binary).cuda()
seq_weight = torch.from_numpy(seq_weight).cuda()
weight_decay = float(args.weight_decay)

## pseudolikelihood method for Potts model
_, len_seq, K = seq_msa_binary.shape
num_node = len_seq * K

seq_msa_binary = seq_msa_binary.reshape(-1, num_node)
seq_msa_idx = torch.argmax(seq_msa_binary.reshape(-1,K), -1)

# h = seq_msa_binary.new_zeros(num_node, requires_grad = True)
# J = seq_msa_binary.new_zeros((num_node, num_node), requires_grad = True)
J_mask = seq_msa_binary.new_ones((num_node, num_node))
for i in range(len_seq):
    J_mask[K*i:K*i+K, K*i:K*i+K] = 0


def calc_loss_and_grad(parameter):
    parameter = parameter.astype(np.float32)
    J = parameter[0:num_node**2].reshape([num_node, num_node])
    h = parameter[num_node**2:]
    
    J = torch.tensor(J, requires_grad = True, device = seq_msa_binary.device)
    h = torch.tensor(h, requires_grad = True, device = seq_msa_binary.device)
    
    logits = torch.matmul(seq_msa_binary, J*J_mask) + h
    cross_entropy = nn.functional.cross_entropy(
        input = logits.reshape((-1,K)),
        target = seq_msa_idx,
        reduce = False)
    cross_entropy = torch.sum(cross_entropy.reshape((-1,len_seq)), -1)
    cross_entropy = torch.sum(cross_entropy*seq_weight)
    loss = cross_entropy + weight_decay*torch.sum((J*J_mask)**2)
    
    loss.backward()

    grad_J = J.grad.cpu().numpy().copy()
    grad_h = h.grad.cpu().numpy().copy()

    grad = np.concatenate((grad_J.reshape(-1), grad_h))
    grad = grad.astype(np.float64)
    return loss.item(), grad

init_param = np.zeros(num_node*num_node + num_node)
#loss, grad = calc_loss_and_grad(init_param)
param, f, d = optimize.fmin_l_bfgs_b(calc_loss_and_grad, init_param, iprint = True, maxiter = args.max_iter)
J = param[0:num_node**2].reshape([num_node, num_node])
h = param[num_node**2:]

## save J and h
model = {}
model['len_seq'] = len_seq
model['K'] = K
model['num_node'] = num_node
model['weight_decay'] = args.weight_decay
model['max_iter'] = args.max_iter
model['J'] = J
model['h'] = h

subprocess.run(['mkdir', '-p', args.output_dir])
with open("{}/model_weight_decay_{:.3f}.pkl".format(args.output_dir, args.weight_decay), 'wb') as output_file_handle:
    pickle.dump(model, output_file_handle)
