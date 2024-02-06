import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sqrtm import sqrtm


def ws_dist(nn_output, label):
    nn_mid=torch.tensor([[(nn_output[0].item()), 0], [0, (nn_output[1].item())]])
    s = torch.sin(nn_output[4])
    c = torch.cos(nn_output[4])
    nn_rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])
    nn_mu=torch.tensor([[nn_output[2].item()],[nn_output[3].item()]])
    nn_cov_sqr=torch.mm(torch.mm(nn_rot.double(),nn_mid.double()),torch.transpose(nn_rot.double(),0,1))
    nn_cov=torch.mm(nn_cov_sqr,nn_cov_sqr)

    label_mid = torch.tensor([[(label[0].item()), 0], [0, (label[1].item())]])
    s = torch.sin(label[4])
    c = torch.cos(label[4])
    label_rot = torch.stack([torch.stack([c, -s]),
                          torch.stack([s, c])])
    label_mu = torch.tensor([[label[2].item()], [label[3].item()]])
    label_cov_sqr = torch.mm(torch.mm(label_rot.double(), label_mid.double()), torch.t(label_rot.double()))
    label_cov=torch.mm(label_cov_sqr,label_cov_sqr)

    dist=torch.sum(torch.square(nn_mu-label_mu))+torch.trace(nn_cov+label_cov-2*sqrtm(torch.mm(torch.mm(nn_cov_sqr, label_cov),nn_cov_sqr)))
    if torch.isnan(dist):
        print("loss=nan")
        exit()
    else:
        return dist


def ws_dist_batch(nn_output, label):
    batch_size=nn_output.size(0)
    distances=torch.zeros(batch_size,1)
    for k in range(batch_size):
        nn_mid = torch.tensor([[(nn_output[k][0].item()), 0], [0, (nn_output[k][1].item())]])
        s = torch.sin(nn_output[k][4])
        c = torch.cos(nn_output[k][4])
        nn_rot = torch.stack([torch.stack([c, -s]),
                               torch.stack([s, c])])
        nn_mu = torch.tensor([[nn_output[k][2].item()], [nn_output[k][3].item()]])
        nn_cov_sqr = torch.mm(torch.mm(nn_rot.double(), nn_mid.double()), torch.transpose(nn_rot.double(), 0, 1))
        nn_cov = torch.mm(nn_cov_sqr, nn_cov_sqr)

        label_mid = torch.tensor([[(label[k][0].item()), 0], [0, (label[k][1].item())]])
        s = torch.sin(label[k][4])
        c = torch.cos(label[k][4])
        label_rot = torch.stack([torch.stack([c, -s]),
                                 torch.stack([s, c])])
        label_mu = torch.tensor([[label[k][2].item()], [label[k][3].item()]])
        label_cov_sqr = torch.mm(torch.mm(label_rot.double(), label_mid.double()), torch.t(label_rot.double()))
        label_cov = torch.mm(label_cov_sqr, label_cov_sqr)

        dist = torch.sum(torch.square(nn_mu - label_mu)) + torch.trace(
            nn_cov + label_cov - 2 * sqrtm(torch.mm(torch.mm(nn_cov_sqr, label_cov), nn_cov_sqr)))
        distances[k]=dist
        mean=torch.mean(distances)
    return mean

def weightedMSE(output,label):
    mean_all=0
    for l in range(len(output)):
        mean_s= (torch.square(output[l][0]-label[l][0])+torch.square(output[l][1]-label[l][1])
            +torch.square(output[l][2]-label[l][2])+torch.square(output[l][3]-label[l][3])
            +torch.square(output[l][4]-label[l][4])*10)
        mean_all=mean_all+mean_s
    mean=mean_all/len(output)
    return mean

def weightedMSEtensor(output, label):
    A=[]
    B=[]
    H=[]
    K=[]
    Tau=[]
    for l in range(len(output)):
        A.append((output[l][0]-label[l][0]).item()**2)
        B.append((output[l][1]-label[l][1]).item()**2)
        H.append((output[l][2]-label[l][2]).item()**2)
        K.append((output[l][3]-label[l][3]).item()**2)
        Tau.append((output[l][4]-label[l][4]).item()**2)
        tensor=[A,B,H,K,Tau]

    return tensor


def plot_losses(v_loss,t_loss,A_list,B_list,H_list,K_list,tau_list):
    val_losses=v_loss
    train_losses=t_loss
    plt.figure()
    plt.plot(np.linspace(0, len(val_losses), len(val_losses)), val_losses, label='V_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure
    plt.plot(np.linspace(0, len(train_losses), len(train_losses)), train_losses, label='T_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0, len(tau_list), len(tau_list)), tau_list, label='Tau')
    plt.xlabel('Epoch')
    plt.ylabel('tau')
    plt.title('Tau')
    plt.legend()
    plt.grid(True)

    plt.plot(np.linspace(0, len(A_list), len(A_list)), A_list, label='A')
    plt.xlabel('Epoch')
    plt.ylabel('A')
    plt.title('A')
    plt.legend()
    plt.grid(True)

    plt.plot(np.linspace(0, len(B_list), len(B_list)), B_list, label='B')
    plt.xlabel('Epoch')
    plt.ylabel('B')
    plt.title('B')
    plt.legend()
    plt.grid(True)

    plt.plot(np.linspace(0, len(H_list), len(H_list)), H_list, label='H')
    plt.xlabel('Epoch')
    plt.ylabel('H')
    plt.title('H')
    plt.legend()
    plt.grid(True)

    plt.plot(np.linspace(0, len(K_list), len(K_list)), K_list, label='K')
    plt.xlabel('Epoch')
    plt.ylabel('K')
    plt.title('K')
    plt.legend()
    plt.grid(True)
    plt.show()

def tau_loss(output, label):
    mean_all = 0
    for l in range(len(output)):
        mean_s = (torch.square(output[l][4] - label[l][4]) * 10)
        mean_all = mean_all + mean_s
        mean = mean_all / len(output)
    return mean

def superellipse(A, B, H, K, tau, n):
    angles = np.linspace(0, 2*np.pi, 1000)
    cos_tau = np.cos(tau)
    sin_tau = np.sin(tau)
    x = A * np.sign(np.cos(angles)) * np.abs(np.cos(angles)) ** (2/n) * cos_tau - B * np.sign(np.sin(angles)) * np.abs(np.sin(angles)) ** (2/n) * sin_tau + H
    y = A * np.sign(np.cos(angles)) * np.abs(np.cos(angles)) ** (2/n) * sin_tau + B * np.sign(np.sin(angles)) * np.abs(np.sin(angles)) ** (2/n) * cos_tau + K
    return x, y

def superellipse_sampler_1(A, B, H, K, tau, q, angles):
    q_dists = (abs(np.cos(angles + tau))**q + abs(np.sin(angles + tau))**q)**(-1/q)
    x = q_dists * np.cos(angles + tau) * A * np.cos(tau) - q_dists * np.sin(angles + tau) * B * np.sin(tau) + H
    y = q_dists * np.cos(angles + tau) * A * np.sin(tau) + q_dists * np.sin(angles + tau) * B * np.cos(tau) + K
    return x, y

def testbed_loss(expanded_output,expanded_label):
    loss_A = (expanded_output[0]-expanded_label[0])**2
    loss_B = (expanded_output[1]-expanded_label[1])**2
    loss_H = (expanded_output[2]-expanded_label[2])**2
    loss_K = (expanded_output[3]-expanded_label[3])**2
    loss_tau = (expanded_output[4]-expanded_label[4])**2
    loss_Q = (expanded_output[5]-expanded_label[5])**2
    total_loss=loss_A+loss_B+loss_H+loss_K+loss_Q+loss_tau
    loss_list=[loss_A,loss_B,loss_H,loss_K,loss_tau,loss_Q,total_loss]
    return loss_list
    