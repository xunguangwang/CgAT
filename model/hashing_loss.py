'''
Reference: https://github.com/swuxyj/DeepHash-pytorch
'''

from turtle import forward
import torch
import random
import numpy as np
from scipy.linalg import hadamard
from .distributional_quantization_losses import *


def get_hashing_loss(method, num_train, bit, num_classes, batch_size, is_single_label=False):
    if method == 'DPSH':
        return DPSHLoss(num_train, bit, num_classes)
    elif method == 'HashNet':
        return HashNetLoss(num_train, bit, num_classes)
    elif method == 'DSDH':
        return DSDHLoss(num_train, bit, num_classes)
    elif method == 'DSDH-C':
        return DSDHCLoss(num_train, bit, num_classes)
    elif method == 'DCH':
        return DCHLoss(batch_size, bit)
    elif method == 'GreedyHash':
        return GreedyHashLoss(bit, num_classes)
    elif method == 'CSQ':
        return CSQLoss(bit, num_classes, is_single_label)
    elif method == 'CSQ-C':
        return CSQCLoss(bit, num_classes, is_single_label)
    else:
        return DPHLoss(num_train, bit, num_classes)


class DPHLoss(torch.nn.Module):
    '''
    DPH
    [TCYB 2020]: Adversarial Examples for Hamming Space Search
    '''
    def __init__(self, num_train, bit, n_class):
        super(DPHLoss, self).__init__()
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, n_class).float().cuda()
        self.bit = bit
    
    def preinstall(self, U, Y):
        self.U = U
        self.Y = Y

    def forward(self, u, y, ind):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        s = 2*s - 1
        inner_product = u @ self.U.t() / self.bit
        likelihood_loss = (inner_product - s)**2
        likelihood_loss = likelihood_loss.mean()

        return likelihood_loss


class DPSHLoss(torch.nn.Module):
    '''
    DPSH
    IJCAI 2016: Feature Learning based Deep Supervised Hashing with Pairwise Labels
    '''
    def __init__(self, num_train, bit, n_class, eta=0.001):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, n_class).float().cuda()
        self.eta = eta
    
    def preinstall(self, U, Y):
        self.U = U
        self.Y = Y

    def forward(self, u, y, ind):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + \
            inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        quantization_loss = self.eta * (u - u.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss


class HashNetLoss(torch.nn.Module):
    '''
    HashNet
    ICCV 2017: HashNet: Deep Learning to Hash by Continuation
    '''
    def __init__(self, num_train, bit, n_class, alpha=0.1, beta=1):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, n_class).float().cuda()

        self.alpha = alpha
        self.beta = beta
    
    def preinstall(self, U, Y):
        self.U = U
        self.Y = Y

    def forward(self, u, y, ind):
        # u = torch.tanh(self.beta * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = self.alpha * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


class DSDHLoss(torch.nn.Module):
    '''
    Hashing Method: DSDH
    [NIPS 2017]: Deep Supervised Discrete Hashing
    '''
    def __init__(self, num_train, bit, n_class, mu=1, nu=1, dcc_iter=10, eta=55):
        super(DSDHLoss, self).__init__()
        self.U = torch.zeros(bit, num_train).float().cuda()
        self.B = torch.zeros(bit, num_train).float().cuda()
        self.Y = torch.zeros(n_class, num_train).float().cuda()

        self.bit = bit
        self.mu = mu
        self.nu = nu
        self.dcc_iter = dcc_iter
        self.eta = eta
    
    def preinstall(self, U, Y):
        self.U = U.t()
        self.B = torch.sign(U.t())
        self.Y = Y.t()

    def forward(self, u, y, ind):
        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        inner_product = u @ self.U * 0.5
        s = (y @ self.Y > 0).float()

        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        # Classification loss
        cl_loss = (y.t() - self.W.t() @ self.B[:, ind]).pow(2).mean()

        # Regularization loss
        reg_loss = self.W.pow(2).mean()

        loss = likelihood_loss + self.mu * cl_loss + self.nu * reg_loss
        return loss

    def update_band_W(self):
        B = self.B
        for dit in range(self.dcc_iter):
            # W-step
            W = torch.inverse(B @ B.t() + self.nu / self.mu * torch.eye(self.bit).cuda()) @ B @ self.Y.t()

            for i in range(B.shape[0]):
                P = W @ self.Y + self.eta / self.mu * self.U
                p = P[i, :]
                w = W[i, :]
                W_prime = torch.cat((W[:i, :], W[i + 1:, :]))
                B_prime = torch.cat((B[:i, :], B[i + 1:, :]))
                B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

        self.B = B
        self.W = W


class DSDHCLoss(DSDHLoss):
    '''
    Hashing Method: DSDH-C
    [CVPR 2022]: 
    '''
    def __init__(self, num_train, bit, n_class, mu=1, nu=1, dcc_iter=10, eta=55, quantization_alpha=0.1):
        super(DSDHCLoss, self).__init__(num_train, bit, n_class, mu, nu, dcc_iter, eta)
        self.quantization_alpha = quantization_alpha

    def forward(self, u, y, ind):
        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        inner_product = u @ self.U * 0.5
        s = (y @ self.Y > 0).float()

        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        # Classification loss
        cl_loss = (y.t() - self.W.t() @ self.B[:, ind]).pow(2).mean()

        # Regularization loss
        reg_loss = self.W.pow(2).mean()

        quantization_loss = quantization_swdc_loss(u.view(u.size(0), -1))

        loss = likelihood_loss + self.mu * cl_loss + self.nu * reg_loss + self.quantization_alpha*quantization_loss
        return loss


class DCHLoss(torch.nn.Module):
    '''
    Hashing Method: DCH
    [CVPR 2018]: Deep Cauchy Hashing for Hamming Space Retrieval
    '''
    def __init__(self, batch_size, bit, gamma=20, lambda_=0.1):
        super(DCHLoss, self).__init__()
        self.one = torch.ones((batch_size, bit)).cuda()

        self.K = bit
        self.gamma = gamma
        self.my_lambda = lambda_
    
    def preinstall(self, U, Y):
        pass

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, ind):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss = cauchy_loss.mean() + self.my_lambda * quantization_loss.mean()

        return loss


class GreedyHashLoss(torch.nn.Module):
    '''
    Hashing Method: GreedyHash
    [NIPS 2018] Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN
    '''
    def __init__(self, bit, n_class, alpha=0.1):
        super(GreedyHashLoss, self).__init__()
        self.fc = torch.nn.Linear(bit, n_class, bias=False).cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        self.alpha = alpha
    
    def preinstall(self, U, Y):
        pass

    def forward(self, u, onehot_y, ind):
        b = GreedyHashLoss.Hash.apply(u)
        # one-hot to label
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        loss2 = self.alpha * (u.abs() - 1).pow(3).abs().mean()
        return loss1 + loss2

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


class CSQLoss(torch.nn.Module):
    '''
    Hashing Method: CSQ
    [CVPR 2020]: Central Similarity Quantization for Efficient Image and Video Retrieval
    '''
    def __init__(self, bit, n_class, single_label, my_lambda=1e-4):
        super(CSQLoss, self).__init__()
        self.hash_targets = self.get_hash_targets(n_class, bit).cuda()
        self.multi_label_random_center = torch.randint(2, (bit,)).float().cuda()

        self.criterion = torch.nn.BCELoss().cuda()

        self.is_single_label = single_label
        self.my_lambda = my_lambda
    
    def preinstall(self, U, Y):
        pass

    def forward(self, u, y, ind):
        # u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + self.my_lambda * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


class CSQCLoss(CSQLoss):
    '''
    Hashing Method: CSQ-C
    [CVPR 2022]: One Loss for Quantization: Deep Hashing with Discrete Wasserstein Distributional Matching
    '''
    def __init__(self, bit, n_class, single_label, my_lambda=1e-4, quantization_alpha=0.1):
        super(CSQCLoss, self).__init__(bit, n_class, single_label, my_lambda)
        # self.hash_targets = self.get_hash_targets(n_class, bit).cuda()
        # self.multi_label_random_center = torch.randint(2, (bit,)).float().cuda()

        # self.criterion = torch.nn.BCELoss().cuda()

        # self.is_single_label = single_label
        # self.my_lambda = my_lambda
        self.quantization_alpha = quantization_alpha
    
    def forward(self, u, y, ind):
        # u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        # Q_loss = (u.abs() - 1).pow(2).mean()
        quantization_loss = quantization_swdc_loss(u.view(u.size(0), -1))
        return center_loss + self.quantization_alpha*quantization_loss
