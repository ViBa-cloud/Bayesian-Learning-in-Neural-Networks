import torch
import torch.nn as nn
# import torchbnn as bnn
# from torchbnn.modules import BayesLinear
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='whitegrid', palette='colorblind')
from classes.layers import BayesLinearMixture
import numpy as np

def main():

    fig, ax = plt.subplots(layout='tight', figsize=(10, 6))
    palette = sns.color_palette('deep')
    alpha = 0.7

    ##################### BNN with Gaussian Prior ###############################
    # input_dim = 784
    # hidden_dim = 1200
    # num_classes = 10
    # log_sigma1 = -1    
    # sigma1 = 10 ** log_sigma1
    # model = nn.Sequential(
    #     bnn.BayesLinear(prior_mu=0, prior_sigma=sigma1, in_features=input_dim, out_features=hidden_dim, bias=False),
    #     nn.ReLU(),
    #     bnn.BayesLinear(prior_mu=0, prior_sigma=sigma1, in_features=hidden_dim, out_features=hidden_dim, bias=False),
    #     nn.ReLU(),
    #     bnn.BayesLinear(prior_mu=0, prior_sigma=sigma1, in_features=hidden_dim, out_features=num_classes, bias=False),
    # )
    # all_weights = []
    # directory = "./models/"
    # filename = os.path.join(directory, "gaussian-prior", "300-epochs", "model_20230407_024943_300")
    # model.load_state_dict(torch.load(filename))
    # model.train(False)
    # for m in model.modules():
    #     if isinstance(m, BayesLinear):
    #         weight = (m.weight_mu + torch.exp(m.weight_log_sigma) * torch.randn_like(m.weight_log_sigma)).data
    #         weight = list(weight.view(-1))
    #         all_weights.extend(weight)
    
    # fig, ax = plt.subplots(layout='tight')
    # ax.hist(all_weights)
    ###########################################################################################################
    ################################ Gaussian mixture prior ###################################################
    input_dim = 784
    hidden_dim = 1200
    num_classes = 10
    log_sigma1 = -1
    log_sigma2 = -6
    pi = 0.75
    sigma1 = 10 ** log_sigma1
    sigma2 = 10 ** log_sigma2
    model = nn.Sequential(
        BayesLinearMixture(prior_mu1=0, prior_sigma1=sigma1, prior_mu2=0, prior_sigma2=sigma2, in_features=input_dim, out_features=hidden_dim, pi=pi),
        nn.ReLU(),
        BayesLinearMixture(prior_mu1=0, prior_sigma1=sigma1, prior_mu2=0, prior_sigma2=sigma2, in_features=hidden_dim, out_features=hidden_dim, pi=pi),
        nn.ReLU(),
        BayesLinearMixture(prior_mu1=0, prior_sigma1=sigma1, prior_mu2=0, prior_sigma2=sigma2, in_features=hidden_dim, out_features=num_classes, pi=pi),
    )

    all_weights = []
    directory = "./models/"
    # filename = os.path.join(directory, "gaussian-mixture-prior","pi0.5", "lr1e-3", "sigma21e-6", "model_20230408_091731_300")
    filename = os.path.join(directory, "gaussian-mixture-prior", "model_20230411_090805_300")
    model.load_state_dict(torch.load(filename))
    model.train(False)

    for m in model.modules():
        if isinstance(m, BayesLinearMixture):
            # weight = (m.weight_mu + torch.exp(m.weight_log_sigma) * torch.randn_like(m.weight_log_sigma)).data
            eps = torch.torch.randn_like(m.weight_rho)
            weight = m.weight_mu + torch.log1p(torch.exp(m.weight_rho)) * eps
            weight = list(weight.data.view(-1))
            all_weights.extend(weight)
    
    # ax.hist(all_weights, bins=100, label='gaussian-mixture', density=True)
    all_weights = np.array(all_weights)
    sns.kdeplot(data=all_weights, ax=ax, label="Bayes by Backprop", fill=True, color=palette[2], alpha=alpha)

    ############################## vanilla SGD #####################################################################
    input_dim = 784
    hidden_dim = 1200
    num_classes = 10
    model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=False),
    )

    all_weights = []
    directory = "./models/"
    filename = os.path.join(directory, "vanilla-SGD", "with-scheduler", "lr0.05", "model_20230407_232022_300")
    model.load_state_dict(torch.load(filename))
    model.train(False)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            weight = list(m.weight.data.view(-1))
            all_weights.extend(weight)
    
    # ax.hist(all_weights, bins=100, label='vanilla-SGD')#, density=True)
    all_weights = np.array(all_weights)
    sns.kdeplot(data=all_weights, ax=ax, label='Vanilla SGD', fill=True, color=palette[0], alpha=alpha)

    ################################ SGD Dropout ########################################################
    input_dim = 784
    hidden_dim = 1200
    num_classes = 10
    model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=False),
    )

    all_weights = []
    directory = "./models/"
    filename = os.path.join(directory, "SGD-dropout", "with-scheduler", "model_20230407_191619_300")
    model.load_state_dict(torch.load(filename))
    model.train(False)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            weight = list(m.weight.data.view(-1))
            all_weights.extend(weight)
    
    # ax.hist(all_weights, bins=100, label='vanilla-SGD')#, density=True)
    all_weights = np.array(all_weights)
    sns.kdeplot(data=all_weights, ax=ax, label='Dropout', fill=True, color=palette[1], alpha=alpha)
    
    ax.set_xlim([-0.3, 0.3])
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
