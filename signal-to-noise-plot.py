import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='whitegrid', palette='colorblind')
from classes.layers import BayesLinearMixture
import numpy as np

def main():

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, layout='tight', figsize=(7, 7))
    palette = sns.color_palette('deep')
    alpha = 0.7

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

    signal_to_noise = []
    directory = "./models/"
    # filename = os.path.join(directory, "gaussian-mixture-prior", "averaging2", "pi0.5", "sigma21e-6", "model_20230408_123457_300")
    filename = os.path.join(directory, "gaussian-mixture-prior", "model_20230411_090805_300")
    model.load_state_dict(torch.load(filename))
    model.train(False)

    for m in model.modules():
        if isinstance(m, BayesLinearMixture):
            mu = m.weight_mu.data.numpy()
            rho = m.weight_rho.data.numpy()
            sigma = np.log1p(np.exp(rho))
            s2n = np.log10(np.abs(mu) / sigma)
            s2n = list(s2n.flatten())
            signal_to_noise.extend(s2n)
    
    print(len(signal_to_noise))
    signal_to_noise = np.array(signal_to_noise)
    sns.kdeplot(data=signal_to_noise, ax=ax1, fill=True, color=palette[2], alpha=alpha)
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Signal-to-Noise Ratio (dB)')
    
    # fig, ax2 = plt.subplots(figsize=(16,9), layout='tight')
    sns.kdeplot(data=signal_to_noise, ax=ax2, lw=3, color='black', cumulative=True)
    ax2.set_ylabel('CDF')
    ax2.set_xlabel('Signal-to-Noise Ratio (dB)')
    
    
    plt.show()
    
if __name__ == "__main__":
    main()
