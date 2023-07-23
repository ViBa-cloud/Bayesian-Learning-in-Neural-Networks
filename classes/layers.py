import math
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

class BayesLinearMixture(Module):
    r"""
    Applies Bayesian Linear
    Arguments:
        prior_mu1 (Float): mean of prior normal distribution 1.
        prior_sigma1 (Float): sigma of prior normal distribution 1.
        prior_mu2 (Float): mean of prior normal distribution 2.
        prior_sigma2 (Float): sigma of prior normal distribution 2.
        pi (Float): The mixture coefficient
    .. note:: other arguments are following linear of pytorch 1.2.0.
    Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    """
    __constants__ = ['prior_mu1', 'prior_sigma1', 'prior_mu2', 'prior_sigma2', 'in_features', 'out_features']

    def __init__(self, prior_mu1, prior_sigma1, prior_mu2, prior_sigma2, pi, in_features, out_features):
        super(BayesLinearMixture, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu1 = prior_mu1
        self.prior_sigma1 = prior_sigma1
        self.prior_log_sigma1 = math.log(prior_sigma1)

        self.prior_mu2 = prior_mu2
        self.prior_sigma2 = prior_sigma2
        self.prior_log_sigma2 = math.log(prior_sigma2)

        self.pi = pi

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(self.prior_log_sigma1)
  
    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        
    def unfreeze(self) :
        self.weight_eps = None
            
    def forward(self, input):
        r"""
        Overriden.
        """
        eps = torch.torch.randn_like(self.weight_rho)
        self.weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps

        return F.linear(input, self.weight, bias=None)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu1={}, prior_sigma1={}, prior_mu2={}, prior_sigma2={}, in_features={}, out_features={}'.format(self.prior_mu1, self.prior_sigma1, self.prior_mu2, self.prior_sigma2, self.in_features, self.out_features)
  