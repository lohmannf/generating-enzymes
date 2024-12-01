import abc
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.distributions import OneHotCategorical
from typing import Callable
from omegaconf import DictConfig

from genzyme.data.utils import onehot2aa
from genzyme.data import ProteinTokenizer

class EnergyBasedModel(abc.ABC, torch.nn.Module):
    '''
    Base class for energy-based models.
    Implements sampling and utility methods
    '''

    @abc.abstractmethod
    def __init__(self, L, d, seed=31):

        super().__init__()

        self.set_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.L = L
        self.d = d
        self.tokenizer = ProteinTokenizer(has_sep=False)
        self.projector = lambda x, is_onehot: x

    @abc.abstractmethod
    def forward(self):
        '''
        Compute the negative energy
        '''
        pass


    def cross_entropy(self, batch, logits, weight):
        '''
        Calculate the unregularized pseudolikelihood for a batch of sequences
        given the token-wise logits and sequence weights
        '''

        if len(batch.size()) == 3:
            #one-hot encoding
            seq_as_int = torch.argmax(batch.reshape(-1,self.d), -1)
        else:
            seq_as_int = batch.reshape(-1)

        cross_entropy = torch.nn.functional.cross_entropy(
            input = logits.reshape((-1, self.d)),
            target = seq_as_int,
            reduction = "none")
        cross_entropy = torch.sum(cross_entropy.reshape((-1, self.L)), -1)

        return torch.sum(cross_entropy*weight)


    def set_seed(self, seed):
        '''
        Set torch and numpy seeds
        '''
        torch.manual_seed(seed)
        np.random.seed(seed)

    def set_device(self, device):
        self.device = device


    def get_random_seq(self):
        '''
        Get a sequence with each position drawn uniformly at random
        '''

        seq = torch.zeros(self.L, self.d)

        for i in range(self.L):
            idx = np.random.randint(self.d)
            seq[i,idx] = 1.

        return seq.double().unsqueeze(0)


    def sample_H1(self, x):
        '''
        Draw a sequence uniformly at random from the Hamming ball
        with radius 1 around x

        Parameters
        ----------
        x: torch.Tensor
            One-hot encoded sequence of shape (1,L,d)

        Returns
        -------
        p: float
            The likelihood of the new sample

        x_p: torch.Tensor
            One-hot encoded next sequence
        '''
        
        p = 1/(self.L * (self.d-1) + 1) #LH of any event in H1
        x_p = x.clone().detach()
        
        if np.random.rand() >= p:

            pos = np.random.randint(self.L)
            aa = np.random.choice(np.arange(self.d)[~x_p[:,pos].squeeze().bool().numpy()])
            
            x_p[:, pos, :] = 0
            x_p[:, pos, aa] = 1
            
        return p, x_p
    

    def diff(self, x: torch.Tensor, T: float = 2.0):
        '''
        Approximate the energy difference with the method from Grathwohl et al. (GWG)

        Parameters
        ----------
        x: torch.Tensor
            The current state

        T: float
            Temperature parameter

        Returns
        -------
        diff: torch.Tensor
            Approximate energy difference between x and all states in the Hamming 
            ball of radius 1
        '''
        x_em = x.requires_grad_()
        gx = torch.autograd.grad(self(x_em).sum(), x)[0]
        gx_cur = (gx * x).sum(-1)[:, :, None]
        return (gx - gx_cur)/T


    def generate(self,
                 cfg: DictConfig,
                 x0: torch.Tensor = None
                ):
        ''' 
        Generate new samples with MCMC using the current model parameters.
        Adapted from https://github.com/wgrathwohl/GWG_release/blob/main/samplers.py

        Parameters
        ----------
        cfg: DictConfig
            Dict-style config containing all the generation hyperparameters
        
        x0: torch.Tensor
            Starting state as one-hot encoded amino acid sequences (1xLxd).
            Chain will use random sequence if x0 is None.

        Returns
        -------
        seqs: list
            The generated sequences, will be empty if cfg.generation.keep_in_memory = False
        '''

        is_training = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        self.set_seed(cfg.generation.seed)

        if x0 is None:
            x0 = self.get_random_seq()
        
        x = []
        buf = []
        x0.requires_grad = True
        x_curr = x0.detach()
        ct = 0

        for i in tqdm(range(cfg.generation.n_burnin + cfg.generation.n_episodes)):

            if cfg.generation.sampler == "local":
                forward_delta = self.diff(x_curr, cfg.generation.temp_proposal)
                # make sure we dont choose to stay where we are!
                forward_logits = forward_delta - 1e9 * x_curr
                # flatten to only sample a single change
                cd_forward = OneHotCategorical(logits=forward_logits.view(x_curr.size(0), -1))
                changes = cd_forward.sample()
                # compute probability of sampling this change
                lp_forward = cd_forward.log_prob(changes)
                # reshape to (1, L, d)
                changes_r = changes.view(x_curr.size())
                # get binary indicator (1, L) indicating which dim was changed
                changed_ind = changes_r.sum(-1)
                # mask out changed dim and add in the change
                x_delta = x_curr.clone() * (1. - changed_ind[:, :, None]) + changes_r
                reverse_delta = self.diff(x_delta, cfg.generation.temp_proposal)
                reverse_logits = reverse_delta - 1e9 * x_delta
                cd_reverse = OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
                reverse_changes = x_curr * changed_ind[:, :, None]

                lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

                m_term = (self(self.projector(x_delta)).squeeze() - self(self.projector(x_curr)).squeeze()) / cfg.generation.temp_marginal
                la = m_term + lp_reverse - lp_forward
                print(m_term, lp_reverse, lp_forward, la, flush=True)

            elif cfg.generation.sampler == "uniform":
                _, x_p = self.sample_H1(x_curr)

                la = (self(self.projector(x_p)).squeeze()-self(self.projector(x_curr)).squeeze()) / cfg.generation.temp_marginal

                #print(f'Acceptance log-probability: {la}')

            else:
                raise NotImplementedError(f"Unknown sampler {cfg.generation.sampler}")
            
            accept = (la.exp() > torch.rand_like(la)).float()
            #print('Accepted' if accept else 'Rejected', flush=True)
            if not accept.any():
                continue

            if cfg.generation.sampler == "local":
                x_p = (x_delta * accept[:, None, None] + x_curr * (1. - accept[:, None, None])).detach()
            
            if i >= cfg.generation.n_burnin:

                ct += 1
                buf.append(torch.argmax(x_p.squeeze(), -1).detach().cpu())
                if cfg.generation.keep_in_memory:
                    x.append(torch.argmax(x_p.squeeze(), -1).detach().cpu())
                
                if len(buf) == cfg.generation.batch_size:
                    with open(cfg.generation.output_file, "a") as file:
                        print(buf[0].size())
                        print(torch.stack(buf).size())
                        for seq in self.tokenizer.batch_decode(torch.stack(buf)):
                            file.write(f'>\n{seq}\n')
                    buf = []

            x_curr = x_p
        
        if len(buf) > 0:
            with open(cfg.generation.output_file, "a") as file:
                for seq in self.tokenizer.batch_decode(torch.stack(buf)):
                    file.write(f'>\n{seq}\n')

        if len(x) > 0:
            x = self.tokenizer.batch_decode(torch.stack(x))

        print(f"Generated {ct} sequences\nAcceptance rate: {np.round(ct / cfg.generation.n_episodes, 3)}")

        self.train(is_training)
        for p in self.parameters():
            p.requires_grad = True

        return x
