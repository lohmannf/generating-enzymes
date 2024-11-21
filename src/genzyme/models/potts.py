import torch
import numpy as np
from torchmin import minimize
import wandb
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf, DictConfig

from genzyme.models.basemodel import BaseModel
from genzyme.models.ebm import EnergyBasedModel
from genzyme.data.utils import aa2int_single


class PottsModel(EnergyBasedModel, BaseModel):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg.model.L, cfg.model.d, cfg.model.seed)

        self.max_energy=cfg.model.max_energy
        self.n = 0
        self.lambda_J = cfg.model.lambda_J * (self.L - 1) * self.d
        self.lambda_h = cfg.model.lambda_h
        # Initialize J and h randomly
        self.J =  torch.nn.Parameter(torch.randn(self.L, self.L, self.d, self.d).double(), True)
        self.h =  torch.nn.Parameter(torch.randn(self.L, self.d).double(), True)

        self.to(self.device)

    def pseudo_lh_param(self, params: torch.Tensor):
        '''
        Objective function for pytorch-minimize.
        Computes the negative log-pseudolikelihood of the data stored internally

        Parameters
        ----------
        params: torch.Tensor
            Model parameters stored in a single tensor of shape (LxLx(d+1)xd)

        Returns
        --------
        loss: torch.Tensor
            Loss at the current iteration
        '''
         
        J = params[:, :self.L, :, :]
        h = params[:, self.L, 0, :].reshape((self.L*self.d))

        mask = torch.ones_like(J).to(self.device)
        for i in range(self.L):
            mask[i,i,:,:] = 0.
        J = J*mask

        J = J.transpose(1,2).reshape((self.L*self.d, self.L*self.d))
        
        logits = torch.matmul(self.x.reshape((-1, self.d*self.L)), J) + h
        cross_entropy = self.cross_entropy(self.x, logits, self.weight)
        loss = cross_entropy + self.lambda_J * torch.norm(J, p='fro') ** 2 + self.lambda_h * torch.norm(h) ** 2

        return loss
    

    def pseudo_lh(self, x: torch.Tensor, weight: torch.Tensor = None):
        '''
        Loss function for optimization with SGD
        Computes the log-pseudolikelihood of the data x using the parameters stored internally

        Parameters
        ----------
        x: torch.Tensor
            A batch of data

        weight: torch.Tensor
            Weight of the samples in x

        Returns
        -------
        loss: torch.Tensor
            The loss of the current batch
        '''

        if weight is None:
            weight = torch.ones((x.size()[0],)).to(self.device)

        J = self.J

        mask = torch.ones_like(J).to(self.device)
        for i in range(self.L):
            mask[i,i,:,:] = 0.
        J = J*mask

        J = J.transpose(1,2).reshape((self.L*self.d, self.L*self.d))
        h = self.h.reshape((self.L*self.d))
        
        logits = torch.matmul(x.reshape((-1, self.d*self.L)), J) + h
        cross_entropy = self.cross_entropy(x, logits, weight)
        loss = cross_entropy + self.lambda_J * torch.norm(J, p='fro') ** 2 + self.lambda_h * torch.norm(h) ** 2

        return loss


    def get_coupling_info(self, apc: bool = False):
        '''
        Compute a contact map out of the J parameter stored internally

        Parameters
        ----------
        apc: bool
            Whether to perform average product correction, default = False

        Returns
        -------
        contact_map: torch.Tensor
            The contact map of shape (L,L), on the same device as the model
        '''

        J = self.J + torch.transpose(self.J, 0,1) /2.0
        coupling_raw = torch.norm(J, p='fro', dim=(-2, -1))
        # Ensuring diagonal elements are not affected as they do not represent interactions
        coupling_raw.fill_diagonal_(0)
        
        if apc:
            sum_rows = coupling_raw.sum(axis=1)
            sum_cols = coupling_raw.sum(axis=0)

            # Calculate the overall average score
            average_all = coupling_raw.mean()

            # Calculate the outer product of the sum of rows and the sum of columns
            correction_matrix = torch.outer(sum_rows, sum_cols) / coupling_raw.sum() #(average_all * coupling_raw.shape[0])

            # Apply the correction
            corrected_scores = coupling_raw - correction_matrix
            corrected_scores.fill_diagonal_(0)
        else:
            corrected_scores= coupling_raw

        return corrected_scores.detach()
     

    def run_training(self, train_data, test_data, cfg: DictConfig):
        """
        Train the model using either quadratic programming or SGD
        """

        if not os.path.exists(cfg.training.work_dir):
            os.mkdir(cfg.training.work_dir)
            os.mkdir(os.path.join(cfg.training.work_dir, "snapshots"))

        OmegaConf.save(os.path.join(cfg.training.work_dir, "config.yaml"), resolve=True)

        if cfg.training.optimizer.method == "l-bfgs":
            params = torch.randn(size = (self.L, self.L + 1, self.d, self.d)).double()
            self.x = next(iter(train_data))
            self.x.requires_grad = False
            self.weight = torch.ones(self.x.size()[0], requires_grad = False)

            #it = 0
            def cbk(x):
                #nonlocal it
                #it += 1
                #print(f"Iteration {it} finished", flush=True)
                print(" ", flush=True) # force flushing of intermediate results


            result = minimize(self.pseudo_lh_param,
                            params,
                            method='l-bfgs', 
                            disp=2,
                            max_iter = cfg.training.optimizer.max_iter,
                            options={"history_size": cfg.training.optimizer.history_size},
                            callback = cbk)

            params = result.x
            self.J = torch.nn.Parameter(params[:, 0:self.L, :, :], False)
            self.h = torch.nn.Parameter(params[:, self.L, 0, :], False)

        elif cfg.training.optimizer.method == "adam":
            
            if cfg.training.use_wandb:
                run = wandb.init(**OmegaConf.to_object(cfg.training.wandb))

            optimizer = torch.optim.Adam(self.parameters(), lr = cfg.training.optimizer.lr, 
                                         betas = (cfg.training.optimizer.beta1, cfg.training.optimizer.beta2))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=cfg.training.optimizer.gamma)

            step = 0

            try:
                for i in range(cfg.training.n_epochs):

                    epoch_loss = 0

                    for batch in iter(train_data):
                        loss = self.pseudo_lh(batch.to(self.device))   
                        optimizer.zero_grad()    
                        loss.backward()
                        optimizer.step()

                        step += 1
                        epoch_loss += loss.item()

                    if scheduler.get_last_lr()[0] > cfg.training.optimizer.lr_limit:
                        scheduler.step()

                    epoch_loss /= len(train_data)

                    if (i) % cfg.training.log_freq == 0:

                        test_loss = 0  # will be lower than training loss because we calculate it after optim step
                        
                        with torch.no_grad():
                            for batch in iter(test_data):
                                test_loss += self.pseudo_lh(batch.to(self.device)).item()
                            
                        test_loss /= len(test_data)
                        if cfg.training.use_wandb:
                            run.log({"epoch": i+1, "train_step": step, "training/loss": epoch_loss, "evaluation/loss": test_loss, "lr": scheduler.get_last_lr()[0]})
                        print({"epoch": i+1, "train_step": step, "training/loss": epoch_loss, "evaluation/loss": test_loss, "lr": scheduler.get_last_lr()[0]}, flush=True)
                    
                    if (i) % cfg.training.contact_map_snapshot_freq == 0:
                        
                        with torch.no_grad():
                            cm = self.get_coupling_info(apc=True)

                        fig, ax = plt.subplots()
                        ax.imshow(cm.cpu().detach().numpy())
                        ax.set_title(f"Epoch {i+1} Train step {step}")
                        plt.savefig(os.path.join(cfg.training.work_dir, f"snapshots/step_{step}.png"), dpi=500)
                        plt.close()

            finally:
                if cfg.training.use_wandb:
                    wandb.finish()
        
        torch.save(self, os.path.join(cfg.training.work_dir, "model.pt"))
        


    def forward(self, seqs: torch.Tensor, w: torch.Tensor = None):
        ''' 
        Get the negative energy of one-hot encoded sequences

        Parameters
        ---------
        seq: torch.Tensor
            One-hot encoded sequences of shape (nxLxd)

        Returns
        -------
        nenergy: torch.Tensor
            The negative energy of the sample
        '''
        
        if len(seqs.size()) == 2:
            seqs = seqs.unsqueeze(0)

        # mask diagonal entries (no auto-correlation)
        mask = torch.ones_like(self.J)
        for i in range(self.L):
            mask[i,i,:,:] = 0.
        J = self.J * mask

        if w is None:
            w = torch.ones(seqs.size()[0])

        nenergy = - w * (torch.einsum('ijkl,pik,pjl->p', J, seqs, seqs) \
                            + torch.einsum('ik,pik->p', self.h, seqs))
        
        if self.max_energy is not None:
            torch.clamp(nenergy, max = None, min = -self.max_energy)
        
        return nenergy


if __name__ == "__main__":

    from genzyme.models import modelFactory
    from genzyme.data import loaderFactory

    loader = loaderFactory("potts")
    loader.load("mid1")
    
    # L = len(loader.get_data()[0])

    cfg = OmegaConf.load("/cluster/home/flohmann/generating-enzymes/configs/potts/config.yaml")
    cfg.training.optimizer.method="l-bfgs"
    cfg.training.optimizer.history_size = 20
    cfg.training.test_split = 0.001

    cfg.model.L = 97
    # train_data, test_data = loader.preprocess(cfg.training.test_split, train_batch_size=int(np.ceil((1-cfg.training.test_split)*loader.length)), 
    #                                           test_batch_size=cfg.training.batch_size, d = 20, shuffle=True)
    
    # model = modelFactory("potts", cfg = cfg)
    # model.run_training(train_data, test_data, cfg.training)

    # torch.save(model, "potts_mid1.pt")

    model = torch.load("potts_mid1.pt", map_location="cpu")

    contacts = model.get_coupling_info(apc=True)
    
    fig, ax = plt.subplots()
    ax.imshow(contacts)
    plt.show()
    plt.savefig("potts_contacts.png", dpi=500)
    plt.close()

    model.projector = lambda x, is_onehot = True: x

    x0_p = np.random.choice(loader.get_data())
    x0_p = torch.tensor(aa2int_single(x0_p)).unsqueeze(0)
    x0_p = torch.nn.functional.one_hot(x0_p, num_classes = model.d).double()
    
    sampler = "local"
    seed = 31

    model.set_seed(seed)

    cfg.generation.temp_marginal = 0.01
    cfg.generation.n_episodes = 10000
    cfg.generation.n_burnin = 10000
    cfg.generation.sampler = sampler
    cfg.generation.output_file = f"./gen_data/mid1/potts/frozen_seed_{cfg.generation.seed}_{cfg.generation.sampler}_T_{cfg.generation.temp_marginal}.fasta"
    print(cfg)

    model.generate(cfg, x0_p)

