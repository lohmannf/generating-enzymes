import torch
import numpy as np
from transformers import EsmForSequenceClassification, AutoTokenizer
import wandb
import math
from omegaconf import OmegaConf, DictConfig
import os
import math

from src.models.ebm import EnergyBasedModel
from src.models.basemodel import BaseModel
from src.data.utils import AA_DICT, onehot2aa, aa2int_single


def get_plh_func(aa_idx, 
                 L, 
                 map_to_aa_idx, 
                 pos_per_chunk,
                 is_loss = True,
                 accum = 1):
    
    d = len(aa_idx)

    def pseudo_lh(model, x, train = True):
        
        cum_loss = 0
        n_seqs = x.size()[0]
        x_idx = map_to_aa_idx[x[:, 1:L+1].reshape(-1)] #flatten the batch to (n_seqs*L,)

        # we want to change only 1 position at a time
        batch = x.repeat_interleave(d*L, dim=0) #(n_seqs*d*L)x(L+2)
        # construct every amino acid at every position
        batch[torch.arange(batch.size()[0], dtype=int), (torch.arange(1, L+1, dtype=int).repeat_interleave(d, dim=0)).repeat(n_seqs)] = aa_idx.repeat(n_seqs*L).long()

        chunk_sz = pos_per_chunk * d # all variants of a single position must end up in the same chunk

        if is_loss:

            for subatch, sub_idx in zip(torch.split(batch, chunk_sz, dim=0), torch.split(x_idx, pos_per_chunk, dim=0)):

                subatch_e = model(subatch.to(model.device)).reshape(-1, d)

                #sum over positions and sequences to get PLH of the batch
                loss = -(torch.nn.functional.log_softmax(subatch_e, dim=-1)[torch.arange(len(subatch_e), dtype=int), sub_idx]).sum()
                
                if train:
                    loss /= accum
                    loss.backward()

                cum_loss += loss.item()

            return cum_loss

        else:

            batch_e = torch.empty((0,d))
            
            for subatch, sub_idx in zip(torch.split(batch, chunk_sz, dim=0), torch.split(x_idx, pos_per_chunk, dim=0)):
                batch_e = torch.cat(batch_e, model(subatch.to(model.device)).reshape(-1, d), dim=0)
            
            return batch_e.reshape(n_seqs, L, d)

    
    return pseudo_lh


class DeepEBM(EnergyBasedModel, BaseModel):

    def __init__(self, model_cfg: DictConfig):

        super().__init__(model_cfg.L, model_cfg.d, model.cfg.seed)

        self.mode = model_cfg.mode

        if model_cfg.energy_model.name == "esm":
            self.em_tokenizer = AutoTokenizer.from_pretrained(model_cfg.energy_model.path) 

            tmp = {k: v for v, k in enumerate(self.em_tokenizer.all_tokens)}
            self.em = EsmForSequenceClassification.from_pretrained(model_cfg.energy_model.path, num_labels = 1)
            
            self.aa_idx = torch.Tensor([v for k, v in tmp.items() if k in list(AA_DICT.keys())])
            self.map_to_aa_idx = -torch.ones(int(self.aa_idx.max().item())+1, dtype=int)
            self.map_to_aa_idx[self.aa_idx.int()] = torch.arange(len(self.aa_idx), dtype = int)

            
        else:
            raise NotImplementedError(f"Unknown energy model {model_cfg.energy_model.name}")
        
        self.em_name = model_cfg.energy_model.name
        self.to(self.device)


    def forward(self, seqs):
        """
        Calculate score function / negative energy.
        Sequences encoded with em_tokenizer
        """
        output = self.em(seqs)

        if self.em_name == "esm":
            output = output.logits

        return output
    
    def diff(self, x: torch.Tensor, T: float = 2.):
        '''
        Calculate `f(x')-f(x) / T` for all
        `x'` in the Hamming ball of radius 1 from `x`

        Parameters
        ----------
        x: torch.Tensor
            The current state of the chain in one-hot encoding

        T: float
            The temperature used for the proposal distribution,
            default = 2.

        Returns
        -------
        energy: torch.Tensor
            The logits for the proposal distribution
        '''
        
        energy = get_plh_func(self.aa_idx, self.L, self.map_to_aa_idx, 3, is_loss = False)(self.projector(x))
        return (energy - energy[0,0,x[0,0,:].bool()]) / T

    def configure_optimizers(self, train_cfg):

        optimizer = torch.optim.Adam(self.parameters(), weight_decay=train_cfg.optimizer.decay, lr=train_cfg.optimizer.lr, betas=(train_cfg.optimizer.beta1, train_cfg.optimizer.beta2))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=train_cfg.optimizer.gamma)

        return optimizer, scheduler


    def run_training(self, 
                     train_data, 
                     test_data, 
                     train_cfg: DictConfig):

        self.train()

        # set the loss function
        if self.mode == "pseudolh":
            self.loss_func = get_plh_func(self.aa_idx, self.L, self.map_to_aa_idx, train_cfg.chunk_sz, train_cfg.optimizer.accum)

        if train_cfg.freeze:
            for p in self.em.esm.parameters():
                p.requires_grad = False

        n_trainable = 0
        for p in self.em.parameters():
            n_trainable += p.requires_grad * p.numel()

        print(f"Number trainable parameters: {n_trainable}", flush=True)

        if not os.path.exists(train_cfg.work_dir):
            os.mkdir(train_cfg.work_dir)

        OmegaConf.save(train_cfg, f = os.path.join(train_cfg.work_dir, "config.yaml"), resolve = True)

        if train_cfg.use_wandb:
            run = wandb.init(project = train_cfg.wandb.project,
                            dir = train_cfg.work_dir,
                            config = OmegaConf.to_object(train_cfg))
        else:
            run = None

        try:
            optimizer, scheduler = self.configure_optimizers(train_cfg)

            train_step = 0
            if train_cfg.n_epochs is None:
                train_cfg.n_epochs = math.ceil(train_cfg.n_steps / len(train_data))
            else:
                train_cfg.n_steps = 1e9


            for epoch in range(train_cfg.n_epochs):
                eval_it = iter(test_data)
                loss = 0
                for batch_em, batch_p in iter(train_data):
                    
                    loss += self.loss_func(self, batch_em)

                    if train_step % train_cfg.log_freq == 0:
                        print({"train_step": train_step, "epoch": epoch, "training/loss": loss, "lr": scheduler.get_last_lr()[0]}, flush=True)
                        if run is not None:
                            run.log({"train_step": train_step, "epoch": epoch, "training/loss": loss, "lr": scheduler.get_last_lr()[0]})

                    if train_step > 0 and train_step % train_cfg.optimizer.accum == 0:
                        torch.nn.utils.clip_grad_norm_(self.em.parameters(), train_cfg.optimizer.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        loss = 0
                        
                    if train_step % train_cfg.eval_freq == 0:
                        with torch.no_grad():
                            eval_batch, _ = next(eval_it)
                            eval_loss = self.loss_func(self, eval_batch, train=False)
                        
                        print({"train_step": train_step, "epoch": epoch, "evaluation/loss": eval_loss}, flush=True)
                        if run is not None:
                            run.log({"train_step": train_step, "epoch": epoch, "evaluation/loss": eval_loss})

                    if train_step % train_cfg.snapshot_freq == 0 and train_step > 0:
                        id = train_step // train_cfg.snapshot_freq
                        torch.save(self.em, os.path.join(train_cfg.work_dir, f"checkpoint_{id}.pt"))

                    train_step += 1
                    if train_step == train_cfg.n_steps:
                        break
                
                scheduler.step()

        finally:
            if train_cfg.use_wandb:
                wandb.finish()


    def load_ckpt(self, path):

        if os.path.isdir(path):
            # find most recent checkpoint
            files = os.listdir(path)
            latest = np.argsort([int(f.split("_")[1].strip(".pt")) for f in files if f.startswith("checkpoint")])[-1]
            path = os.path.join(path, files[latest])

        print(f"Loading checkpoint {path}", flush=True)
        self.em = torch.load(path, map_location=self.device)


if __name__ == "__main__":

    from src.data import loaderFactory

    loader = loaderFactory("debm")
    loader.load("ired", remove_ast=True)
    loader.unify_seq_len(290)

    cfg = OmegaConf.load("/cluster/home/flohmann/generating-enzymes/configs/deep_ebm/config.yaml")
    cfg.training.freeze = True
    cfg.training.batch_size = 1
    cfg.training.log_freq = 16
    cfg.training.optimizer.lr = 1e-3
    cfg.training.snapshot_freq = 50
    cfg.training.chunk_sz = 3
    cfg.training.optimizer.accum = 16
    OmegaConf.resolve(cfg)

    model = DeepEBM(cfg.model)

    train_dl, test_dl = loader.preprocess(0.2, cfg.training.batch_size, cfg.training.batch_size, tokenizer = model.em_tokenizer)

    print(cfg)

    model.run_training(train_dl, test_dl, cfg.training)

    # seed = 31
    # sampler = "gwg"
    # T_marg = 10
    # dir = f"./gen_data/deep_ebm/unfrozen_seed_{seed}_{sampler}_T_{T_marg}"
    # os.mkdir(dir)

    # model = DeepEBM(290, 20, "esm")
    # model.set_seed(seed)
    # #model.load_ckpt("/cluster/project/krause/flohmann/deep_ebm_f_True_lr_0.0008/checkpoint_2.pt")
    # model.load_ckpt("/cluster/project/krause/flohmann/deep_ebm_f_False_lr_8e-05/")

    # x0_p = np.random.choice(loader.get_data())
    # x0_p = torch.tensor(aa2int_single(x0_p)).unsqueeze(0)
    # x0_p = torch.nn.functional.one_hot(x0_p, num_classes = model.d).double()

    # seqs = model.generate(x0_p, 10000, 10000, dir, sampler = "gwg", T_marg = T_marg, keep_in_memory=False)

