import torch
import numpy as np
from transformers import EsmForSequenceClassification, EsmModel, AutoTokenizer
import wandb
import math
from omegaconf import OmegaConf, DictConfig
import os
import math

from genzyme.models.ebm import EnergyBasedModel
from genzyme.models.basemodel import BaseModel
from genzyme.data.utils import AA_DICT, aa2int_single


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
                batch_e = torch.cat([batch_e, model(subatch.to(model.device)).reshape(-1, d)], dim=0)
            
            return batch_e.reshape(n_seqs, L, d)

    
    return pseudo_lh


def get_projector(p2em: torch.Tensor):

    def to_em_encoding(x_p: torch.Tensor, is_onehot: bool = True):
        '''
        Return x_p in energy model integer encoding
        '''

        if is_onehot:
            x_int = torch.argmax(x_p, dim = -1)
        else:
            x_int = x_p
        
        n_seqs = x_p.size()[0]
        x_em = torch.cat([torch.zeros(n_seqs, 1), p2em[x_int], 2*torch.ones(n_seqs, 1)], dim=1)

        return x_em.to(x_p)
    
    return to_em_encoding


class DeepEBM(EnergyBasedModel, BaseModel):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg.model.L, cfg.model.d, cfg.model.seed)

        self.mode = cfg.model.mode

        if cfg.model.energy_model.name == "esm":
            self.em = EsmForSequenceClassification.from_pretrained(cfg.model.energy_model.path, num_labels = 1)
            self.configure_tokenizers(cfg)

        elif cfg.model.energy_model.name == "quadratic":
            self.em = EsmModel.from_pretrained(cfg.model.energy_model.path, add_pooling_layer=False)
            self.A = torch.nn.Parameter(torch.randn(self.em.config.hidden_size, self.em.config.hidden_size), True)

            # don't train the PLM
            for p in self.em.parameters():
                p.requires_grad = False

            self.configure_tokenizers(cfg)
            
        else:
            raise NotImplementedError(f"Unknown energy model {cfg.model.energy_model.name}")
        
        self.em_name = cfg.model.energy_model.name
        self.to(self.device)

    
    def configure_tokenizers(self, cfg: DictConfig):
        '''
        Setup the tokenizer and mappings between energy model encoding and simple protein encoding
        '''
        
        self.em_tokenizer = AutoTokenizer.from_pretrained(cfg.model.energy_model.path) 
        tmp = {k: v for v, k in enumerate(self.em_tokenizer.all_tokens)}

        self.aa_idx = torch.Tensor([v for k, v in tmp.items() if k in list(AA_DICT.keys())]) #all positions in em encoding that are also in protein encoding
        self.map_to_aa_idx = -torch.ones(int(self.aa_idx.max().item())+1, dtype=int)
        self.map_to_aa_idx[self.aa_idx.int()] = torch.arange(len(self.aa_idx), dtype = int)

        # map from protein encoding to em encoding
        p2em = torch.tensor([tmp[aa] for aa in np.array(list(AA_DICT.keys()))[np.argsort(list(AA_DICT.values()))]])
        self.projector = get_projector(p2em)

        

    def forward(self, seqs):
        """
        Calculate score function / negative energy.
        Sequences encoded with em_tokenizer
        """

        output = self.em(seqs.long())

        if self.em_name == "esm":
            output = output.logits

        elif self.em_name == "quadratic":
            embed_p_seq = output.last_hidden_state[:,1:self.L+1,:].mean(1)
            output = -torch.einsum("si,sj,ij->s", embed_p_seq, embed_p_seq, self.A)

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
        
        energy = get_plh_func(self.aa_idx, self.L, self.map_to_aa_idx, 3, is_loss = False)(self, self.projector(x.long()), train=False)
        return (energy - energy[0,0,x[0,0,:].bool()]) / T

    def configure_optimizers(self, cfg):

        optimizer = torch.optim.Adam(self.parameters(), weight_decay=cfg.training.optimizer.decay, 
                                     lr=cfg.training.optimizer.lr, betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=cfg.training.optimizer.gamma)

        return optimizer, scheduler


    def run_training(self, 
                     train_data, 
                     test_data, 
                     cfg: DictConfig):

        
        if cfg.training.ckpt is not None:
            train_step = self.load_ckpt(cfg.training.ckpt, True)
            print(f"Resuming training from step {train_step}", flush=True)
        else:
            train_step = 0
        
        self.train()

        # set the loss function
        if self.mode == "pseudolh":
            self.loss_func = get_plh_func(self.aa_idx, self.L, self.map_to_aa_idx, cfg.training.chunk_sz, True, cfg.training.optimizer.accum)

        if cfg.training.freeze and self.em_name == "esm":
            for p in self.em.esm.parameters():
                p.requires_grad = False

        n_trainable = 0
        for p in self.em.parameters():
            n_trainable += p.requires_grad * p.numel()

        print(f"Number trainable parameters: {n_trainable}", flush=True)

        if not os.path.exists(cfg.training.work_dir):
            os.mkdir(cfg.training.work_dir)

        if train_step == 0:
            OmegaConf.save(cfg.training, f = os.path.join(cfg.training.work_dir, "config.yaml"), resolve = True)

        if cfg.training.use_wandb:
            run = wandb.init(project = cfg.training.wandb.project,
                            dir = cfg.training.work_dir,
                            config = OmegaConf.to_object(cfg.training))
        else:
            run = None

        try:
            optimizer, scheduler = self.configure_optimizers(cfg)

            if cfg.training.n_epochs is None:
                cfg.training.n_epochs = math.ceil(cfg.training.n_steps / len(train_data))
            else:
                cfg.training.n_steps = 1e9


            for epoch in range(cfg.training.n_epochs):
                eval_it = iter(test_data)
                loss = 0
                for batch_em, batch_p in iter(train_data):
                    
                    loss += self.loss_func(self, batch_em)

                    if train_step > 0:

                        if train_step % cfg.training.log_freq == 0:
                            print({"train_step": train_step, "epoch": epoch, "training/loss": loss, "lr": scheduler.get_last_lr()[0]}, flush=True)
                            if run is not None:
                                run.log({"train_step": train_step, "epoch": epoch, "training/loss": loss, "lr": scheduler.get_last_lr()[0]})
                            
                        if train_step % cfg.training.eval_freq == 0:
                            with torch.no_grad():
                                eval_batch, _ = next(eval_it)
                                eval_loss = self.loss_func(self, eval_batch, train=False)
                            
                            print({"train_step": train_step, "epoch": epoch, "evaluation/loss": eval_loss}, flush=True)
                            if run is not None:
                                run.log({"train_step": train_step, "epoch": epoch, "evaluation/loss": eval_loss})
                    
                    if train_step % cfg.training.optimizer.accum == 0:
                            torch.nn.utils.clip_grad_norm_(self.em.parameters(), cfg.training.optimizer.grad_clip)
                            optimizer.step()
                            optimizer.zero_grad()
                            loss = 0

                    if train_step > 0 and train_step % cfg.training.snapshot_freq == 0:
                        id = train_step // cfg.training.snapshot_freq
                        self.save_to_disk(os.path.join(cfg.training.work_dir, f"checkpoint_{id}.pt"))

                    train_step += 1
                    if train_step == cfg.training.n_steps:
                        break
                
                scheduler.step()

        finally:
            if cfg.training.use_wandb:
                wandb.finish()

    def save_to_disk(self, path):
        if self.em_name == "esm":
            torch.save(self.em, path)
        elif self.em_name == "quadratic":
            torch.save(self.A, path)


    def load_ckpt(self, path: str, get_step: bool = False):

        if get_step:
            if os.path.isdir(path):
                cfg = OmegaConf.load(os.path.join(path, "config.yaml"))
            else:
                cfg = OmegaConf.load(os.path.join(os.path.dirname(path), "config.yaml"))
                
        if os.path.isdir(path):
            # find most recent checkpoint
            files = os.listdir(path)
            files = [f for f in files if f.startswith("checkpoint")]
            latest = np.argsort([int(f.split("_")[1].strip(".pt")) for f in files])[-1]
            path = os.path.join(path, files[latest])

        print(f"Loading checkpoint {path}", flush=True)
        if self.em_name == "esm":
            self.em = torch.load(path, map_location=self.device)
        elif self.em_name == "quadratic":
            self.A = torch.load(path, map_location=self.device)

        if get_step:
            ckpt_id = int(path.split("_")[-1].strip(".pt"))
            step = (ckpt_id * cfg.snapshot_freq)
            return step + 1


if __name__ == "__main__":

    from genzyme.data import loaderFactory

    # loader = loaderFactory("debm")
    # loader.load("ired", remove_ast=True)
    # loader.unify_seq_len(290)
    loader = loaderFactory("debm")
    loader.load("mid1")

    cfg = OmegaConf.load("/cluster/home/flohmann/generating-enzymes/configs/deep_ebm/config.yaml")
    # cfg.training.freeze = True
    # cfg.training.batch_size = 1
    # cfg.training.log_freq = 16
    # cfg.training.optimizer.lr = 1e-3
    # cfg.training.snapshot_freq = 1000
    # cfg.training.chunk_sz = 10
    # cfg.training.optimizer.accum = 16
    # cfg.model.energy_model.name = "quadratic"
    # cfg.model.L = 97

    # #cfg.training = OmegaConf.load("/cluster/project/krause/flohmann/deep_ebm_f_True_lr_0.001_acc_16/config.yaml")
    # #cfg.training.ckpt = "/cluster/project/krause/flohmann/deep_ebm_f_True_lr_0.001_acc_16/"

    # cfg.training.work_dir = cfg.training.work_dir[:-1]+"_mid1_quadratic"
    # OmegaConf.resolve(cfg)
    # model = DeepEBM(cfg)
    # train_dl, test_dl = loader.preprocess(0.2, cfg.training.batch_size, cfg.training.batch_size, tokenizer = model.em_tokenizer)
    # print(cfg)
    # model.run_training(train_dl, test_dl, cfg)

    seed = 31
    sampler = "uniform"
    T_marg = 1
    dir = f"./gen_data/mid1/deep_ebm/frozen_seed_{seed}_{sampler}_T_{T_marg}.fasta"
    os.mkdir(dir)


    work_dir = "/cluster/project/krause/flohmann/deep_ebm_f_True_lr_0.001_acc_16_mid1_quadratic"

    cfg.model.energy_model.name = "quadratic"
    model = DeepEBM(cfg)
    model.set_seed(seed)
    #model.load_ckpt("/cluster/project/krause/flohmann/deep_ebm_f_True_lr_0.0008/checkpoint_2.pt")
    model.load_ckpt(work_dir)

    x0_p = np.random.choice(loader.get_data())
    x0_p = torch.tensor(aa2int_single(x0_p)).unsqueeze(0)
    x0_p = torch.nn.functional.one_hot(x0_p, num_classes = model.d).double()
    
    cfg.generation.temp_marginal = T_marg
    cfg.generation.n_episodes = 100000
    cfg.generation.n_burnin = 10000
    cfg.generation.sampler = sampler
    cfg.generation.output_file = f"./gen_data/mid1/deep_ebm/quadratic_frozen_seed_{seed}_{sampler}_T_{cfg.generation.temp_marginal}.fasta"
    print(cfg)
    seqs = model.generate(cfg, x0_p)

