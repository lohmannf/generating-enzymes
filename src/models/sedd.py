from omegaconf import DictConfig, OmegaConf
import wandb
import torch.multiprocessing as mp
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import numpy as np
import warnings
from itertools import chain
from tqdm import tqdm

from src.models.sedd_utils import graph_lib, noise_lib, losses, sampling
from src.models.sedd_utils import model as sm
from src.models.sedd_utils.model import utils as mutils
from src.models.basemodel import BaseModel
from src.data import loaderFactory, ProteinTokenizer

torch.backends.cudnn.benchmark = True
DISABLE_WANDB = True

def load_ckpt(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        warnings.warn(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_ckpt(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def run_training(rank: int, 
                world_size: int, 
                train_cfg: DictConfig, 
                run, 
                distributed: bool):

        # Only log from the main process
        def mprint(msg):
            #FIXME: Use an actual logger
            if rank == 0:
                print(msg, flush=True)

        def mlog(log_dict):
            if rank == 0 and not DISABLE_WANDB:
                run.log(log_dict)
        
        mprint(train_cfg)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        sample_dir = os.path.join(train_cfg.work_dir, "samples")
        checkpoint_dir = os.path.join(train_cfg.work_dir, "checkpoints")
        checkpoint_meta_dir = os.path.join(train_cfg.work_dir, "checkpoints-meta", "checkpoint.pth")
        if rank == 0:
            os.mkdir(sample_dir)
            os.mkdir(checkpoint_dir)
            os.mkdir(os.path.dirname(checkpoint_meta_dir))

        if device.type == "cpu":
            mprint("WARNING: Using only cpu")

        graph = graph_lib.get_graph(train_cfg, device)

        mprint(f"Graph exp is {graph._exp}")
        
        model = sm.SEDD(train_cfg).to(device)
        if distributed:
            model = DDP(model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

        noise = noise_lib.get_noise(train_cfg).to(device)
        if distributed:
            noise = DDP(noise, device_ids=[rank], static_graph=True)
        
        sampling_eps = 1e-5

        num_params = 0
        num_train = 0
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                num_train += p.numel()

        num_params = sum(p.numel() for p in model.parameters())
        mlog({"n_params_total": num_params , "n_params_trainable": num_train})

        ema = sm.ExponentialMovingAverage(model.parameters(), decay = train_cfg.training.ema)

        mprint(model)

        # build optimization state
        optimizer = losses.get_optimizer(train_cfg, chain(model.parameters(), noise.parameters()))
        scaler = torch.cuda.amp.GradScaler()
        state = dict(optimizer=optimizer, scaler=scaler, model=model, noise=noise, ema=ema, step=0) 


        state = load_ckpt(checkpoint_meta_dir, state, device)
        initial_step = int(state['step'])
        
        dl = loaderFactory("sedd", block_sz = train_cfg.model.length)
        dl.load(train_cfg.data.name)
        dl._replace("*")
        #dl.data = dl.data[:100]
        data_dict = dl.preprocess(test_frac = train_cfg.data.test_split, group = train_cfg.data.grouped)
        
        train_ds, eval_ds = dl.get_torch_loaders(data_dict["train"], data_dict["test"], 
                                                train_cfg.training.batch_size, train_cfg.eval.batch_size, 
                                                train_cfg.training.accum, distributed=distributed)
        train_iter = iter(train_ds)
        eval_iter = iter(eval_ds)
        
        mprint(f'Epoch size: {np.ceil(len(data_dict["train"])/train_cfg.training.batch_size)}')
        mlog({"steps_per_epoch": np.ceil(len(data_dict["train"])/train_cfg.training.batch_size)})

        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(train_cfg)
        train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, train_cfg.training.accum, ltype=train_cfg.training.loss)
        eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, train_cfg.training.accum, ltype=train_cfg.training.loss)

        # one step generation function
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (train_cfg.sampling.batch_size, train_cfg.model.length), train_cfg.sampling.predictor, train_cfg.sampling.steps, device=device
        )
        tokenizer = ProteinTokenizer(group = train_cfg.data.grouped)

        mprint(f"Starting training loop at step {initial_step}.")

        while state['step'] < train_cfg.training.n_iters + 1:
            
            step = state['step']
            batch = next(train_iter)['input_ids'].to(device)
            loss = train_step_fn(state, batch)

            # flag to see if there was movement ie a full batch got computed
            if step != state['step']:
                if step % train_cfg.training.log_freq == 0:
                    if distributed:
                        dist.all_reduce(loss)
                        loss /= world_size

                    mprint({"train_step": step, "training/loss":  loss.item()})
                    mlog({"train_step": step, "training/loss":  loss.item()})
                
                if step % train_cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                    save_ckpt(checkpoint_meta_dir, state) 

                if step % train_cfg.training.eval_freq == 0:
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                    eval_loss = eval_step_fn(state, eval_batch)

                    if distributed:
                        dist.all_reduce(eval_loss)
                        eval_loss /= world_size

                    mlog({"train_step": step, "evaluation/loss":  eval_loss.item()})
                    mprint({"train_step": step, "evaluation/loss":  eval_loss.item()})

                if train_cfg.training.snapshot_sampling and step % train_cfg.snapshot_freq == 0:
                    # sample from the model
                    with torch.no_grad():
                        for _ in range(train_cfg.sampling.n_batch):
                            samples = sampling_fn(model).cpu().numpy()
                            seqs = tokenizer.batch_decode(samples)

                            with open(os.path.join(train_cfg.work_dir, f"samples/step_{step}.fasta"), "a") as f:
                                for seq in seqs:
                                    f.write(f">\n{seq}\n")

                
                if step > 0 and step % train_cfg.training.snapshot_freq == 0 or step == train_cfg.training.n_iters:
                    # Save the checkpoint.
                    save_step = step // train_cfg.training.snapshot_freq
                    if rank == 0:
                        save_ckpt(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)


def run_training_mp(rank: int, 
                    world_size: int, 
                    port: int, 
                    train_cfg: DictConfig, 
                    run):

    try:
        # Env setup
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        # initialize the process group
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
        )

        # Run the actual training
        run_training(rank, world_size, train_cfg, run, True)

    finally:
        # Env cleanup
        dist.destroy_process_group()



class SEDDWrapper(BaseModel):
    # Adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/run_train.py

    def __init__(self, config_dir = "./../../configs/sedd/"):
        if config_dir.startswith('./'):
            # path is relative to current file
            self.cfg_dir = os.path.join(os.path.dirname(__file__), config_dir)

        else:
            self.cfg_dir = config_dir

    def parse_config(self, cfg_dir = None, resolve_model = True):

        if cfg_dir is None:
            cfg_dir = self.cfg_dir

        def get_dt(date):
            curr_dt = datetime.datetime.now()
            return curr_dt.strftime(date)

        try:
            OmegaConf.register_new_resolver("now", get_dt)
        except:
            pass
            
        cfg = OmegaConf.load(os.path.join(cfg_dir, "config.yaml"))

        if resolve_model:
            model_cfg = OmegaConf.load(os.path.join(cfg_dir, f"{cfg.model}.yaml"))
            cfg.model = model_cfg

        return cfg


    def run_training(self, train_cfg: DictConfig = None):

        target_cfg = self.parse_config()

        if train_cfg is not None:
            # override default config values
            train_cfg = OmegaConf.merge(target_cfg, train_cfg)
        else:
            train_cfg = target_cfg

        OmegaConf.resolve(train_cfg)

        if not os.path.exists(train_cfg.work_dir):
            os.makedirs(train_cfg.work_dir)
        else:
            warnings.warn(f"Directory {train_cfg.work_dir} already exists, may overwrite some files")

        OmegaConf.save(train_cfg, f = os.path.join(train_cfg.work_dir, "config.yaml"), resolve = True)

        os.environ["WANDB__SERVICE_WAIT"] = "300" # wait a bit longer than default
        if not DISABLE_WANDB:
            run = wandb.init(project = train_cfg.wandb.project,
                            dir = train_cfg.work_dir,
                            config = OmegaConf.to_object(train_cfg))
        else:
            run = None
        
        try: 
            if train_cfg.training.distributed:
                # Run training in DDP mode
                port = int(np.random.randint(10000, 20000))
                ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                mp.set_start_method("forkserver")
                mp.spawn(run_training_mp, args=(ngpus, port, train_cfg, run), nprocs=ngpus, join=True)

            else:
                run_training(0, 1, train_cfg, run, distributed=False)
        
        finally:
            if not DISABLE_WANDB:
                wandb.finish()


    def generate(self, model_path: str, gen_cfg: DictConfig = None):

        target_cfg = self.parse_config()

        if gen_cfg is not None:
            # override default config values
            gen_cfg = OmegaConf.merge(target_cfg, gen_cfg)
        else:
            gen_cfg = target_cfg
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_cfg = self.parse_config(model_path, resolve_model = False)

        try:
            ckpt = gen_cfg.checkpoint
        except KeyError:
            ckpt = None
        
        ckpt_dir = os.path.join(model_path, "checkpoints")
        if ckpt is None:
            # get the most recent checkpoint
            avail_ckpts = os.listdir(ckpt_dir)
            ckpt = avail_ckpts[np.argmax([int(name.split("_")[1].strip(".pth")) for name in avail_ckpts])]

        # load the checkpoint
        loaded_state = torch.load(os.path.join(ckpt_dir, ckpt), map_location=device)
        
        graph = graph_lib.get_graph(model_cfg, device)
        noise = noise_lib.get_noise(model_cfg).to(device)
        score_model = sm.SEDD(model_cfg).to(device)
        ema = sm.ExponentialMovingAverage(score_model.parameters(), decay=model_cfg.training.ema)

        score_model.load_state_dict(loaded_state['model'])
        ema.load_state_dict(loaded_state['ema'])

        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

        print(gen_cfg)

        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (gen_cfg.sampling.batch_size, model_cfg.model.length), gen_cfg.sampling.predictor, gen_cfg.sampling.steps, device=device
        )

        # generate batches
        n_batches = np.ceil(gen_cfg.sampling.n_samples/gen_cfg.sampling.batch_size).astype(int)
        tokenizer = ProteinTokenizer(group = model_cfg.data.grouped)
        for i in tqdm(range(n_batches), desc="Generating batches"):
            samples = sampling_fn(score_model).cpu().numpy()
            seqs = tokenizer.batch_decode(samples)

            with open(os.path.join(gen_cfg.out_dir, gen_cfg.data.name+".fasta"), "a") as f:
                for seq in seqs:
                    f.write(f">\n{seq}\n")




if __name__ == "__main__":

    sedd = SEDDWrapper()
    # # train_overrides = OmegaConf.create({"data": {"name": "ired", "test_split": 0.99, "grouped": False}, 
    # #                                     "training": {"n_iters": 100000, "distributed": True, "loss": "dwdse", "batch_size": 2},
    # #                                     "optim": {"lr": 3e-4},
    # #                                     "noise": {"type": "loglinear"},
    # #                                     "eval": {"batch_size": 32},
    # #                                     "model": {"length": 300}})
    # train_overrides = OmegaConf.create({"data": {"name": "ired", "test_split": 0.2, "grouped": False}, 
    #                                    "training": {"n_iters": 100000, "distributed": True, "loss": "ce", "batch_size": 128},
    #                                    "optim": {"lr": 3e-5},
    #                                    "noise": {"type": "linear"}, #loglinear
    #                                    "eval": {"batch_size": 64},
    #                                    "model": {"length": 291}})
    # sedd.run_training(train_overrides)  #128 64

    gen_overrides = OmegaConf.create({"sampling" : {"batch_size": 32, "steps": 1000, "n_samples": 5000, "predictor": "euler"}, 
                                      "checkpoint" : "checkpoint_8.pth",
                                      "out_dir": "./gen_data/sedd/"})

    #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.09.30/13:54:43", gen_overrides) #sedd
    #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.09.30/13:53:30", gen_overrides) #dfm
    #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.09.30/10:16:48", gen_overrides) #dfm overfit lr 3e-5
    #sedd.generate('/cluster/project/krause/flohmann/sedd/ired/2024.09.30/19:45:34', gen_overrides) #sedd overfit lr 3e-5
    #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.10.08/20:36:52", gen_overrides) #dfm w/o padding
    sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.10.08/09:46:36", gen_overrides) #sedd w/o padding