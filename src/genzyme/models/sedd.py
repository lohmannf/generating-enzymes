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

from genzyme.models.sedd_utils import graph_lib, noise_lib, losses, sampling
from genzyme.models.sedd_utils import model as sm
from genzyme.models.sedd_utils.model import utils as mutils
from genzyme.models.basemodel import BaseModel
from genzyme.data import loaderFactory, ProteinTokenizer

torch.backends.cudnn.benchmark = True
DISABLE_WANDB = False

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
                cfg: DictConfig, 
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
        
        mprint(cfg)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        sample_dir = os.path.join(cfg.work_dir, "samples")
        checkpoint_dir = os.path.join(cfg.work_dir, "checkpoints")
        checkpoint_meta_dir = os.path.join(cfg.work_dir, "checkpoints-meta", "checkpoint.pth")
        if rank == 0:
            os.mkdir(sample_dir)
            os.mkdir(checkpoint_dir)
            os.mkdir(os.path.dirname(checkpoint_meta_dir))

        if device.type == "cpu":
            mprint("WARNING: Using only cpu")

        graph = graph_lib.get_graph(cfg, device)

        mprint(f"Graph exp is {graph._exp}")
        
        model = sm.SEDD(cfg).to(device)
        if distributed:
            model = DDP(model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

        noise = noise_lib.get_noise(cfg).to(device)
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

        ema = sm.ExponentialMovingAverage(model.parameters(), decay = cfg.training.ema)

        mprint(model)

        # build optimization state
        optimizer = losses.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
        scaler = torch.cuda.amp.GradScaler()
        state = dict(optimizer=optimizer, scaler=scaler, model=model, noise=noise, ema=ema, step=0) 


        state = load_ckpt(checkpoint_meta_dir, state, device)
        initial_step = int(state['step'])
        
        dl = loaderFactory("sedd", block_sz = cfg.model.length)
        dl.load(cfg.data.name)
        dl._replace("*")
        #dl.data = dl.data[:100]
        data_dict = dl.preprocess(test_frac = cfg.data.test_split, group = cfg.data.grouped)
        
        train_ds, eval_ds = dl.get_torch_loaders(data_dict["train"], data_dict["test"], 
                                                cfg.training.batch_size, cfg.eval.batch_size, 
                                                cfg.training.accum, distributed=distributed)
        train_iter = iter(train_ds)
        eval_iter = iter(eval_ds)
        
        mprint(f'Epoch size: {np.ceil(len(data_dict["train"])/cfg.training.batch_size)}')
        mlog({"steps_per_epoch": np.ceil(len(data_dict["train"])/cfg.training.batch_size)})

        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(cfg)
        train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum, ltype=cfg.training.loss)
        eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum, ltype=cfg.training.loss)

        # one step generation function
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (cfg.sampling.batch_size, cfg.model.length), cfg.sampling.predictor, cfg.sampling.steps, device=device
        )
        tokenizer = ProteinTokenizer(group = cfg.data.grouped)

        mprint(f"Starting training loop at step {initial_step}.")

        while state['step'] < cfg.training.n_iters + 1:
            
            step = state['step']
            batch = next(train_iter)['input_ids'].to(device)
            loss = train_step_fn(state, batch)

            # flag to see if there was movement ie a full batch got computed
            if step != state['step']:
                if step % cfg.training.log_freq == 0:
                    if distributed:
                        dist.all_reduce(loss)
                        loss /= world_size

                    mprint({"train_step": step, "training/loss":  loss.item()})
                    mlog({"train_step": step, "training/loss":  loss.item()})
                
                if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                    save_ckpt(checkpoint_meta_dir, state) 

                if step % cfg.training.eval_freq == 0:
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                    eval_loss = eval_step_fn(state, eval_batch)

                    if distributed:
                        dist.all_reduce(eval_loss)
                        eval_loss /= world_size

                    mlog({"train_step": step, "evaluation/loss":  eval_loss.item()})
                    mprint({"train_step": step, "evaluation/loss":  eval_loss.item()})

                if cfg.training.snapshot_sampling and step % cfg.snapshot_freq == 0:
                    # sample from the model
                    with torch.no_grad():
                        for _ in range(cfg.sampling.n_batch):
                            samples = sampling_fn(model).cpu().numpy()
                            seqs = tokenizer.batch_decode(samples)

                            with open(os.path.join(cfg.work_dir, f"samples/step_{step}.fasta"), "a") as f:
                                for seq in seqs:
                                    f.write(f">\n{seq}\n")

                
                if step > 0 and step % cfg.training.snapshot_freq == 0 or step == cfg.training.n_iters:
                    # Save the checkpoint.
                    save_step = step // cfg.training.snapshot_freq
                    if rank == 0:
                        save_ckpt(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)


def run_training_mp(rank: int, 
                    world_size: int, 
                    port: int, 
                    cfg: DictConfig, 
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
        run_training(rank, world_size, cfg, run, True)

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


    def run_training(self, cfg: DictConfig = None):

        target_cfg = self.parse_config()

        if cfg is not None:
            # override default config values
            cfg = OmegaConf.merge(target_cfg, cfg)
        else:
            cfg = target_cfg

        OmegaConf.resolve(cfg)

        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir)
        else:
            warnings.warn(f"Directory {cfg.work_dir} already exists, may overwrite some files")

        OmegaConf.save(cfg, f = os.path.join(cfg.work_dir, "config.yaml"), resolve = True)

        os.environ["WANDB__SERVICE_WAIT"] = "300" # wait a bit longer than default
        if not DISABLE_WANDB:
            run = wandb.init(project = cfg.wandb.project,
                            dir = cfg.work_dir,
                            config = OmegaConf.to_object(cfg))
        else:
            run = None
        
        try: 
            if cfg.training.distributed:
                # Run training in DDP mode
                port = int(np.random.randint(10000, 20000))
                ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                mp.set_start_method("forkserver")
                mp.spawn(run_training_mp, args=(ngpus, port, cfg, run), nprocs=ngpus, join=True)

            else:
                run_training(0, 1, cfg, run, distributed=False)
        
        finally:
            if not DISABLE_WANDB:
                wandb.finish()


    def generate(self, model_path: str, cfg: DictConfig = None):

        target_cfg = OmegaConf.load(os.path.join(model_path, "config.yaml"))

        if cfg is not None:
            # override default config values
            cfg = OmegaConf.merge(target_cfg, cfg)
        else:
            cfg = target_cfg
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_cfg = self.parse_config(model_path, resolve_model = False)

        try:
            ckpt = cfg.checkpoint
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

        print(cfg)

        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (cfg.sampling.batch_size, model_cfg.model.length), cfg.sampling.predictor, cfg.sampling.steps, device=device
        )

        # generate batches
        n_batches = np.ceil(cfg.sampling.n_samples/cfg.sampling.batch_size).astype(int)
        tokenizer = ProteinTokenizer(group = model_cfg.data.grouped)
        for i in tqdm(range(n_batches), desc="Generating batches"):
            samples = sampling_fn(score_model).cpu().numpy()
            seqs = tokenizer.batch_decode(samples)

            with open(os.path.join(cfg.out_dir, cfg.data.name+".fasta"), "a") as f:
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

    # train_overrides = OmegaConf.create({"data": {"name": "mid1", "test_split": 0.2, "grouped": False}, 
    #                                    "training": {"n_iters": 100000, "distributed": True, "loss": "dwdse", "batch_size": 128},
    #                                    "optim": {"lr": 3e-5},
    #                                    "noise": {"type": "loglinear"},
    #                                    "eval": {"batch_size": 64},
    #                                    "model": {"length": 98}})
    # sedd.run_training(train_overrides)

    # gen_overrides = OmegaConf.create({"sampling" : {"batch_size": 32, "steps": 1000, "n_samples": 5000, "predictor": "euler"}, 
    #                                   "checkpoint" : "checkpoint_8.pth",
    #                                   "out_dir": "./gen_data/sedd/"})

    # #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.09.30/13:54:43", gen_overrides) #sedd
    # #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.09.30/13:53:30", gen_overrides) #dfm
    # #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.09.30/10:16:48", gen_overrides) #dfm overfit lr 3e-5
    # #sedd.generate('/cluster/project/krause/flohmann/sedd/ired/2024.09.30/19:45:34', gen_overrides) #sedd overfit lr 3e-5
    # #sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.10.08/20:36:52", gen_overrides) #dfm w/o padding
    # sedd.generate("/cluster/project/krause/flohmann/sedd/ired/2024.10.08/09:46:36", gen_overrides) #sedd w/o padding
    gen_overrides = OmegaConf.create({"sampling" : {"batch_size": 32, "steps": 1000, "n_samples": 10000, "predictor": "euler-dfm"}, 
                                      "checkpoint" : "checkpoint_9.pth",
                                      "out_dir": "./gen_data/mid1/dfm/"})

    #sedd.generate("/cluster/project/krause/flohmann/sedd/mid1/2024.10.16/17:10:01", gen_overrides)   #sedd mid1
    sedd.generate("/cluster/project/krause/flohmann/sedd/mid1/2024.10.16/17:07:55", gen_overrides)   #dfm mid1