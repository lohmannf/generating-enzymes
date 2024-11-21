import os
import numpy as np
import transformers
import datasets
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Trainer,
    AutoConfig,
    default_data_collator,
    TrainingArguments,
    )
import torch
import math
from tqdm import tqdm
import logging
from transformers.trainer_utils import get_last_checkpoint
import wandb
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf

from genzyme.models.basemodel import BaseModel
from genzyme.data.utils import SpecialTokens
from genzyme.models.utils import GenerateSeqCallback

logger = logging.getLogger(__name__)

class ZymCTRL(BaseModel):
    '''Wrapper for the ZymCTRL model by Mundsamey et al., 2024'''

    def __init__(self, cfg: DictConfig):
        '''
        Parameters
        ----------
        cfg: DictConfig
            Dict-style config providing model dir and seed


        Returns
        --------
        ZymCTRL instance
        '''
        
        if cfg.model.dir.startswith('./'):
            # path is relative to current file
            self.model_dir = os.path.join(os.path.dirname(__file__), cfg.model.dir)

        else:
            self.model_dir = cfg.model.dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_dir).to(self.device)
        except:
            pass

        self.special = SpecialTokens("<start>", "<end>", "<pad>", "<|endoftext|>", "<sep>", " ", "[UNK]")
        self.seed = cfg.model.seed
        
        set_seed(self.seed)


    def run_training(self,
                    train_dataset: str | Dataset,
                    eval_dataset: str | Dataset,
                    cfg: DictConfig):
        
        ''' 
        Finetune the model on the provided training data

        Parameters
        ---------
        train_dataset: str | Dataset
            The path to the training data or the training dataset

        eval_dataset: str | Dataset
            The path to the evaluation data or the evaluation dataset
        '''
        
        # logger setup
        log_level = logging.__dict__[cfg.training.log_level.upper()]
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)

        # detect any checkpoints
        if cfg.training.ckpt is None and cfg.training.ckpt_dir is not None:
            cfg.training.ckpt = get_last_checkpoint(cfg.training.ckpt_dir)
            
        if cfg.training.ckpt:
            logger.info(f'Resuming training at {cfg.training.ckpt}')

        # get the data if it's stored on disk
        if isinstance(train_dataset, str):
            train_dataset = load_from_disk(train_dataset)

        if isinstance(eval_dataset, str):
            eval_dataset = load_from_disk(eval_dataset)

        # instantiate the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, 
                                                           cache_dir=cfg.training.cache_dir, 
                                                           use_auth_token=None)
        
        # instantiate the model
        try:
            config = GPT2Config.from_json_file(self.model_dir+'/config.json')
        except FileNotFoundError:
            config = GPT2Config.from_pretrained(self.model_dir)

        config.update({'cache_dir': cfg.training.cache_dir,
                       'use_auth_token': None,
                       'tie_word_embeddings': not cfg.training.freeze})

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, 
                                                            from_tf = False,
                                                            config = config)

        except OSError:
            self.model = AutoModelForCausalLM.from_config(config)

        
        if cfg.training.freeze:
            for k,v in self.model.named_parameters():
                if k.startswith('transformer'):
                    v.requires_grad = False

        if cfg.training.use_wandb:
            wandb.init(**OmegaConf.to_object(cfg.training.wandb))
        else:
            # prevent connection errors
            wandb.init(mode = "disabled") 
        
        # instatiate the trainer
        training_args = TrainingArguments(
            do_train = True,
            do_eval = True,
            output_dir = cfg.training.work_dir,
            evaluation_strategy = 'steps',
            eval_steps = cfg.training.eval_freq,
            logging_steps = cfg.training.log_freq,
            save_steps = cfg.training.snapshot_freq,
            num_train_epochs = cfg.training.n_epochs,
            learning_rate = cfg.training.optimizer.lr,
            dataloader_drop_last = True,
            per_device_train_batch_size = cfg.training.train_batch_size,
            per_device_eval_batch_size = cfg.training.eval_batch_size,
            save_total_limit = cfg.training.snapshot_limit,
            report_to = "wandb" if cfg.training.use_wandb else None,
        )
        trainer = Trainer(model = self.model,
                        args = training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer = self.tokenizer,
                        data_collator = default_data_collator)
        
        
        # initialize stepwise generation
        if cfg.training.sample:
            cbk = GenerateSeqCallback(self.tokenizer, 
                                           cfg.training.snapshot_sampling.prompt,
                                           self.device,
                                           cfg.training.snapshot_sampling.output_file,
                                           self.postproc,
                                           n_steps = cfg.training.snapshot_sampling.freq,
                                           num_return_sequences = cfg.training.snapshot_sampling.batch_size,
                                           **OmegaConf.to_object(cfg.training.snapshot_sampling.kwargs))
            trainer.add_callback(cbk)

        
        # sanity check
        if cfg.training.freeze and trainer.get_num_trainable_parameters() != 1:
            n_train = 0
            for k,v in self.model.named_parameters():
                n_train += v.requires_grad

            if n_train != 1:     
                raise ValueError(f'Detected {n_train} trainable parameters')
        
        logger.info("Running training")

        result = trainer.train(resume_from_checkpoint = cfg.training.ckpt)
        trainer.save_model()

        OmegaConf.save(cfg, f = os.path.join(cfg.training.work_dir, "train_config.yaml"), resolve = True)

        metrics = result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Running evaluation")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
            
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    def generate(self,
                 cfg: DictConfig):
        '''
        Perform inference using the provided control tags
        
        Parameters
        ----------
        cfg: DictConfig
            Dict-style config that contains the generation hyperparameters

        Returns
        -------
        sequences: list | None
            Generated sequences if cfg.generation.keep_in_memory is True
        '''

        self.model = self.model.to(self.device)

        n_batch = math.ceil(cfg.generation.n_seqs/cfg.generation.batch_size)

        result = [] if cfg.generation.keep_in_memory else None
                    
        with tqdm(total = n_batch * cfg.generation.batch_size * len(cfg.generation.prompts), desc = "Generating sequences") as pbar:
            for tag in cfg.generation.prompts:
                for _ in range(n_batch):
                    # tokenize control tag
                    input = self.tokenizer.encode(tag, return_tensors='pt').to(self.device)

                    # generate sequence batch
                    output = self.model.generate(input,
                                                 num_return_sequences=cfg.generation.batch_size,
                                                 **OmegaConf.to_object(cfg.generation.kwargs)
                                                 )
                    
                    # filter out truncated sequences
                    output = [x for x in output if x[-1] == 0 or x[-1]==1]
                    if not output:
                        logger.warning('No non-truncated sequences in batch')

                    if len(output) < cfg.generation.batch_size:
                        logger.warning(f'Removed {cfg.generation.batch_size-len(output)} truncated sequences')
                    
                    # decode sequences
                    dec_output = self.tokenizer.batch_decode(output)
                    dec_output = self.postproc(dec_output, True)
                    
                    # calculate perplexity
                    ppl = self.eval(output, exp = True, encode = False)

                    if cfg.generation.keep_in_memory:
                        result.extend(list(zip([tag]*len(ppl), ppl, dec_output)))

                    # save results to disk
                    with open(cfg.generation.output_file, "a") as file:
                        for p, seq in zip(ppl, dec_output):
                            file.write(f">{tag} {p}\n{seq}\n")

                    pbar.update(cfg.generation.batch_size)

        return result
    


    def postproc(self,
                 raw_seq,
                 remove_unk: bool = True):
        
        '''
        Perform postprocessing on decoded sequences.
        Removes special characters and extracts the AA sequences

        Parameters
        ----------
        raw_seq: torch.longTensor
            The decoded model outputs

        remove_unk: bool
            Whether to remove the unknown token from the sequences,
            default = True

        Returns
        --------
        output: np.array
            Postprocessed sequences
        '''
        
        
        output = np.array([self.remove_special(x, remove_unk) for x in raw_seq])
        output = output[output != '']

        return output


    

    def eval(self, data: list, exp: bool = True, encode: bool = True):
        '''Get the model loss for the given sequences
        
        Parameters
        ----------
        data: list
            The sequences for which to compute the losses
        
        exp: bool = True
            Return perplexity

        encode: bool = True
            Whether to tokenize the sequences before running them through the model, default = True.

        Returns
        -------
        result: list
            The loss/perplexity values for each sequence
        '''

        if encode:
            data = [self.tokenizer.encode(x, return_tensors="pt").to(self.device) for x in data]

        with torch.no_grad():
            result = [math.exp(self.model(x, labels=x)[0]) if exp else self.model(x, labels=x)[0] for x in data]

        return result
                    

    def remove_special(self, output: str, remove_unk: bool = True):
        '''Remove the control tag and any special tokens from output
        
        Parameters
        ----------
        output: str
            Decoded model output

        remove_unk : bool
            Whether to remove the unknown token from the output, default = True

        Returns
        -------
        sequence: str
            AA sequence without special tokens
        '''

        if not remove_unk:
            unk_tok = self.special.unk
            self.special.unk = None

        try:
            sequence = output.split(self.special.sep)[1]

            for tok in self.special.__dict__.values():
                if tok is not None:
                    sequence = sequence.replace(tok, '')

        except:
            logger.warning('Output without separator token')
            sequence = ''

        if not remove_unk:
            self.special.unk = unk_tok

        return sequence
    
