import torch
from transformers import PreTrainedTokenizer, TrainerCallback, TrainingArguments
from typing import Callable
import math

class GenerateSeqCallback(TrainerCallback):
    # adapted from https://discuss.huggingface.co/t/generating-text-while-model-is-still-training/57482
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 prompt: str, 
                 device, 
                 output_file: str, 
                 post_func: Callable,
                 n_steps=100, 
                 n_batches=50,
                 **gen_kwargs):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.device = device
        self.n_steps = n_steps
        self.gen_kwargs = gen_kwargs
        self.output_file = output_file
        self.pad_id = 0
        self.eot_id = 1
        self.post = post_func
        self.n_batches = n_batches

        self.step_count = 0
        

    def on_step_end(self, 
                    args: TrainingArguments, 
                    state, 
                    control, 
                    **kwargs):
        
        self.step_count += 1

        if self.step_count % self.n_steps == 0:
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(self.device)

            for i in range(self.n_batches):
                with torch.no_grad():
                    generated_text = kwargs['model'].generate(input_ids, **self.gen_kwargs)
                    #generated_text =  [x for x in generated_text if x[-1] == 0 or x[-1]==1]
                    msk = torch.logical_or(generated_text[:,-1] == self.pad_id, generated_text[:,-1] == self.eot_id)
                    generated_text = generated_text[msk]
                    ppl = [math.exp(kwargs['model'](x, labels=x)[0]) for x in generated_text]
                
                decoded_text = self.tokenizer.batch_decode(generated_text)
                decoded_text = self.post(decoded_text)

                with open(self.output_file, "a") as file:
                    for p, seq in zip(ppl, decoded_text):
                        file.write(f'>{self.prompt} {p} {self.step_count}\n{seq}\n')
            