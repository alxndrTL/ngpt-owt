# Copyright(c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT License
# [https://opensource.org/license/mit](https://opensource.org/license/mit)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# The text below is the original header from the nanoGPT library
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import sys
from contextlib import nullcontext
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import GPTConfig, GPT
from torch.nn import functional as F
from datetime import timedelta

# -----------------------------------------------------------------------------
# I/O

eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'ngpt-owt'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 64 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
max_iters = 600000 # total number of training iterations
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# 
time_limit_seconds = 1000000000     # stop after x seconds 
max_iters_per_launch = 1000000000   # stop after x steps of the current

use_nGPT = 1
learning_rate = 15e-4 

# model size and seqlen
if (1): 
    n_layer = 12
    n_head = 16
    n_embd = 1024
    block_size = 1024 # = context/sequence length

if (use_nGPT == 0):
    min_lr = 0.0 
    weight_decay = 0.1
    warmup_iters = 2000 
if (use_nGPT == 1):
    min_lr = 0.0
    weight_decay = 0.0
    warmup_iters = 0 

tlaunch = time.time()
print("Current Directory:", os.getcwd())
# the input configurations will overwrite all configs given above!
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if (use_nGPT == 0):
    base_scale = 0.02 # can be interpreted as init_std
if (use_nGPT == 1):
    base_scale = 1.0 / n_embd ** 0.5

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    #init_process_group(backend=backend)
    dist.init_process_group(backend=backend,
        timeout=timedelta(milliseconds=20*60000) # Setting a 20-minute timeout
    )  
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    dist.barrier()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


out_dir='./'
if master_process:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


local_seed = seed_offset
np.random.seed(local_seed)
torch.manual_seed(local_seed)
torch.cuda.manual_seed(local_seed)


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
tdataloading_begin = time.time()
if os.path.exists('./../../data'):
    data_dir = os.path.join('./../../data', dataset)
else:   
    data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0


# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
print("Data loading time: %f sec" % (time.time()-tdataloading_begin))


# model init
tmodelinit_begin = time.time()
model_args = dict(use_nGPT=use_nGPT, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, base_scale=base_scale, 
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['use_nGPT', 'base_scale', 'n_layer', 'n_head',  'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
print("Model initialization/loading time: %f sec" % (time.time()-tmodelinit_begin))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, config=config)

# training loop
#X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed


if master_process:
    print("learning_rate: %f" % (learning_rate))
    print("min_lr: %f" % (min_lr))
    print("max_iters: %f" % (max_iters))
    print("lr_decay_iters: %f" % (lr_decay_iters))
    print("warmup_iters: %f" % (warmup_iters))
    print("batch_size: %f" % (batch_size))
    print("gradient_accumulation_steps: %f" % (gradient_accumulation_steps))
    print("block_size: %f" % (block_size))
    print("weight_decay: %f" % (weight_decay))

def get_hparams_str(model):
    if (use_nGPT == 0):
        return ""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        transformer = model.module.transformer
        config = model.module.config
        module = model.module
    else:
        transformer = model.transformer
        config = model.config
        module = model
    
    resstr = "%.5f " % torch.mean( module.sz * (module.sz_init_value/module.sz_init_scaling) )
    
    for layer_idx in range(0, config.n_layer):
        block = transformer["h"][layer_idx] 
        sqk = block.sqk * (block.sqk_init_value/block.sqk_init_scaling)
        attn_alpha = block.attn_alpha * (block.attn_alpha_init_value / block.attn_alpha_init_scaling)
        mlp_alpha = block.mlp_alpha * (block.mlp_alpha_init_value / block.mlp_alpha_init_scaling)
        suv = block.suv * (block.suv_init_value/block.suv_init_scaling)

        resstr = resstr + "%.5f " % torch.mean( sqk )
        resstr = resstr + "%.5f " % torch.mean( attn_alpha )
        resstr = resstr + "%.5f " % torch.mean( mlp_alpha )
        resstr = resstr + "%.5f " % torch.mean( suv )
         
    return resstr

stat_fname = out_dir + "/stat"
if master_process:
    if init_from == 'scratch':
        file = open(stat_fname, "w")
        resstr = f"{0:.6e} {0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0:.4e} {0.0:.4e}"
        resstr = resstr + get_hparams_str(model) + "\n"
        file.write(resstr)
        arguments = sys.argv
        fname_arg = out_dir + "/args"
        with open(fname_arg, 'w') as file_arg:
            for arg in arguments:
                file_arg.write(arg + '\n')

    if init_from == 'resume':
        file = open(stat_fname, "a")


time_spent = time.time() - tlaunch
print(f"Time spent: {time_spent} seconds")
starting_iter_num = iter_num
print("starting_iter_num: %d" % iter_num)

if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    transformer = model.module.transformer
    config = model.module.config
    module = model.module
else:
    transformer = model.transformer
    config = model.config
    module = model

def justnorm(x, idim=-1):
    dtype = x.dtype
    x = x.float()
    res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype) 
    return res

def normalize_matrices():
    transformer.wte.weight.data.copy_(justnorm(transformer.wte.weight.data, 1))         # V, n_embd
    module.lm_head.weight.data.copy_(justnorm(module.lm_head.weight.data, 1))           # V, n_embd
    

    for layer_idx in range(0, config.n_layer):
        block = transformer["h"][layer_idx]

        block.query.weight.data.copy_(justnorm(block.query.weight.data, 1))             # n_proj, n_embd
        block.key.weight.data.copy_(justnorm(block.key.weight.data, 1))                 # n_proj, n_embd
        block.value.weight.data.copy_(justnorm(block.value.weight.data, 1))             # n_proj, n_embd
        block.att_c_proj.weight.data.copy_(justnorm(block.att_c_proj.weight.data, 0))   # n_embd, n_proj

        block.c_fc.weight.data.copy_(justnorm(block.c_fc.weight.data, 1))               # n_proj, n_embd
        block.mlp_c_proj.weight.data.copy_(justnorm(block.mlp_c_proj.weight.data, 0))   # n_embd, n_proj

if (use_nGPT == 1):
    normalize_matrices()

while True:
    #sys.stdout.flush()
    if (local_iter_num > max_iters_per_launch):
        break
    if (1):
        local_seed = 100*iter_num + seed_offset # local_seed should never exceed 2.147e+9 because of np.random.seed, 100 here should be > nworkers
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed(local_seed)
        #if (iter_num % 10 == 0):    # uncomment to make sure different seeds are used
        #    print("iter: %d seed: %d" % (iter_num, local_seed))

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        rng_state_pytorch = torch.get_rng_state()
        rng_state_bytes = rng_state_pytorch.numpy().tobytes()
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
       
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr
            })

        if always_save_checkpoint:
            if iter_num > starting_iter_num:
                tcheckpointsaving_begin = time.time()
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'config': config,
                    'rng_state_pytorch_bytes': rng_state_bytes,
                    'rng_state_numpy': np.random.get_state()
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print("Checkpoint saving time: %f sec" % (time.time()-tcheckpointsaving_begin))
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    X, Y = get_batch('train')
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        #.scale(loss).backward()
        loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms")
    
    if (use_nGPT == 1):
        normalize_matrices()

    if (iter_num % 100 == 0) and master_process:
        print("lr=%f" % lr)

    if master_process:
        resstr = f"{iter_num:.6e} {lr:.4e} {losses['train']:.4e} {losses['val']:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0:.4e} {0.0:.4e} "
        resstr = resstr + get_hparams_str(model) + "\n"
        
        file.write(resstr)
        file.flush()

        if iter_num >= max_iters:
            finished_fname = out_dir + "/finished"
            finished_file = open(finished_fname, "w")
            finished_file.write("1")
            finished_file.close()

    if (time.time() - tlaunch > time_limit_seconds):
        break

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break
time_spent = time.time() - tlaunch
print(f"Time spent: {time_spent} seconds")
if ddp:
    dist.barrier()
    dist.destroy_process_group()

