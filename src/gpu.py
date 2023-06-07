#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


import argparse
import time
import math
import os, sys
import itertools
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


def add_gpu_params(parser: argparse.ArgumentParser):
    parser.add_argument("--random_seed", default=10, type=int, help="random seed")


def distributed_opt(model, opt, grad_acc=1):
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        output_device=torch.cuda.current_device(),
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    return model, opt


def distributed_gather(tensor):
    g_y = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(g_y, tensor, async_op=False)
    return torch.stack(g_y)


def parse_gpu(args):
    torch.manual_seed(args.random_seed)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    # Set the device for each process
    args.device = torch.device("cuda", args.local_rank)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.device)
    print("args.device:", args.device)

    print(
        ("global_rank:", args.rank),
        ("local_rank:", args.local_rank),
        ("device_count:", torch.cuda.device_count()),
        ("world_size:", args.world_size),
        sep="\n"
    )

def cleanup():
    torch.distributed.destroy_process_group()

