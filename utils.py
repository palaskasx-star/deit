# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import argparse
import torch
import plotly.graph_objs as go
import plotly.io as pio
from plotly.colors import sample_colorscale

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def create_divergence_names(layer_names):
    divergence_names = layer_names.copy()
    divergence_names.append( 'y')
    print(f"divergence_names{divergence_names}")

    step_divergence_names = []
    for i in range(len(divergence_names)-1):
        step_divergence_names.append(fr'$D_J \left( p \left({divergence_names[i]}\right),p \left({divergence_names[i+1]}\right) \right)$')
    print(f"step_divergence_names{step_divergence_names}")

    label_divergence_names = []
    for i in range(len(divergence_names)-1):
        label_divergence_names.append(fr'$D_J \left(  p \left({divergence_names[-1]}\right), p \left({divergence_names[i]}\right) \right)$')
    print(f"label_divergence_names{label_divergence_names}")

    base_divergence_names = []
    for i in range(len(divergence_names)-1):
        base_divergence_names.append(fr'$D_J \left(  p \left({divergence_names[i+1]}\right), p \left({divergence_names[0]}\right) \right)$')
    print(f"base_divergence_names{base_divergence_names}")

    return step_divergence_names, label_divergence_names, base_divergence_names


def plot_divergence(divergence, plots_path, directory, layer_pair_names, title, epochs, cosine_sim):

    fig = go.Figure()

    num_epochs = divergence.shape[0]

    colors = sample_colorscale("Viridis", [i / (len(layer_pair_names) - 1) for i in range(len(layer_pair_names))])

    for i in range(len(layer_pair_names)):
        fig.add_trace(go.Scatter(
            x=list(range(1, divergence.shape[0] + 1)),  # Epochs
            y=divergence[:, i],  # KL values over epochs for one layer pair
            mode='lines',
            line=dict(color=colors[i]),
            name= layer_pair_names[i],
            hoverinfo='name+y'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            xaxis=dict(tickfont=dict(size=20)),   
            yaxis_title="divergence",
            yaxis=dict(tickfont=dict(size=20)),  
            template="plotly_white",
            legend=dict(font=dict(size=16))
        )

    # Ensure directory exists
    plot_path = os.path.join(plots_path, directory)
    os.makedirs(plot_path, exist_ok=True)

    # Save as HTML
    if cosine_sim:
        html_file = os.path.join(plot_path, title + "_lines_cos.html")
    else:
        html_file = os.path.join(plot_path, title + "_lines.html")
    pio.write_html(fig, file=html_file, auto_open=False, include_mathjax="cdn")

    # Save as PNG (requires kaleido)
    if cosine_sim:
        png_file = os.path.join(plot_path, title + "_lines_cos.png")
    else:
        png_file = os.path.join(plot_path, title + "_lines.png")
    fig.write_image(png_file, format="png", scale=2)  # scale=2 for higher resolution

    fig = go.Figure()
    colors = sample_colorscale("Viridis", [i / (num_epochs) for i in range(num_epochs)])

    for i, row in enumerate(divergence):
        fig.add_trace(go.Scatter(
            x=layer_pair_names,
            y=row,
            mode='lines',
            line=dict(color=colors[i]),
            name=f"Epoch {i+1}",
            hoverinfo='name+y'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=r"$\mathrm{D}_J\left(P_{i \mid j} \,\middle\|\, Q_{i \mid j}\right)$",
        yaxis_title="divergence",
        template="plotly_white"
    )

    # Ensure directory exists
    plot_path = os.path.join(plots_path, directory)
    os.makedirs(plot_path, exist_ok=True)

    # Save as HTML
    if cosine_sim:
        html_file = os.path.join(plot_path, title + "_cos.html")
    else:
        html_file = os.path.join(plot_path, title + ".html")
    pio.write_html(fig, file=html_file, auto_open=False, include_mathjax="cdn")

    # Save as PNG (requires kaleido)
    if cosine_sim:
        png_file = os.path.join(plot_path, title + "_cos.png")
    else:
        png_file = os.path.join(plot_path, title + ".png")
    fig.write_image(png_file, format="png", scale=2)  # scale=2 for higher resolution

