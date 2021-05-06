from threading import Thread

import torch
import GPUtil
from GPUtil import showUtilization as gpu_usage


__all__ = ["make_hook",
           "_add_memory_hooks",
           "log_mem", 
           "Monitor"]


def make_hook(logs, idx, hook_type):
    def hook(self, *args):
        torch.cuda.synchronize()
        logs.append({"layer_idx": idx,
                     "call_idx": len(logs),
                     "layer_type": type(self).__name__,
                     "hook_type": hook_type,
                     "memory_allocated": torch.cuda.memory_allocated() / 2**20,
                     "memory_cached": torch.cuda.memory_cached() / 2**20})
    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, "pre", exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, "fwd", exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, "bwd", exp))
    hr.append(h)


def log_mem(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = model(inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
