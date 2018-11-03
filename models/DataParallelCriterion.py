import threading 
from encoding.parallel import Reduce
import torch.nn 
import torch

class DataParallelCriterion(torch.nn.DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """
    def forward(self, inputs, targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        if not self.device_ids:
            return self.module(inputs, targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, targets, **kwargs)

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)

        
        if isinstance(outputs[0], list) or isinstance(outputs[0], tuple):
            
            if isinstance(outputs[0][1], dict):
                first = [out_from_one_gpu[0] for out_from_one_gpu in outputs]
                first_out = Reduce.apply(*first) / len(outputs)

                out_dict = {}
                for k in outputs[0][1].keys():
                    d = [out_from_one_gpu[1][k] for out_from_one_gpu in outputs]
                    o = Reduce.apply(*d) / len(outputs)
                    out_dict[k] = o

                return first_out, out_dict

            else:
                outs = []

                for l in range(len(outputs[0])):
                    d = [out_from_one_gpu[l] for out_from_one_gpu in outputs]
                    o = Reduce.apply(*d) / len(outputs)
                    outs.append(o)

                return outs
        else:
            return Reduce.apply(*outputs) / len(outputs)

from torch.cuda._utils import _get_device_index


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
   
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
       
        torch.set_grad_enabled(grad_enabled)
       
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target,
                                          kwargs, device),)
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs