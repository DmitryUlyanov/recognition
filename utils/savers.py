import math
import torch
from torch.optim.optimizer import Optimizer

from huepy import red
import torch
import sys

def get_saver(name, args):
    if name in sys.modules[__name__].__dict__:
        return sys.modules[__name__].__dict__[name](args)
    else:
        assert False, red(f"Cannot find saver with name {name}")



class Saver(object):
    
    def __init__(self, args, save_fn, tq_maxsize = 5, clean_dir=True, num_workers=5):
        super(Saver, self).__init__()
        self.args = args

        self.save_dir = args.dump_path
        self.need_save = False
        if 'save_driver' in args and args.save_driver is not None:
            
            # print('-----------------')
            if clean_dir and os.path.exists(args.dump_path):
                shutil.rmtree(args.dump_path) 

            os.makedirs(args.dump_path, exist_ok=True)

            self.tq = TaskQueue(maxsize=tq_maxsize, num_workers=num_workers, verbosity=0) 

            self.save_fn = save_fn
            self.need_save = True

    def maybe_save(self, iteration, **kwargs):
        if self.need_save:
            self.tq.add_task(self.save_fn, kwargs, save_dir=self.save_dir, args=self.args, iteration=iteration)  

    def stop(self):
        if self.need_save:
            self.tq.stop_()


class DummySaver(object):
    
    def __init__(self, *args,  **kwargs):
        super().__init__()
      

    def maybe_save(self, iteration, **kwargs):
        pass

    def stop(self):
        pass