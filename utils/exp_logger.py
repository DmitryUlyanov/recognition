import os.path
import sys
import datetime
from tensorboardX import SummaryWriter
from pathlib import Path
from huepy import *
import os.path
import subprocess

class MySummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super(MySummaryWriter, self).__init__(*args, **kwargs)
        
        self.last_it = 0 


class TeeLogger(object):
    '''
        Equivalent to 

        <cmd> | tee ${filename}
    '''
    def __init__(self, filename, writer):
        self.terminal = sys.stdout
        self.log = open(filename, "w", 1)
        self.writer = writer
        
        self.w_idx= 0
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

        # if self.writer:
        #     self.writer.add_text('Log', message, self.w_idx)
        #     self.w_idx +=1 
            
    def flush(self):
        pass


def get_postfix(args, default_args, args_to_ignore, delimiter='__'):
    s = []

    for arg in sorted(args.keys()):
        if not isinstance(arg, Path) and arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}^{args[arg]}"]
    
    return delimiter.join(s).replace('/', '+')#.replace(';', '+')


def print_experiment_info(args, default_args, save_dir):
    args_v = vars(args)
    default_args_v = vars(default_args)
    
    print(bold(lightblue(' - ARGV: ')), '\n', ' '.join(sys.argv), '\n')
    # Get list of default params and changed ones    
    s_default = ''     
    s_changed = ''
    for arg in sorted(args_v.keys()):
        value = args_v[arg]
        if default_args_v[arg] == value:
            s_default += f"{lightblue(arg):>50}  :  {orange(value if value != '' else '<empty>')}\n"
        else:
            s_changed += f"{lightred(arg):>50}  :  {green(value)} (default {orange(default_args_v[arg] if default_args_v[arg] != '' else '<empty>')})\n"
            
    print(f'{bold(lightblue(" - Save dir:"))} {save_dir}\n\n')
    print(f'{bold(lightblue("Unchanged args")):>69}\n\n'
          f'{s_default[:-1]}\n\n'
          f'{bold(red("Changed args")):>68}\n\n'
          f'{s_changed[:-1]}\n')


def get_log_path(save_dir, postfix):
    filename = 'log.log'
    return os.path.join(save_dir, filename)


def setup_logging(args, default_args, args_to_ignore, exp_name_use_date=False, archive_code=True, tensorboard=True):

    time = datetime.datetime.now()

    postfix = get_postfix(vars(args), vars(default_args), args_to_ignore)
    if exp_name_use_date:
        postfix = time.strftime(f"%m-%d_%H-%M___{postfix}")

    save_dir = os.path.join(args.experiments_dir, postfix)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)
    


    if archive_code:

        paths = [f"extensions/{args.extension}/{args.config_name}.py",
                 f"extensions/{args.extension}/models/{args.model}.py",
                 f"extensions/{args.extension}/dataloaders/{args.dataloader}.py",
                 f"extensions/{args.extension}/runners/{args.runner}.py",
                 f"models/{args.model}.py",
                 f"dataloaders/{args.dataloader}.py",
                 f"runners/{args.runner}.py"
        ]

        s = ''
        for p in paths:
            if os.path.exists(p):
                s += f' <(echo {p})'

        cmd = f'cat <(git ls-files)'\
              f' <(cd extensions/{args.extension} && git ls-files | awk \'$0="extensions/{args.extension}/"$0\')'\
              f'{s}'\
              f' | sort -u | tar Tczf - "{save_dir}/source.tar.gz"'

              
        print(cmd)
        subprocess.call(["bash","-c",cmd])
    


    writer = MySummaryWriter(log_dir = save_dir, filename_suffix='_train') if tensorboard else None
    

    l = TeeLogger(get_log_path(save_dir, postfix), writer)
    sys.stdout, sys.stderr = l, l

    print_experiment_info(args, default_args, save_dir)


    # Log current name
    with open ('experiments.log', 'a') as f:
        f.write(f'{time.strftime("%I:%M%p on %B %d, %Y")}: '
                f'{save_dir}\n')


    return save_dir, writer
