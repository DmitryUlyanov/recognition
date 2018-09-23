import yaml 

def save_yaml(what, where):
    with open(where, 'w') as f:
        f.write(yaml.dump(what, default_flow_style = False))
        
def load_yaml(where):
    with open(where, 'r') as f:
        config = yaml.load(f)