from data import *
from torch.utils.data import DataLoader, Subset

def get_config(name):
    if name == 'PIE':
        config = {
        'batch_size': 100,
        'lr': 0.001,
        'layer_num': 3,
        'epochs': 500
        }
        dataloader = PIE()
        
    elif name == 'Scene':
        config = {
        'batch_size': 100,
        'lr': 0.001,
        'layer_num': 3,
        'epochs': 500
        }
        dataloader = Scene()
        
    elif name == 'LandUse':
        config = {
        'batch_size': 100,
        'lr': 0.001,
        'layer_num': 3,
        'epochs': 500
        }   
        dataloader = LandUse()
        
    elif name == 'HW':
        config = {
        'batch_size': 100,
        'lr': 0.001,
        'layer_num': 3,
        'epochs': 500
        }     
        dataloader = HandWritten()
        
    elif name == 'NUSOBJ':
        config = {
        'batch_size': 400,
        'lr': 0.0002,
        'layer_num': 3,
        'epochs': 500
        }
        dataloader = NUSOBJ() 
        
    elif name == 'Fashion':
        config = {
        'batch_size': 400,
        'lr': 0.0002,
        'layer_num': 2,
        'epochs': 500
        }
        dataloader = Fashion()
        
    elif name == 'Leaves':
        config = {
        'batch_size': 100,
        'lr': 0.001,
        'layer_num': 2,
        'epochs': 500
        }
        dataloader = Leaves()
        
    elif name == 'MSRC':
        config = {
        'batch_size': 100,
        'lr': 0.001,
        'layer_num': 1,
        'epochs': 500
        }
        dataloader = MSRC()
        
    return config, dataloader

def get_dataloader(name, conflictive_test):
    config, dataloader = get_config(name)
    
    print('-' * 80)
    print('dataset name:', name)
    for key in config:
        print(key, ": ", config[key])
    print('-' * 80)

    num_samples = len(dataloader)
    dims = dataloader.dims

    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    if conflictive_test == True:
        dataloader.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)
    
    print("dims = ",  dims)

    train_loader = DataLoader(Subset(dataloader, train_index), batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(Subset(dataloader, test_index), batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader