import torch
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path
import multiprocessing
from dataset import normalize_pc

BASEDIR = os.path.dirname(os.path.abspath(__file__))

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class ModelWrapper(torch.nn.Module):
    def __init__(self, model_impl) -> None:
        super().__init__()
        self.model_impl = model_impl
    
    def forward(self, data):
        pc = data[0]
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()
        res = self.model_impl(pc.transpose(1,2).cuda())
        return res
    

class SaliencyCriterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs, outputs):
        loss = F.cross_entropy(outputs.reshape(-1, 2), inputs[1].view(-1).cuda(), ignore_index=-1)
        return loss
    

class CorrespondenceCriterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs, outputs):
        kp_indexs = inputs[1]
        
        loss = []
        for b, kp_index in enumerate(kp_indexs):
            loss_rot = []
            for rot_kp_index in kp_index:
                loss_rot.append(F.cross_entropy(outputs[b][None], rot_kp_index[None].long().cuda(), ignore_index=-1))
            loss.append(torch.min(torch.stack(loss_rot)))
        loss = torch.mean(torch.stack(loss))
        return loss
    

def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    return graph_shortest_path(graph, directed=False)


def gen_geo_dists_wrapper(args):
    pc_name, pc = args
    return (pc_name, gen_geo_dists(pc))


def load_geodesics(dataset, split):
    fn = os.path.join(BASEDIR, 'cache', '{}_geodists_{}.pkl'.format(dataset.catg, split))
    # need a large amount of memory to load geodesic distances!!!
    if not os.path.exists(os.path.join(BASEDIR, 'cache')):
        os.makedirs(os.path.join(BASEDIR, 'cache'))
    if os.path.exists(fn):
        print('Found geodesic cache...')
        geo_dists = pickle.load(open(fn, 'rb'))
    else:
        print('Generating geodesics, this may take some time...')
        geo_dists = []
        with multiprocessing.Pool(processes=os.cpu_count() // 2) as pool:
            for res in tqdm(pool.imap_unordered(gen_geo_dists_wrapper, 
                                                [(dataset.mesh_names[i], normalize_pc(dataset.pcds[i]) if dataset.cfg.normalize_pc else dataset.pcds[i]) for i in range(len(dataset))]), 
                            total=len(dataset)):
                geo_dists.append(res)
        geo_dists = dict(geo_dists)
        pickle.dump(geo_dists, open(fn, 'wb'))
    return geo_dists