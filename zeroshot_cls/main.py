import os
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from dassl.engine import build_trainer
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed, collect_env_info

import datasets.scanobjnn
import datasets.modelnet40
from trainers import best_param

from trainers import zeroshot
from trainers.post_search import search_weights_zs, search_prompt_zs

def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    cfg.TRAINER.EXTRA = CN()


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup_cfg(args)
    
    # set random seed
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)
    print("\nTrainer: ", trainer)

    # zero-shot classification
    if args.zero_shot:
        trainer.test_zs()

    

    # TODO: iterate through all states and experiment data and promps and save the scores to dictionaries to be loaded by plotting script
    pointclip_full_dict = {}
    pointclip_full_dict_w ={}

    # create n_steps dictionary to store the number of steps for each shape
    n_steps = {'flower': 7, 
            'X': 8, 
            'ring': 6, 
            'heart': 10, 
            'column': 5, 
            'house': 11, 
            'tree': 8, 
            'cone': 8}
    
    base_path = '/home/alison/Documents/GitHub/point_flow_actor/experiments/discrete_dough_human/Exp1_'
    shape_list = ['flower', 'X', 'ring', 'heart', 'column', 'house', 'tree', 'cone']
    # iterate through shapes
    for shape in tqdm(shape_list):
        # create a dictionary for each shape
        pointclip_shape_dict = {}
        pointclip_shape_dict_w = {}

        for i in range(6): # there are 6 prompt variations currently
            pointclip_shape_dict[i] = np.zeros(n_steps[shape])
            pointclip_shape_dict_w[i] = np.zeros(n_steps[shape])
        
        # iterate through trajectory states
        for j in range(n_steps[shape]):
            # load in the image and crop
            full_path = base_path + shape + '/state' + str(j) + '_pcl.npy'
            pcl = np.load(full_path)

            # if pcl.shape[0] < 2048:
            #     # randomly duplicate points until there are 2048 points
            #     while pcl.shape[0] < 2048:
            #         pcl = np.vstack((pcl, pcl[np.random.choice(pcl.shape[0], 1, replace=False)]))
            #     print("Shape: ", pcl.shape)
            # else:
            #     pcl = pcl[np.random.choice(pcl.shape[0], 2048, replace=False)]
            
            # conver point cloud to tensor on cuda and add in batch dimension
            pc = torch.tensor(pcl).cuda().unsqueeze(0).float()

            prompts = [f'{shape}', f'simple {shape}', f'a 2D {shape}', f'a clay {shape}', f'a {shape} shape in clay', f'a simple {shape} sculpture'] 
        
            # iterate through prompts
            for k in range(len(prompts)):
                text = prompts[k]
                cosine_sim, cosine_sim_w = trainer.get_pointclip_score(pc, text)
                pointclip_shape_dict[k][j] = cosine_sim
                pointclip_shape_dict_w[k][j] = cosine_sim_w
        
        pointclip_full_dict[shape] = pointclip_shape_dict
        pointclip_full_dict_w[shape] = pointclip_shape_dict_w

    print("\nPointclip full dictionary: ", pointclip_full_dict)
    print("\nPointclip full dictionary w: ", pointclip_full_dict_w)
    dict_save_path = '/home/alison/Documents/GitHub/point_flow_actor/experiments/discrete_dough_human'
    # save the dictionary with pickle
    with open(dict_save_path + '/no_downsample_pointclip_full_dict.pkl', 'wb') as f:
        pickle.dump(pointclip_full_dict, f)
    with open(dict_save_path + '/no_downsample_pointclip_full_dict_w.pkl', 'wb') as f:
        pickle.dump(pointclip_full_dict_w, f)

    # pcl = np.load('/home/alison/Documents/GitHub/point_flow_actor/experiments/discrete_dough_human/Exp1_X/state7_pcl.npy')
    # text = "a clay X"
    # # downsample pcl to 2048 points for consistency with training data 
    # pcl = pcl[np.random.choice(pcl.shape[0], 2048, replace=False)]
    # # conver point cloud to tensor on cuda and add in batch dimension
    # pc = torch.tensor(pcl).cuda().unsqueeze(0).float()
    # print("Point cloud shape: ", pc.shape)
    # cosine_sim, cosine_sim_w = trainer.get_pointclip_score(pc, text)
    
    assert False
        
    # view weight and prompt search
    vweights = best_param.best_prompt_weight['{}_{}_test_weights'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)]
    prompts = best_param.best_prompt_weight['{}_{}_test_prompts'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)]
    if args.post_search:
        if args.zero_shot:
            prompts, image_feature = search_prompt_zs(cfg, vweights, searched_prompt=prompts)
            #vweights = search_weights_zs(cfg, prompts, vweights, image_feature)
            return
            
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='', help='output directory')
    parser.add_argument('--seed', type=int,default=2,help='only positive value enables a fixed seed')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation methods')
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--dataset-config-file', type=str, default='', help='path to config file for dataset setup')
    parser.add_argument('--trainer', type=str, default='', help='name of trainer')
    parser.add_argument('--backbone', type=str, default='', help='name of CNN backbone')
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument('--zero-shot', action='store_true', help='zero-shot only')
    parser.add_argument('--post-search', default=True, action='store_true', help='post-search only')
    parser.add_argument('--model-dir', type=str, default='',help='load model from this directory for eval-only mode')
    parser.add_argument('--load-epoch', type=int, default=175, help='load model weights at this epoch for evaluation')
    parser.add_argument('--no-train', action='store_true', help='do not call trainer.train()')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='modify config options using the command-line')
    args = parser.parse_args()
    main(args)
    
