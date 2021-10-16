#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
# %%
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR
from tqdm import tqdm
import re
import pickle
# %matplotlib inline

#%%
from joblib import Parallel,delayed

batchsize = 100

def compute_feature(index,norm_center_dict,afl):
    names = []
    norm_centers = []
    for name in afl.seq_list[index:index+batchsize]:
        afl_ = afl.get(name)
        path, name = os.path.split(name)
        name, ext = os.path.splitext(name)

        agent_feature, obj_feature_ls, lane_feature_ls,  norm_center = compute_feature_for_one_seq(
            afl_.seq_df, am, OBS_LEN, LANE_RADIUS, OBJ_RADIUS, viz=False, mode='nearby')
        df = encoding_features(
            agent_feature, obj_feature_ls, lane_feature_ls )

        # if VIS:
        #     visualize_vectors(df)
        save_features(df, name, os.path.join(
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"))
        names.append(name)
        norm_centers.append(norm_center)
    return names,norm_centers




if __name__ == "__main__":
    am = ArgoverseMap()
    for folder in os.listdir(DATA_DIR):
        if not re.search(r'forecasting_sample', folder):
        # FIXME: modify the target folder by hand ('val|train|sample|test')
        # if not re.search(r'test', folder):
            continue
        print(f"folder: {folder}")
        data_folder = folder + "/data"
        afl = ArgoverseForecastingLoader(os.path.join(DATA_DIR, data_folder))
        seq_list_len = len(afl.seq_list)
        norm_center_dict = {}
        
        names_list, norm_centers_list = zip(*Parallel(n_jobs=-1)(delayed(compute_feature)(
            i,
            norm_center_dict,
            afl,) for i in tqdm(range(0,seq_list_len,batchsize))))
        
        for names, norm_centers in tqdm(zip(names_list, norm_centers_list)):
            for name, norm_center in zip(names, norm_centers):
                norm_center_dict[name] = norm_center
            # agent_feature, obj_feature_ls, lane_feature_ls, norm_center = compute_feature_for_one_seq(
            #     afl_.seq_df, am, OBS_LEN, LANE_RADIUS, OBJ_RADIUS, viz=False, mode='nearby')
            # df = encoding_features(
            #     agent_feature, obj_feature_ls, lane_feature_ls)
            # save_features(df, name, os.path.join(
            #     INTERMEDIATE_DATA_DIR, f"{folder}_intermediate"))

            # norm_center_dict[name] = norm_center
        
        with open(os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}-norm_center_dict.pkl"), 'wb') as f:
            pickle.dump(norm_center_dict, f, pickle.HIGHEST_PROTOCOL)
            # print(pd.DataFrame(df['POLYLINE_FEATURES'].values[0]).describe())


# %%


# %%
