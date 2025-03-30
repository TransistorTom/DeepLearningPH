from pysr import PySRRegressor
import numpy as np
import torch
import pandas as pd

import sys
import os

# paths and roots
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

save_dir=os.path.join(project_root,"results", "symbolic_regression")
os.makedirs(save_dir, exist_ok=True)

def sym_reg(job_id, timestamp, N_test = [3,4,5], train = True, test = True):
    
    # paths and roots
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
    sys.path.append(project_root)
    save_dir=os.path.join(project_root,"results", "symbolic_regression")
    os.makedirs(save_dir, exist_ok=True)
    load_path = os.path.join(project_root, f"results/training_habrok/job{job_id}_{timestamp}")
    load_path = load_path("16006282", "2025-03-29_23-03-05")

    if train:
        train_df = pd.read_csv(os.path.join(load_path, "train_messages.csv"))
        
        train_df['dx'] = train_df['pos_i_x'] - train_df['pos_j_x']
        train_df['dy'] = train_df['pos_i_y'] - train_df['pos_j_y']
        train_df['r'] = np.sqrt(train_df['dx']**2 + train_df['dy']**2)
        train_df['r3'] = train_df['r']**3

        train_X = train_df[['mass_j','mass_i', 'dx', 'r']]
        train_Y = train_df[['mass_j','mass_i', 'dy', 'r']]
        train_y_x = train_df['message_x']
        train_y_y = train_df['message_y']

        train_model_x = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*"],
            model_selection="score",  # Select best tradeoff between complexity and error
            select_k_features=4, 
            extra_sympy_mappings={"inv_r3": lambda r: 1 / r**3}
            )

        x_trained = train_model_x.fit(train_X.values, train_y_x.values, variable_names = ['mass_j','mass_i', 'dx', 'r'])

        train_model_y = PySRRegressor(
                niterations=100,
                binary_operators=["+", "-", "*"],
                model_selection="score",
                select_k_features=4,  # small number of features
                extra_sympy_mappings={"inv_r3": lambda r: 1 / r**3}
            )

        y_trained = train_model_y.fit(train_X.values, train_y_y.values, variable_names = ['mass_j','mass_i', 'dy', 'r'])
    
    if test:    
    
        test_dfs = {}
    
        for N in N_test: 
            test_dfs[N] = pd.read_csv(os.path.join(load_path, "test_messages.csv"))

    return x_trained, y_trained
