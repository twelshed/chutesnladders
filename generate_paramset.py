from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import loguniform
import json
import os
import numpy as np
import random
import string

if __name__ == "__main__":

    letters = string.ascii_letters
    batch_id = ''.join(random.choice(letters) for i in range(4))
    exp_id = ''
    grid = {
              'batch_id' : [batch_id],
              'exp_id' :[exp_id],
              'env_y': np.linspace(1e-6, 3e-5,1),
              'membrane': ['sigmoid','step'],
              'stick_mag': np.linspace(0,1,1),
              'sticking_time': np.linspace(0,100,1)}

    griditer = ParameterGrid(grid)

    if not os.path.exists(f'paramset/{batch_id}'):
        os.mkdir(f'paramset/{batch_id}')
    else:
        print('paramset batch directory exists, you should clear it if you dont want to overwrite. Paramsets will be overwritten. Continueing.')
    for i, params in enumerate(griditer):
        exp_id = ''.join(random.choice(letters) for i in range(6))
        params['exp_id'] = exp_id
        with open(f"paramset/{batch_id}/{exp_id}.json", "w") as outfile:
            json.dump(params, outfile)