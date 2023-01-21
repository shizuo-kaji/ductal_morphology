#%% compute surface area of 3D array by Crofton's formula
## original by https://cvtech.cc/crofton-surface-area/

import numpy as np
import skimage.morphology
import os
import pandas as pd
from tqdm.notebook import tqdm
import nrrd
import itertools

def crofton_surface_area(X): # X: 3D binary np.array
    padded = np.pad(X,1).astype(np.uint8)
    result = 0
    for axis1 in range(3):
        result += np.abs(np.sign(np.diff(padded,axis=axis1))).sum()
        for axis2 in range(axis1+1,3):
            for sgn in [1, -1]:
                rolled = np.copy(padded)
                for i in range(padded.shape[axis1]):
                    slicer = tuple([(slice(None) if j != axis1 else [i]) for j in range(3)])
                    rolled[slicer] = np.roll(rolled[slicer], i*sgn, axis=axis2)
                result += np.abs(np.sign(np.diff(rolled, axis=axis1))).sum()* (2**-0.5)

    for sgn1,sgn2 in itertools.product([1,-1],[1,-1]):
        rolled = np.copy(padded)
        for i in range(padded.shape[0]): # diagonal slide
            rolled[[i]] = np.roll(rolled[[i]], i*sgn1, axis=1)
            rolled[[i]] = np.roll(rolled[[i]], i*sgn2, axis=2)
        result += np.abs(np.sign(np.diff(rolled, axis=0))).sum() * (3**-0.5)

    return (2*result/13) # taking the mean

# check with ball
def test_ball(r):
    X = skimage.morphology.ball(r)
    print(X.sum(), 4/3*np.pi*r**3)
    print(crofton_surface_area(X), 4*np.pi*r**2)

#%%
if __name__== "__main__":

    #test_ball(30)

    root_dir = os.path.expanduser("D:/ML/CT/longitude/airway_woTracheaFC56_20220227")
    #df_file = os.path.expanduser("longitudinal_PH/MDlong_KIT_airwaylungdata_firstset_20220324.csv")
    df_file = os.path.expanduser("longitudinal_PH/MDdata_2ndData_3year_20220324.csv")
    idcol = "CTID"

    df = pd.read_csv(df_file, header=0)
    area, volume = [],[]
    for i in tqdm(range(len(df))):
        fn = df.loc[i,idcol]
        dfn = os.path.join(root_dir,fn+".nrrd")
        X, header = nrrd.read(dfn, index_order='C')
        X = (X>-2000)
        area.append(crofton_surface_area(X))
        volume.append(X.sum())

    pd.DataFrame({idcol: df[idcol], 'area': area, 'volume': volume}).to_csv("area_volume.csv", index=False)
# %%
