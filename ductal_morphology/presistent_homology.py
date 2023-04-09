import cripser,tcripser
import numpy as np
import os
import cripser,tcripser
from scipy.stats import entropy

# PH computation for geodesic distances
def compute_PH(CTID,dist_vol=None,maxdim=2, OUTSIDE=1, PH_dir={},recompute_PH=True,save_numpy=True,verbosity=0):
    PH={}
    for k,mode in enumerate(PH_dir.keys()):
        if PH_dir is not None:
            phfn = os.path.join(PH_dir[mode],f'{CTID}.npy')
        else:
            phfn = None
        if os.path.isfile(phfn) and not recompute_PH:
            PH[mode] = np.load(phfn)
            if verbosity>0:
                print("precomputed PH loaded from ",phfn)
        else:
            PH[mode] = tcripser.computePH(dist_vol[mode], maxdim=maxdim) # use tcripser (instead of cripser) to account for diagonal connectivity
            if phfn is not None and save_numpy:
                np.save(phfn,PH[mode])
            # import gudhi
            # gd = gudhi.CubicalComplex(top_dimensional_cells=dist_vol[mode])
            # p = np.array(gd.persistence(2,0)) # coeff = 2
            # print("Betti numbers: ", gd.persistent_betti_numbers(np.inf,-np.inf))
        PH[mode] = PH[mode][PH[mode][:,2]<OUTSIDE] # remove cycles killed by outside region
        if verbosity>0:
            print(f'{mode}: ', ''.join([f'betti {i} {sum(PH[mode][:,0]==i)}, ' for i in range(maxdim+1)]))
        PH[mode] = PH[mode][PH[mode][:,0]==0] # focus on PH0
    return(PH)

# compute PH metrics
def PH_metrics(PH,density=False, min_life={}, max_life={}, min_birth={}, max_birth={}, num_bins={},verbosity=0):
    res = dict()
    for k,mode in enumerate(max_life.keys()):
        ph = PH[mode]
        # life and birth
        life = np.abs(ph[:,2] - ph[:,1])
        birth = np.abs(ph[:,1])
        res[f"{mode}_entropy"] = entropy(life, base=2) # / np.log2(sum(life))
        ## make histogram
        if str(max_life[mode]).endswith('%'):
            mp = float(str(max_life[mode])[:-1])
            max_life[mode] = np.percentile(life,mp)
        if str(max_birth[mode]).endswith('%'):
            mp = float(str(max_birth[mode])[:-1])
            max_birth[mode] = np.percentile(birth,mp)
        hists = {}
        bins = {}
        # filtering
        ph=ph[(min_life[mode]<=life)&(life<=max_life[mode])&(min_birth[mode]<=birth)&(birth<=max_birth[mode])]
        life = np.abs(ph[:,2] - ph[:,1])
        birth = np.abs(ph[:,1])
        hist2d, xedge, yedge = np.histogram2d(birth,life, range=([min_birth[mode],max_birth[mode]],[min_life[mode],max_life[mode]]), bins=num_bins[mode])
        #hists['Birth_Life'] = hist2d.ravel()
        hists['Life'], bins['Life'] = np.histogram(life,bins=num_bins[mode], range=(min_life[mode],max_life[mode]))
        hists['Birth'], bins['Birth'] = np.histogram(birth,bins=num_bins[mode], range=(min_birth[mode],max_birth[mode]))
        hists['Ratio'], bins['Ratio'] = np.histogram(life/np.abs(birth),bins=num_bins[mode], range=(0,1))
        for k,X in hists.items():
            if density:
                x = x/x.sum()               
            for i,y in enumerate(X):
                res[f"{mode}_{k}_{i}"] = y
        for k,X in bins.items():
            for i,y in enumerate(X):
                res[f"{mode}_{k}_bin{i}"] = y        
    return(res)
