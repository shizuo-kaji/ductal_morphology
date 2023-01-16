import cripser,tcripser
import numpy as np
import os
import cripser,tcripser
from scipy.stats import entropy

# PH computation for geodesic distances
def compute_PH(dist_vol,CTID,maxdim=2,density=False, min_life=[], max_life=[], max_birth=[], num_bins=[], PH_dir=[],OUTSIDE=1,recompute_PH=True,save_numpy=True,verbosity=0):
    res = dict()
    H = dict()
    for k,mode in enumerate(max_life.keys()):
        if PH_dir is not None:
            phfn = os.path.join(PH_dir[mode],f'{CTID}.npy')
        else:
            phfn = None
        if os.path.isfile(phfn) and not recompute_PH:
            PH = np.load(phfn)
            print("precomputed PH loaded from ",phfn)
        else:
            PH = tcripser.computePH(dist_vol[mode], maxdim=maxdim) # use tcripser (instead of cripser) to account for diagonal connectivity
            if phfn is not None and save_numpy:
                np.save(phfn,PH)
            # import gudhi
            # gd = gudhi.CubicalComplex(top_dimensional_cells=dist_vol[mode])
            # p = np.array(gd.persistence(2,0)) # coeff = 2
            # print("Betti numbers: ", gd.persistent_betti_numbers(np.inf,-np.inf))

        if PH is not None:
            PH = PH[PH[:,2]<OUTSIDE] # remove cycles killed by outside region
            if verbosity>0:
                print(f'{mode}: ', ''.join([f'betti {i} {sum(PH[:,0]==i)}, ' for i in range(maxdim+1)]))
            PH = PH[PH[:,0]==0] # focus on PH0
            ## make histogram
            life = np.abs(PH[:,2] - PH[:,1])
            birth = np.abs(PH[:,1])
            if max_life[mode] is None:
                max_life[mode] = np.percentile(life,95)
            if max_birth[mode] is None:
                max_birth[mode] = np.percentile(birth,100)
            res[f"{mode}_entropy"] = entropy(life, base=2) # / np.log2(sum(life))
            H[mode] = PH
            hists = {}
            bins = {}
            hists['Life'], bins['Life'] = np.histogram(life,bins=num_bins[mode], range=(min_life[mode],max_life[mode]))
            hists['Birth'], bins['Birth'] = np.histogram(birth,bins=num_bins[mode], range=(0,max_birth[mode]))
            hists['Ratio'], bins['Ratio'] = np.histogram(life/np.abs(birth),bins=num_bins[mode], range=(0,1))
            for k,X in hists.items():
                if density:
                    x = x/x.sum()               
                for i,y in enumerate(X):
                    res[f"{mode}_{k}_{i}"] = y
            for k,X in bins.items():
                for i,y in enumerate(X):
                    res[f"{mode}_{k}_bin{i}"] = y        
    return(res,H)
