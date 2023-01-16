import queue
import numpy as np
from tqdm.auto import tqdm
from skimage.draw import line_nd
from skimage.util import invert

from .lung_ct import geodesic_distance_transform

# a variant of flood fill diffusing only to ascending neighbours
def ascending_flood(img,seed_point,tol=0, lo=-1000, high=1000):
    q = queue.Queue()
    q.put(tuple(seed_point)) # elements in the queue are tuples describing the coordinates of pixels to investigate
    neighbours = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    tovisit = np.logical_and(img>=lo, img<high) # pixels within the valid value range: !! img<high excluding equality is important. Otherwise, outside may be included
    #print(tovisit.sum())
    mask = np.zeros(img.shape,dtype=bool)
    while not q.empty():
        p = q.get()
        #print(p,lo,high,tovisit.sum())
        tovisit[p]=False
        for v in neighbours:
            nextp = tuple(np.array(p)+np.array(v)) # check the neighbouring pixel
            if 0 <= nextp[0] < img.shape[0] and 0 <= nextp[1] < img.shape[1] and 0 <= nextp[2] < img.shape[2]:
                #print(p,nextp,tovisit[nextp],mask[nextp])
                if tovisit[nextp] and img[nextp] >= img[p]+tol and not mask[nextp]: # valid range and ascending and not yet marked 
                    #print(nextp,tovisit[nextp],tovisit.shape)
                    tovisit[nextp] = False
                    mask[nextp] = True
                    q.put(nextp) # add to the queue
    return(mask)

# cycle annotated volume
def annotated_volume(binarised_volume, skeleton, origin, metrics, H, mode, annot_type, annot_metric, min_life=[], max_life=[], num_bins=[], verbosity=0, use_tqdm=True):
    out = binarised_volume.copy().astype(np.uint8)
    vol = out.sum()
    dist_vol = geodesic_distance_transform(skeleton,binarised_volume,origin,restrict_to_centerline=False)
    if mode=="radial":
        tol = 0
    else:
        tol = 0.1

    # label classes
    char_cycles = []
    for i in range(num_bins[mode]):
        if annot_metric == "birth":
            char_cycles.append({'dim':0, 'b0':-metrics[f"{mode}_Birth_bin{i+1}"], 'b1':-metrics[f"{mode}_Birth_bin{i}"], 'l0':min_life[mode], 'l1':max_life[mode], 'col': i+2 })
        elif annot_metric == "life":
            char_cycles.append({'dim':0, 'l0':metrics[f"{mode}_Life_bin{i}"], 'l1':metrics[f"{mode}_Life_bin{i+1}"], 'b0':-np.inf, 'b1':np.inf, 'col': i+2 })
        elif annot_metric == "ratio":
            char_cycles.append({'dim':0, 'l0':metrics[f"{mode}_Ratio_bin{i}"], 'l1':metrics[f"{mode}_Ratio_bin{i+1}"], 'b0':None, 'b1':None, 'col': i+2 })
            
    # counter for each label
    cnt = np.zeros(len(char_cycles))

    # normalising by the maximum birth/life values
    prog = tqdm(enumerate(H[mode]), total=len(H[mode])) if use_tqdm else enumerate(H[mode])
    for i,p in prog:
        # iterate over PH cycles
        for j,cc in enumerate(char_cycles):
            d,b0,b1,l0,l1 = cc['dim'],cc['b0'],cc['b1'],cc['l0'],cc['l1']
            if p[0] != d:
                continue
            if annot_metric=="ratio" and not (l1>=abs((p[2]-p[1])/p[1])>=l0):
                continue                
            elif annot_metric!="ratio" and not ((l1>=p[2]-p[1]>=l0) and (b1>=p[1]>=b0)):
                continue
            # cycle to annotate
            bx,by,bz = p[3:6].astype(np.int32)
            dx,dy,dz = p[6:9].astype(np.int32)
            sp = (bx,by,bz)
            if verbosity > 1:
                print("birth (z,y,x)=",sp,"death (z,y,x)=",dx,dy,dz, "birth%, life%=",b0, l0)
            cnt[j] += 1
            if annot_type == 'fill':
                mask = ascending_flood(dist_vol[mode], seed_point=sp, lo=p[1], high=p[2], tol=tol)
                #mask[bx,by,bz] = True
                if verbosity > 1:
                    print(p[1],p[2],dist_vol[mode][sp], mask.sum())
                out[mask] = cc['col']
            elif annot_type == 'line':
                lin = line_nd((sp), (dx,dy,dz), endpoint=False)
                out[lin] = cc['col']
    if verbosity > 0:
        print(f'cycle counts: ',cnt)
        print(f'label percentage', [(i+2,(out==i+2).sum()/vol) for i in range(len(char_cycles))]+[('none',(out==1).sum()/vol)])
    return(out)
