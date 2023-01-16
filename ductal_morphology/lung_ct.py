#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from genericpath import isfile
import os
import skfmm
from scipy.ndimage import distance_transform_edt,binary_fill_holes
from skimage.morphology import skeletonize,medial_axis,dilation
import skimage
import numpy as np
from genericpath import isfile
import skan
import networkx as nx
from tqdm.auto import tqdm


# find fattest node ID and its coorrdinates
def fattest_node(skeleton_img,mask,skel):
    inds = distance_transform_edt(np.ma.masked_array(~skeleton_img,~mask),return_distances=False,return_indices=True)
    inds = inds[:,mask]
    inv_dt = np.zeros(skeleton_img.shape)
    for x,y,z in tqdm(zip(inds[0],inds[1],inds[2]),total=inds[0].size):
        inv_dt[x,y,z]+=1
    degrees = np.diff(skel.graph.indptr)
    c=skel.coordinates[degrees>2].astype(int)
    fattest_node_id = np.argmax(inv_dt[c[:,0],c[:,1],c[:,2]])
    return(fattest_node_id,c[fattest_node_id])


# count num of connected components
def count_cc(img, bg_color=-1):
    cc = {}
    for val in np.unique(img):
        if val != bg_color:
            labels = skimage.measure.label(img==val, background=0)
            cc[val] = labels.max()
    return(cc)

# select largest connected component if there are more than one connected components
def largest_connected_component(binary_img,verbosity=0):
    cc = skimage.measure.label(binary_img)
    if cc.max() != 1:
        print("The input volume must be connected! The result will be incorrect! Check the threshold.")
        area = [(cc==val).sum() for val in range(1,cc.max()+1)]
        print(f"Areas {area}...selecting the largest component")
        largest_cc_val = np.argsort(area)[-1]+1
        binary_img=(cc==largest_cc_val)
    if verbosity>1:
        print(f"binarised image #connected components: volume {cc.max()}")
    return(binary_img)

# compute the graph structure of the skeleton
def create_skeleton_graph(skeleton, graph_creation="networkx", verbosity=0):
    skel = skan.csr.Skeleton(skeleton) #source_image=volume)
    if graph_creation=="skan":
        #NOTE: as of Skan 0.10, the detected skeleton can be disconnected even when the skelton image is connected.
        #This causes removal of relevant airway branches since we consider the largest connected component
        #degrees = np.diff(skel.graph.indptr) # list of degrees of the tree points
        #node_ids = np.where(degrees!=2)[0][1:] # list of IDs of nodes in the tree; the ID=0 is a dummy so that we remove it
        branch = skan.csr.branch_statistics(skel.graph)
        path_coords = [skel.path_coordinates(i).astype(int) for i in range(skel.n_paths)]
        paths = np.vstack([skel.paths.indices[skel.paths.indptr[:-1]],skel.paths.indices[skel.paths.indptr[1:] - 1]]).T # paths[i] = [src[i],dst[i]]
        edges = paths # branch[:,:2].astype(int) # list of edges: [[Node ID, Node ID],...]
        skeleton_graph = nx.Graph()
        skeleton_graph.add_weighted_edges_from([e[0],e[1],w] for e,w in zip(edges,branch[:,2]))
        skeleton_MST = nx.minimum_spanning_tree(skeleton_graph) # remove cycles from the skeleton by taking the minimum spanning tree
    else:
        # degree 2 nodes are contracted
        skeleton_graph = nx.from_scipy_sparse_array(skel.graph)
        G = skeleton_graph.copy()
        for node in list(G.nodes()):
            if G.degree(node) == 2:
                edges = list(G.edges(node))
                G.add_edge(edges[0][1], edges[1][1])
                G.remove_node(node)
        # remove cycles from the skeleton by taking the minimum spanning tree
        skeleton_MST = G.subgraph(nx.minimum_spanning_tree(G).nodes) # this may still contain cycles
        skeleton_MST = nx.minimum_spanning_tree(skeleton_MST)
        #skeleton_MST.remove_edges_from(nx.selfloop_edges(skeleton_MST))
    skeleton_MST.remove_node(0) # 0 is always a dummy node in Skan's convention
    nx.set_node_attributes(skeleton_MST, {v: skel.coordinates[v].astype(int) for v in skeleton_MST.nodes}, 'coords')
    if verbosity>0:
        degrees = np.array([skeleton_graph.degree[i] for i in skeleton_graph.nodes])
        print(f'initial skeleton graph #nodes {(degrees !=2).sum()}, #path {skel.n_paths}, #voxels {skel.graph.shape[0]-1}, {(skeleton>0).sum()}, #nonzero in adj {skel.graph.count_nonzero()}')

    return(skeleton_MST, skel)

# find trachea carina and nodes contained in trachea
def find_trachea(skeleton_MST, min_branch_children=20, min_branch_separation=0, verbosity=0):
    vert_coords = np.array([skeleton_MST.nodes[v]['coords'] for v in skeleton_MST.nodes],dtype=int)
    degrees = np.array([skeleton_MST.degree[i] for i in skeleton_MST.nodes])
    c=vert_coords[degrees==1] # leaf coordinates
    origin_id = list(skeleton_MST.nodes)[np.where(degrees==1)[0][np.argmin(c[:,0])]] # the leaf with minimum z-coordinates (should corresponds to throat)
    height = np.ptp(c[:,0])
    trachea_nodes = [] # remove nodes in trachea
    bfs = nx.bfs_successors(skeleton_MST,source=origin_id)
    bfsT= nx.bfs_tree(skeleton_MST,source=origin_id)
    parent = {origin_id:origin_id}
    origin_id,succ = next(bfs)
    for k in succ:
        parent[k]=origin_id
    if verbosity>0:
        print(f"Highest vert ID: {origin_id}, coord: {skeleton_MST.nodes[origin_id]['coords']}, height: {height}")
    while True: # identify the carina node
        if skeleton_MST.degree[origin_id] < 3: 
            #if skeleton_MST.coords[parent[origin_id]][0] <= skeleton_MST.coords[origin_id][0]:
            trachea_nodes.append(origin_id)
                #origin_id = dfs[origin_id][0]
        else:
            xyz = skeleton_MST.nodes[origin_id]['coords']
            num_children = np.array([len(nx.descendants(bfsT,s)) for s in succ])
            argn = np.argsort(num_children)
            s1,s2 = succ[argn[-1]], succ[argn[-2]]
            xyz1,xyz2 = skeleton_MST.nodes[s1]['coords'],skeleton_MST.nodes[s2]['coords']
            # d1 = nx.descendants_at_distance(bfsT,succ[0],1) | nx.descendants_at_distance(bfsT,succ[0],2)
            # d2 = nx.descendants_at_distance(bfsT,succ[1],1) | nx.descendants_at_distance(bfsT,succ[1],2)
            # elif bool(d1 & d2): # this should not happen in MST
            #     if verbose:
            #         print("confluent branches:", d1,d2,origin_id,succ)
            #     trachea_nodes.append(origin_id)
            if num_children[argn[-2]]<min_branch_children: # branches are too small
                if verbosity>0:
                    print(f"one of the branches of {origin_id} is too small having {num_children[argn[-2]]}, {num_children[argn[-1]]} children",succ)
                trachea_nodes.append(origin_id)
            elif abs(xyz1[2]-xyz2[2])<min_branch_separation: # branches are too close
                if verbosity>0:
                    print(f"children of {origin_id} are too close in x-coords:",xyz1,xyz2,succ)
                trachea_nodes.append(origin_id)
            elif xyz[0]<height/5:
                if verbosity>0:
                    print(f"{origin_id} has too small z-coords:",xyz)
                trachea_nodes.append(origin_id)
            else:
                break # carina found!
        # not found and continue
        try:
            origin_id,succ = next(bfs)
        except StopIteration:
            print("Cannot find trachea carina!")
            break
        for k in succ:
            parent[k]=origin_id
        if verbosity>0:
            print(f"traversing node {origin_id} at {skeleton_MST.nodes[origin_id]['coords']} with childeren {succ}")
    return(origin_id,trachea_nodes)        

# remove degree two nodes and degree one nodes (leafs) with small generation.
def removed_deg2_nodes(skeleton_MST, carina_id, leaf_removal_max_generation=3, verbosity=0):
    node_removed = True
    removed_deg2, removed_leaf = 0,0
    while node_removed:
        node_removed = False
        gens = nx.shortest_path_length(skeleton_MST, source=carina_id, weight=None)
        for node in list(skeleton_MST.nodes()):
            if skeleton_MST.degree(node) == 1 and node != carina_id and gens[node]<= leaf_removal_max_generation:
                skeleton_MST.remove_node(node)            
                node_removed = True
                removed_leaf += 1
                if verbosity>1:
                    print(f"the leaf node {node} at generation {gens[node]} is removed.")
            elif skeleton_MST.degree(node) == 2 and node != carina_id:
                #print(f'node {node} removed!')
                edges = list(skeleton_MST.edges(node))
                skeleton_MST.add_edge(edges[0][1], edges[1][1])
                skeleton_MST.remove_node(node)
                node_removed = True
                removed_deg2 += 1
                if verbosity>1:
                    print(f"the degree 2 node {node} at generation {gens[node]} is removed.")
    if verbosity>0:
        print(f"#degree 2 nodes removed {removed_deg2}, #leaf nodes removed {removed_leaf}")
    return(skeleton_MST,gens)

#
def create_generation_volumes(volume,skeletonize_method=None, graph_creation="networkx", threshold=-2000, min_branch_children=20, min_branch_separation=0, remove_trachea=True, verbosity=0):
    """Computes 3d array whose values indicate generation numbers

    Args:
        volume (float numpy 3d array): airway CT volume

    Returns:
        skeleton_generation
        volume_generation
        trachea_removed: 
        origin (tuple(int,int,int)): the coordinate of the trachea carina
        ac_gens (dict): dictionary of {generation: airway counts in generation}
    """
    ## represent airway tree struture as a weighted graph
    binarised_volume = binary_fill_holes(volume>threshold) # binarised airway volume
    binarised_volume = largest_connected_component(binarised_volume, verbosity)
    skeleton = skeletonize(binarised_volume,method=skeletonize_method) # skelton(centerline)
    if verbosity>0:
        print(f'skeleton #connected components {skimage.measure.label(skeleton).max()}')
    ## compute the graph structure of the skeleton
    skeleton_MST, skel = create_skeleton_graph(skeleton, graph_creation=graph_creation, verbosity=verbosity)
    if verbosity>0:
        print(f'minimum spanning tree #vertices {len(skeleton_MST.nodes())}, #edges {len(skeleton_MST.edges())}')
    ## create skeleton binary image without cycles
    skeleton_graph = nx.from_scipy_sparse_array(skel.graph)
    skeleton_cleaned = np.zeros_like(skeleton)
    for e in skeleton_MST.edges:
        voxels = nx.shortest_path(skeleton_graph,e[0],e[1])  # node indices contained in the path
        c = np.array([skel.coordinates[v] for v in voxels],dtype=int) # node coordinates contained in the path
        skeleton_cleaned[c[:,0],c[:,1],c[:,2]]=255
    skeleton_dt = distance_transform_edt(~skeleton_cleaned) # this will be compared to determine the voxels to be removed from the original volume

    ## identify trachea carina as the first deg>2 node with enough children
    carina_id, trachea_nodes = find_trachea(skeleton_MST, min_branch_children=min_branch_children, min_branch_separation=min_branch_separation, verbosity=verbosity)
    if verbosity>0:
        print("Trachea Carina: ",carina_id,skeleton_MST.nodes[carina_id]['coords']," trachea vertices: ",[(k,skeleton_MST.nodes[k]['coords']) for k in trachea_nodes]) # skeleton_MST[k]
    ## construct the airway tree rooted at the trachea carina: remove vertices in trachea
    MST_nodes = set(skeleton_MST.nodes())
    if remove_trachea:
        skeleton_MST.remove_nodes_from(trachea_nodes)
    #nx.write_gexf(skeleton_MST, "MST2.gexf")
    cc = max(nx.connected_components(skeleton_MST), key=len) # take the largest connected component
    if verbosity>0:
        print("removed vertices: ",MST_nodes-set(cc)) # removed nodes
        # coord_nodes = skeleton_MST.coords[degrees!=2]
        # for x in list(set(skeleton_MST.nodes())-set(cc)):
        #     dist = ((skeleton_MST.coords[x]-coord_nodes)**2).sum(axis=1)
        #     print(x, np.sort(dist)[1] )
    #
    skeleton_MST = skeleton_MST.subgraph(cc).copy()
    if verbosity>0:
        print(f'skeleton graph after trachea removal #vertices {len(skeleton_MST.nodes())}, #edges {len(skeleton_MST.edges())}')
    ## remove degree two nodes once again (except for the root) and leaf nodes with small generation
    skeleton_MST,gens = removed_deg2_nodes(skeleton_MST, carina_id, leaf_removal_max_generation=3, verbosity=verbosity)
    # airway conunt by generations
    ac_gens = {}
    i=0
    while True:
        ac = list(gens.values()).count(i)
        if ac == 0:
            break
        ac_gens[i] = ac
        i += 1
    if verbosity>0:
        print(f"#nodes at gens: {ac_gens}")
    # final data
    attr = {node: {'gen':gens[node], 'deg':skeleton_MST.degree(node)} for node in skeleton_MST}
    nx.set_node_attributes(skeleton_MST,attr)

    origin = tuple(skeleton_MST.nodes[carina_id]['coords']) # origin(=trachea carina) coordinates for geodesic distance for treeH
    if binarised_volume[origin]==0:
        print("The root must lie inside the ductal structure!")

    ## assign airway generation to the skeleton tree
    skeleton_generation = np.zeros(skeleton.shape,dtype=np.uint16)
    for e in skeleton_MST.edges:
        voxels = nx.shortest_path(skeleton_graph,e[0],e[1])  # node indices contained in the path
        c = np.array([skel.coordinates[v] for v in voxels],dtype=int) # node coordinates contained in the path
        skeleton_generation[c[:,0],c[:,1],c[:,2]]=max(gens[e[0]],gens[e[1]])
    print(f'maximum generation {skeleton_generation.max()}, #connected components {skimage.measure.label(skeleton_generation>0).max()}, #vertices {len(skeleton_MST.nodes())}, #edges {len(skeleton_MST.edges())}, #leaves {sum([i[1]==1 for i in skeleton_MST.degree])}')

    ## compute trachea removed volume: those voxels having different distance from the original and the final tree centerlines will be removed
    skeleton_MST_dt,inds = distance_transform_edt(skeleton_generation==0,return_indices=True)
    mask = (skeleton_dt != skeleton_MST_dt) ## regions to be removed
    trachea_removed = volume.copy()
    trachea_removed[mask] = volume.min()
    binarised_volume = binary_fill_holes(trachea_removed>threshold) # binarised airway volume

    ## airway generation volume
    volume_generation=skeleton_generation[inds[0],inds[1],inds[2]].reshape(volume.shape)
    volume_generation[~binarised_volume]=0
    if skimage.measure.label(skeleton_generation>0).max()>1:
        print("the volume is disconnected!")

    return(skeleton_generation,volume_generation,trachea_removed,origin,ac_gens,skeleton_MST)


# compute the signed distance from the origin and the centerline
def geodesic_distance_transform(skeleton,binarised_volume,origin,outside_fill=1,restrict_to_centerline=True):
    roi = np.ones(skeleton.shape)
    roi[origin] = 0
    # negative of geodesic distance transform: 0 means the voxel on the ROI, 1 means outside.
    dist_vol = {"tree": (-skfmm.distance(np.ma.MaskedArray(roi,~binarised_volume))).filled(fill_value=outside_fill)}
    dist_vol["radial"] = (-skfmm.distance(np.ma.MaskedArray(~skeleton,~binarised_volume))).filled(fill_value=outside_fill)
    if restrict_to_centerline:
        dist_vol["tree"][~skeleton]=outside_fill # restrict to the centerline
        
    #dist_vol["tree"]=(-skfmm.distance(np.ma.MaskedArray(roi,~dilation(skeleton)))).filled(fill_value=OUTSIDE)
    return(dist_vol)

