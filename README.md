# Morphological metrics for 3D ductal structures
This is a companion code of the paper:

- Naoya Tanabe, Shizuo Kaji (equal contribution), Susumu Sato, Tomoki Maetani, Yusuke Shiraishi, Tsuyoshi Oguma, Ryo Sakamoto, Motonari Fukui, Shigeo Muro, Toyohiro Hirai,
[Homological features of airway tree on computed tomography and longitudinal change in lung function in patients with chronic obstructive pulmonary disease]()

## Licence
MIT Licence

### Requirements
- python 3: [Anaconda](https://anaconda.org) is recommended
- Jupter Notebook
- Dependent packages: install by
``` 
pip install -U git+https://github.com/shizuo-kaji/CubicalRipser_3dim
pip install scikit-fmm pynrrd persim skan statsmodels
```

## Description
- Open ductal_morphology.ipynb in Jupter Notebook, and follow the instruction.
- The code computes the two metrics described in the paper (treeH and radialH), and provides their visualisation in the form of labeled volumes.
- treeH computes the persistent homology of the geodesic distance from the specified origin of the ductal structure, quantifying the complexity in the longitudinal direction.
- radialH computes the persistent homology of the distance from the skeleton of the ductal structure, quantifying the irregularities in the radial direction.
- Prepare a 3D volumetric image containing the segmented ductal structure to be analysed.
The code assumes the 3D image is given in the form of 3D numpy array.
If you are not sure, it is recommended to prepare the image in the NRRD format (as the included Test01.nrrd) so that the example code works as it is.

### Flow of the algorithm
- The algorithm takes a 3D volumetric image as input. The loaded 3D image will be binarised by a simple thresholding.
- The skeletal graph structure is computed from the skeletonised binarised volume.
- The "origin" (trachea carina) is identified as the degree three node connected to the leaf with the minimum z-coordinate (which is assumed to be closest to the mouth).
- The generation numbers are computed according to the graph distance from the origin.
- The skeletal tree is the minimum spanning tree of the skeletal graph minus those nodes from the origin to the mouth.
- The ductal volume corresponding to the skeletal tree is computed (roughly, the original volume minus those voxels in trachea or forming cycles)
- Two types of distance transform are applied to the ductal volume, and their persistent homologies are computed.
- In addition, volumed annotated with PH cycles are produced.


## Sample output
<div>
<figure>
<img src="https://github.com/shizuo-kaji/ductal_morphology/blob/main/image/generation_labeling.png?raw=true" width="30%" />
<figcaption>Generation Labeling</figcaption>
</figure>
<figure>
<img src="https://github.com/shizuo-kaji/ductal_morphology/blob/main/image/persistence_diagram.png?raw=true" width="50%" />
<figcaption>Persistence diagram</figcaption>
</figure>
<figure>
<img src="https://github.com/shizuo-kaji/ductal_morphology/blob/main/image/radial_PH_cycle_visualisation.png?raw=true" width="30%" />
<figcaption>Visualisation of persistent homology cycles for radial distance transform</figcaption>
</figure>
<figure>
<img src="https://github.com/shizuo-kaji/ductal_morphology/blob/main/image/radial_distance_transform.png?raw=true" width="30%" />
<figcaption>Radial Distance Transform</figcaption>
</figure>
</div>
