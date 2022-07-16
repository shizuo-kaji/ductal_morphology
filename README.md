# Morphological metrics for 3D ductal structures
This is a companion code of the paper:

- Naoya Tanabe, Shizuo Kaji (equal contribution), Susumu Sato, Tomoki Maetani, Yusuke Shiraishi, Tsuyoshi Oguma, Ryo Sakamoto, Motonari Fukui, Shigeo Muro, Toyohiro Hirai,
[Homological features of airway tree on computed tomography and longitudinal change in lung function in patients with chronic obstructive pulmonary disease]()

## Licence
MIT Licence

### Requirements
- python 3: [Anaconda](https://anaconda.org) is recommended
- Jupter Notebook

## Description
- Open ductal_morphology.ipynb in Jupter Notebook, and follow the instruction.
- The code computes the two metrics described in the paper (treeH and radialH), and provides their visualisation in the form of labeled volume.
- treeH computes the persistent homology of the geodesic distance from the specified origin of the ductal structure, quantifying the complexity in the longitudinal direction.
- radialH computes the persistent homology of the distance from the skeleton of the ductal structure, quantifying the irregularities in the radial direction.
- Prepare a 3D volumetric image containing the ductal structure to be analysed.
The code assumes the 3D image is given in the numpy array.
If you are not sure, it is recommended to prepare the image in the NRRD format so that the example code (with the included test.nrrd) works as it is.
- The algorithm takes a binary 3D volume as input. The loaded 3D image will be binarised by a simple thresholding.

