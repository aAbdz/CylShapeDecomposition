# Cylindrical Shape Decomposition

We develop the cylindrical shape decomposition (CSD) algorithm to decompose an object, a union of several tubular structures, into its semantic components. We decompose the object using its curve skeleton and translational sweeps. CSD partitions the object curve skeleton into maximal-length sub-skeletons over an orientation cost, each sub-skeleton corresponds to a semantic component. To find the intersection of the tubular components, CSD translationally sweeps the object in decomposition intervals to identify critical points at which the object's shape changes substantially. CSD cuts the object at critical points and assigns the same label to parts along the same sub-skeleton, thereby constructing a semantic component. CSD further reconstructs the semantic components between parts using generalized cylinders.

If you use any part of the cylindrical shape decomposition algorithm in your research, please cite:

Abdollahzadeh, A., Sierra, A. & Tohka, J. Cylindrical Shape Decomposition for 3D Segmentation of Tubular Objects. IEEE Access 9, 23979–23995 (2021).

The outline of the CSD algorithm:

<img src="figs/outline.png" width="750" height="150" />


If you had the "RuntimeError: Negative discriminant in time marcher quadratic" error in the skeletonization, you may install an older version of skfmm, e.g., 0.0.8.

## Installation
The CSD algorithm requires no installation. To install the necessary dependencies, run:

```python
pip install -r requirements.txt
```
Due to the size of datasets, we provided examples as 3d-coordinates and their corresponding bounding box in dict files: 'objSz', 'objVoxInx'.

The mAxon_mError file is an example of an under-segmentation error, where two myelinated axons incorrectly were merged. The bounding box of this 3D object is 2004x1243x180, acquired at 50 nm x 50 nm x 50 nm resolution. To retrieve an object, load the coordinates as:

```python
import numpy as np

fn = './example/mAxon_mError.npz'
data = np.load(fn)

obj_sz = data['objSz']
obj_voxInd = data['objVoxInx']
bw = np.zeros(obj_sz, dtype=np.uint8)
for ii in obj_voxInd: bw[tuple(ii)]=1 
```

### Skeletonization
To skeletonize a 3D voxel-based object, we implemented the method from Hassouna & Farag (CVPR 2005); the method detects ridges in the distance field of the object surface. If you use skeleton3D in your research, please cite:

- Abdollahzadeh, A., Sierra, A. & Tohka, J. Cylindrical Shape Decomposition for 3D Segmentation of Tubular Objects. IEEE Access 9, 23979–23995 (2021).

The implementation only requires numpy and scikit-fmm

```python
from skeleton3D import skeleton
skel = skeleton(bw)
```
You can modify the code to define the shortest path either as

- Euler shortest path: sub-voxel precise 
- discrete shortest path: more robust but voxel precise

and also to exclude branches shorter than *length_threshold* value.

With the default values, one should be able to reproduce the results in ./results/skeleton_mAx.npy. The skeletonization of the example object, 2004x1243x180 voxels, takes about 9 mins on a machine with 2xIntel Xeon E5 2630 CPU 2.4 GHz with 512 GB RAM. 

### Cylindrical shape decomposition
To apply CSD on a 3D voxel-based object, given its skeleton as *skel*, we have:

```python
from shape_decomposition import object_analysis
decomposed_image, decomposed_skeleton = object_analysis(bw, skel)
```
decomposed_image is a list, which length is equal to the number of decomposition components. The bounding box for each component is equal to the bounding box of BW image, the input object. In ./results, we provided dec_obj1 and dec_obj2 for the mAxon_mError file with an under-segmentation error. Due to the size of datasets, we provided the decomposed objects as 3d-coordinates and their corresponding bounding box in dict files: 'objSz', 'objVoxInx'.

```python
import numpy as np

fn = './example/dec_obj1.npz'
data = np.load(fn)

obj_sz = data['objSz']
obj_voxInd = data['objVoxInx']
obj1 = np.zeros(obj_sz, dtype=np.uint8)
for ii in obj_voxInd: obj1[tuple(ii)]=1
```
The same holds for obj2. 

You can modify the code to define *H_th* value, which lies in the range [0, 1]. *H_th* is the similarity threshold between cross-sectional contours. Also, setting *ub* and *lb*, where *ub>lb* also modifies the zone of interest.

DeepACSON instance segmentation, i.e., the CSD algorithm, can be run for every 3D object, i.e., an axon, independently on a CPU core. Therefore, instance segmentation of *N* axons can be run in parallel, where *N* is the number of CPU cores assigned for the segmentation task.


