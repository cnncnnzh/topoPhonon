# topoPhonon
#### topoPhonon is a python package that allows users to phononic calculate topological properties, by building phonon tight-binding model.
* Build tight-binding models from user's input, FORCE_CONSTANTS files or Phonopy API. The last method is much faster, but need to import Phonopy package and manually create phonopy.harmonic.dynamical_matrix.DynamicalMatrix object.  
* Calculate berry phase, berry curvature, wannier charge center evolution around Weyl points from tight-binding model.  
* Build slab/ribbon models for surface/edge.  
* Plot 3d band surfaces.   

## Installation:

`pip install topophonon` 

## Basic usage:  

Use one line of code to build tight-binding model:  
`model = read_from_files(path)`  
where the path should contain POSCAR, SPOSCAR and FORCE_CONSTANTS files.   

From the model just built, build a slab model with `multi` layers along `fin_dir`:     
`model_2d = model.cut_piece(multi, fin_dir)`  
`model_2d = model_2d.atom_projected_band(q_path, node_names)`  

Create a Topology object:  
`tp = Topology(model)`  
Plot the energy surfaces of `band1` and `band2` and find the degenerate points on z=0 plane:  
`model.plot_3d_band([band1, band2], center, xy_range)`  
Then plot wannier charge center evolution around `center` for `band_indices`:  
`tp.wcc_evol_sphere(band_indices, center)`  
and the berry curvature distribution on $k_i$=`kz` plane, where $k_i$ = $k_z, k_y, k_z$ if `dirc` = 0, 1, 2, respectively:  
`tp.berry_curvature_proj(band_indices, dirc, kz)`  


#### More examples can be found in runExamples.py file
