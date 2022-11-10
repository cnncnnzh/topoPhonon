# topoPhonon
topoPhonon package is a python package that allows users to calculate topological properties, by building phonon tight-binding model.
* Build tight-binding models from user's input or phonopy FORCE_CONSTANTS files.  
* Calculate berry phase, berry curvature, wannier charge center evolution from tight-binding model.  
* Build slab/ribbon models for surface/edge.  
* plot 3d band surfaces.   

Basic usage:  
Use one line of code to build tight-binding model:  
`model = read_from_files(path)`  
where the path should contain POSCAR, SPOSCAR and FORCE_CONSTANTS files.   

From the model just built, build a slab model with `multi` layers along `fin_dir`:     
`model_2d = model.cut_piece(multi, fin_dir)`  
`model_2d = model_2d.atom_projected_band(q_path, node_names)`  

Create a Topology object:  
`tp = Topology(model)`  
Then plot wannier charge center evolution around `center` for `band_indices`:  
`tp.wcc_evol_sphere(band_indices, center)`  
and the berry curvature distribution on $k_i$=`kz` plane, where $k_i$ = $k_z, k_y, k_z$ if `dirc` = 0, 1, 2, respectively:  
`tp.berry_curvature_proj(band_indices, dirc, kz)`  
