# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:08:31 2022

@author: zhuhe
"""


from structure import Structure
from model import read_from_files, Model
from topology import Topology


######### LiCaAs #############
print('LiCaAs!!')
# read files from one line of code
model = read_from_files(r"..\Examples\LiCaAs")
# specify the k path 
k_path = [[0.5,0.5,0.5], [0.5,0.25,0.75], [0.5,0.0,0.5], [0.0,0.0,0.0]]
# plot the band structure of the tight binding model
model.atom_projected_band(k_path, node_names=['L', 'W', 'X', "$\Gamma$"], k_num=80)

# build a Topology object
tp = Topology(model) 
# wannier center evolution of the third band around [0.5,0.222,0.722]
tp.wcc_evol_sphere([2], [0.5,0.222,0.722], r=0.003, dirc=2, num=30,)
# berry curvature distribution
tp.berry_curvature_proj([2],2,0.722,center=[0.5,0.222],xy_range=0.005,num=5)


######### BeAu ##############
print('BeAu!!')
model = read_from_files(r"..\Examples\BeAu")

k_path = [[0.0,0.0,0.0], [0.0,0.5,0.0], [0.5,0.5,0.0],[0.0,0.0,0.0], [0.5,0.5,0.5]]
# plot atom-projected band
model.atom_projected_band(k_path, node_names=["$\Gamma$", 'X', 'M', "$\Gamma$", 'R'],
                          k_num=80,site_comb=[[0,1,2,3],[4,5,6,7]], unit="cm-1")

tp = Topology(model)
## wannier center evolution
# at Gamma point
tp.wcc_evol_sphere([4], [0,0,0], r=0.0001, dirc=2, num=30,)
tp.wcc_evol_sphere([9], [0,0,0], r=0.0001, dirc=2, num=30,)
# at R point, 4-fold degenerate points
tp.wcc_evol_sphere([0,1], [0.5,0.5,0.5], r=0.0001, dirc=2, num=30,)
tp.wcc_evol_sphere([8,9], [0.5,0.5,0.5], r=0.0001, dirc=2, num=30,)

# berry curvature distribution at Gamma
tp.berry_curvature_proj([9],2,0,xy_range=0.04,num=5)
tp.berry_curvature_proj([10],2,0,xy_range=0.01,num=5)
# berry curvature distribution at R
tp.berry_curvature_proj([0,1], 2, 0.5, [0.5,0.5], xy_range=0.01, num=5,)
tp.berry_curvature_proj([8,9], 2, 0.5, [0.5,0.5], xy_range=0.01, num=5,)

## 2d slab model
# build a 2d slab on z plane with 17 layers along z direction 
model_2d = model.cut_piece(17, 2, bottom_shift=0)
# plot the surface band structure
q_path_2d = [[-0.5,0.0], [0.0,-0.5], [0.5,0.0], [0.0,0.0], [0.5,0.5], [0.5,0.0],[0.0,0.5], [-0.5,0.0]]
model_2d.atom_projected_band(q_path_2d, node_names=['X', 'Y', 'X', "$\Gamma$", 'M', 'X', 'Y', 'X'],
                              k_num=70, y_min=100, y_max=126, margin_highlight=[0.45,0], unit="cm-1")
# plot the gaussian smeared surface band structure
model_2d.plot_edge(q_path_2d, [1,0], [100,126], k_num=150, fin_dirc=None, unit="cm-1")


######### graphene ##############
print('Graphene!!')
model = read_from_files(r"..\Examples\graphene")
# make the model 2d!
model_2d = model.cut_piece(1, 2)
node_names=[r"$\Gamma$", "M", "K", r"$\Gamma$", "K'"]
k_path_2d = [[0.0,0.0],[0.5,0.5],[2/3,1/3],[0.0,0.0],[1/3,2/3]]
model_2d.atom_projected_band(k_path_2d, site_comb=[[0],[1]], node_names=node_names, k_num=80)


# plot 3d surface band structures and highlight the degenerate points
model_2d.plot_3d_band([0,1] ,[2/3,1/3], 0.005, view=[-160,50])
model_2d.plot_3d_band([4,5] ,[0.261,0.522], 0.005, tol=5.0, view=[-160,50])

## plot edge modes
model_1d = model_2d.cut_piece(30, 1, bottom_shift=0)
k_path_1d = [[0.0],[0.5],[1.0]]
model_1d.plot_edge(k_path_1d, [0,1], [6, 30], k_num=100, fin_dirc=1, sigma=3)
model_1d.atom_projected_band(k_path_1d, y_min=0, y_max=50, fin_dirc=1, margin_highlight=[0, 1])

# explicitly specify dim=2
tp = Topology(model_2d, dim=2)
wfs1 = tp.gen_circle_wfs([2/3,1/3],r=0.0001)
wfs2 = tp.gen_circle_wfs([0.45,0.45],r=0.01)
wfs3 = tp.gen_circle_wfs([0.26,0.52],r=0.01)
print("berry phase for dirac point1", tp.wilson_loop([0], wfs1,))
print("berry phase for dirac point2", tp.wilson_loop([4], wfs2,))
print("berry phase for dirac point3", tp.wilson_loop([4], wfs3,))


######## SrTiO3 with NAC ########## 
print('SrTiO3!!')
## phonopy setup
import phonopy
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.phonon.band_structure import BandStructure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# create phonopy object
unitcell = r"..\Examples\SrTiO3\POSCAR"
supercell = r"..\Examples\SrTiO3\SPOSCAR"
born = r"..\Examples\SrTiO3\BORN"
force_const = r"..\Examples\SrTiO3\FORCE_CONSTANTS"
ph = phonopy.load(supercell_filename=supercell,
                  born_filename=born,
                  force_constants_filename=force_const,)
# create DynamicalMatrix object
dm = get_dynamical_matrix(ph.force_constants,
                          ph.supercell,
                          ph.primitive,
                          nac_params=ph.nac_params)

## topoPhonon side
# build a Structure object
structure = Structure(3)
structure.read_POSCAR(unitcell)
structure.read_supercell(supercell)
# build a model object with Phonopy DynamicalMatrix object
model = Model(structure, dm=dm)
k_path = [[0.0,0.0,0.0], [0.0,0.5,0.0], [0.5,0.5,0.0], [0.0,0.0,0.0], [0.5,0.5,0.5]]
model.atom_projected_band(k_path, node_names=["$\Gamma$", 'X', 'M', "$\Gamma$", 'R'],
                          site_comb=[[0],[1],[2,3,4]])





