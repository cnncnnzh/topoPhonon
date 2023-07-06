# -*- coding: utf-8 -*-
"""
This module contains the class that describes the phonon tight binding model
"""

from topophonon.structure import Structure
from topophonon.units import VASP2THZ, unit_dict, HBAR, HBARCGS, KB, MOLE
from topophonon.units import ATOMS, SCATTER_PAR_A, SCATTER_PAR_B, SCATTER_PAR_C
from topophonon.units import masses_dict
from topophonon.utils import _convert_pair_to_str, _cartesian_to_direct, _modify_freq, _set_ylabel, _coth

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import os
import re
import copy
from tqdm import tqdm
import functools
from typing import List, Union, Tuple

class Model(object):
    
    PATTERN = "(-?\d+\.?\d+)\s*(-?\d+\.?\d+)\s*(-?\d+\.?\d+).*"
    TOL = 10**(-5)
    
    def __init__(self,
                 structure: Structure,
                 dm=None):        
        """
        Parameters
        ----------
        structure: Structure
            a Structure object
        dm: phonopy.harmonic.dynamical_matrix.DynamicalMatrix, optional
            the DynamicalMatrix object from phonopy API
            The default is None
        """
        
        self.structure = structure 
        self.VASPcal = False
        if dm is not None:
            # interface for Phonopy package
            assert dm is not None, "Phonopy object must be specified"
            self.ph_dm = dm
            self.dim = 3
            self.VASPcal = True
            self._read_from_phonopy = True
        else:
            # Start from reading files
            self.dim = self.structure.dim
            self.fc = []    
            #initialize the container for unique pairs of atoms
            self.unique_pair = set()   
            self._read_from_phonopy = False
   
        # initialize the finite direction
        self.fin_dirc = []
        self.bottom_del = set()
        self.top_del = set()
    

    def set_fc(self,
               atom_1: int,
               atom_2: int,
               lattice_vec: Union[List[float], np.ndarray],
               lattice_vec_int: Union[List[int], np.ndarray],
               fc: Union[List[complex], np.ndarray],):
        """
        set force constants manually
        
        Parameters
        ----------
        atom_1 : int
            the index of the 1st atom in the original unit cell
        atom_2 : int
            the index of the 2st atom in the original unit cell
        lattice_vec : list or ndarray of float
            the vector from atom_1 to atom_2 in direct coordinate.
        lattice_vec_int: list of int
            the vector that connect the unit cell of atom_1 and the unit cell of 
            atom_2 in direct coordinate. The elements must be integers. 
        fc: list or ndarray of complex
            the "dim" * "dim" force constants matrix 
        """
        
        assert isinstance(fc, (list, np.ndarray)),\
            "the force constants must be given in a list or array"
        assert len(fc) == self.structure.dim,\
            "the dimension of the system and the force constant must match"
        assert len(lattice_vec) == self.structure.dim,\
            "the dimension of the system and lattice_vec must match"
        assert len(lattice_vec_int) == self.structure.dim,\
            "the dimension of the system and lattice_vec_int must match"
        #the elements in lattic_Vec must be intgers
        
        str_pair = _convert_pair_to_str(atom_1,
                                        atom_2,
                                        np.array(lattice_vec_int, dtype=int))
        if str_pair not in self.unique_pair:
            self.fc.append([atom_1, 
                            atom_2, 
                            np.array(lattice_vec, dtype=float), 
                            np.array(lattice_vec_int, dtype=float),
                            np.array(fc)])
            self.unique_pair.add(str_pair)
    

    def _atom_index(self,
                    cart: np.ndarray = None,
                    atom: Union[int, None] = None
                    ) -> (int, np.ndarray):
        """
        find the index in the primitive cell of
        an atom in the supercell, as well as its lattice translation
        relative to the primitive cell 
        """
        
        for i, org_cart in enumerate(self.structure.prm_cart):
            if isinstance(cart, (list, np.ndarray)):
                diff = cart - org_cart
            elif atom is not None:
                diff = self.structure.super_cart[atom] - org_cart
            else:
                raise Exception("Failed to calculate an atomic index")
            lattice_disp = np.dot(diff, np.linalg.inv(self.structure.lat))
            rounded_lat = np.array([round(x) for x in lattice_disp],dtype=int)
            if abs(np.linalg.norm(lattice_disp-rounded_lat)) < Model.TOL:
                return i, rounded_lat
        raise Exception("""An error occurs when reading POSCAR/SPOSCAR. A possible
                        reason is that your lattice tensors in POSCAR and SPOSCAR 
                        are not related by a diagonal matrix""")

    
    def _pair_to_prm_vec(self,
                         atom_1: int,
                         atom_2: int
                         ) -> (int, int, np.ndarray): 
        """
        express the vector of a pair in the unit of primitive lattice.
        the atoms are specified by their indexes in the supercell.
    
        """
        i_1, lat_disp_1 = self._atom_index(atom=atom_1)
        i_2, lat_disp_2 = self._atom_index(atom=atom_2)
        #move atom_2 back to the primitive cell
        dist_rel_to_prm = self.structure.super_cart[atom_1]\
            - np.dot(lat_disp_2, self.structure.lat)
        _, lat_disp = self._atom_index(cart=dist_rel_to_prm)
        return i_1, i_2, lat_disp
        
    
    def _coord_to_prm_vec(self,
                          cart_1: np.ndarray,
                          cart_2: np.ndarray) -> (int, int, np.ndarray):
        """
        express the vector of a pair in the unit of primitive lattice.
        the atoms are specified by their cartesian coordinate.
    
        """
        i_1, lat_disp_1 = self._atom_index(cart=cart_1)
        i_2, lat_disp_2 = self._atom_index(cart=cart_2)
        #move atom_2 back to the primitive cell
        dist_rel_to_prm = cart_1 - np.dot(lat_disp_2, self.structure.lat)
        _, lat_disp = self._atom_index(cart=dist_rel_to_prm)
        return i_1, i_2, lat_disp
    
    
    def _shortest_disp(self,
                       index_super_1: int,
                       index_super_2: int
                       ) -> (np.ndarray, np.ndarray, int):
        """
        find the nearest lattice displacement between a pair of atoms in the supercell,
        also find the multiplication. In this gauge the lattice displacements
        are not integers

        """     
        cart_1 = self.structure.super_cart[index_super_1]
        cart_2 = self.structure.super_cart[index_super_2]
        min_dist = float('inf')
        min_vec = []
        min_vec_int = []

        for i in [1,0,-1]:
            for j in [1,0,-1]:
                for k in [1,0,-1]:
                    cart_translation = np.dot(np.array([i,j,k]),
                                              self.structure.super_lat)
                    cart_2_trans = cart_2 + cart_translation 
                    diff = cart_2_trans - cart_1
                    dist = np.linalg.norm(diff)
                    lat_disp = np.dot(diff, np.linalg.inv(self.structure.lat))
                    i_1, i_2, lat_disp_int = self._coord_to_prm_vec(cart_1,
                                                                cart_2_trans)
                    #if the distance equals to the minimum, store it 
                    if abs(min_dist - dist) < Model.TOL:
                        min_vec.append(lat_disp)
                        min_vec_int.append(lat_disp_int)
                    #if the distance is smaller than the minimum, build a new container
                    elif dist < min_dist:
                        min_dist = dist
                        min_vec = [lat_disp]
                        min_vec_int = [lat_disp_int]
        multi = len(min_vec)
        return np.array(min_vec), np.array(min_vec_int,dtype=int), multi
    
    
    def read_fc(self,
                force_const: str
                ):
        """
        read force constant from FORCE_CONSTANT file
    
        Parameters
        ----------
        force_const : str
            the path of the FORCE_CONSTANTS file
        """    
        # in the iteration, find which unit cell an atom belongs to, as well
        # as its relative coord
        self.VASPcal = True
        
        #start reading force constants
        
        print("start reading force constants...")
        assert os.path.exists(force_const), "{} not found".format(force_const)
        with open(force_const) as f:
            num_super = int(f.readline().strip().split()[0])
            assert num_super == len(self.structure.super_dirc),\
                "number of atoms in {} and supercell don't match"\
                    .format(force_const)
            count = num_super * num_super
            next_line = f.readline()
            # self.multi = [[0 for _ in range(len(self.structure.prm_dirc))] for _ in range(len(self.structure.prm_dirc))]
            # self.vecs = []
            visited = set([])
            #make a progress bar
            with tqdm(total=count) as pbar:
                while len(next_line) > 1:
                    pbar.update(1)
                    find_pair = re.findall("(\d+)\s*(\d+)\s*", next_line)[0]
                    atom_1, atom_2 = int(find_pair[0])-1, int(find_pair[1])-1 
                    
                    index_prm_1, index_prm_2, lat = self._pair_to_prm_vec(atom_1,
                                                                          atom_2) 
                    #avoid duplicates
                    str_pair = _convert_pair_to_str(index_prm_1, index_prm_2, lat)
                    if str_pair in visited:
                        for _ in range(4):
                            next_line = f.readline()
                        count += 1
                        continue
                    visited.add(str_pair)
                    
                    vecs, vecs_int, multi = self._shortest_disp(atom_1, atom_2)
                    # self.multi[atom_1][index_prm_2] = multi
                    # self.vecs.append(vecs)
                    force_list = []
                    for _ in range(3):
                        force_str = re.findall(Model.PATTERN, f.readline())[0]
                        force_list.append([float(r) for r in force_str])
                    force_array = np.array(force_list) 
                    
                    for i in range(multi):
                        self.set_fc(index_prm_1,
                                    index_prm_2,
                                    vecs[i],
                                    vecs_int[i],
                                    force_array/multi)
                    
                    next_line = f.readline()
                print("successully read force constants")
       
        
    @functools.lru_cache()
    def _assign_new_index(self,
                          index: int
                          ) -> int:
        """
        after removing the bottom/top atoms, assign new indexes
        """
        r = index
        for i in self.bottom_del:
            if i < index:
                r -= 1       
        for i in self.top_del:
            if i < index:
                r -= 1       
        return r
    
    
    def _make_force_const_matrix(self,
                                 k: np.ndarray,
                                 k_direction: Union[np.ndarray, None] = None
                                 ) -> np.ndarray:
        """
        convert the input force constants to force constant matrix;
        K-point should be given in reduced coordinates
        """

        if self._read_from_phonopy:
            if k_direction is None:
                self.ph_dm.run(np.array(k))
            else:
                self.ph_dm.run(np.array(k), k_direction)
            dy_mt = self.ph_dm.get_dynamical_matrix()
        else:
            #the size of the dynamical matrix is the (# of atoms in a unit cell) * (dim)
            dim = self.dim
            num_del = len(self.bottom_del) + len(self.top_del)
            num_atom = len(self.structure.masses) - num_del
            dy_mt = np.zeros((num_atom*dim,
                              num_atom*dim), 
                              dtype=complex)
            
            for index_I, index_J, lattice_vec, vec_int, fc in self.fc:
                #find the positions in the matrix after cutting edge/surface atoms
                index_I = self._assign_new_index(index_I)  
                index_J = self._assign_new_index(index_J)
                # consider the directions that are periodic
                # per = [i for i in range(dim) if i not in self.fin_dirc]
                phase_factor = np.exp(2j * np.pi * np.vdot(k, vec_int))
                # phase_factor = np.exp(2j * np.pi * np.vdot(k, lattice_vec))
                row, col = (index_I)*dim, (index_J)*dim
                dy_mt[row:row+dim, col:col+dim] += phase_factor * fc
            # make dynamical matrix hermitian
            dy_mt = (dy_mt + dy_mt.conj().transpose()) / 2           
        return dy_mt
        
    
    def _make_dynamical_matrix(self,
                               k: np.ndarray,
                               k_direction: Union[np.ndarray, None] = None
                               ) -> np.ndarray:
        """
        convert the input force constants to dynamical matrix;
        K-point should be given in reduced coordinates
        """

        if self._read_from_phonopy:
            if k_direction is None: 
                self.ph_dm.run(np.array(k))
            else:
                self.ph_dm.run(np.array(k),
                               q_direction=k_direction)
            dy_mt = self.ph_dm.get_dynamical_matrix()
        else:
            #the size of the dynamical matrix is the (# of atoms in a unit cell) * (dim)
            dim = self.dim
            num_del = len(self.bottom_del) + len(self.top_del)
            num_atom = len(self.structure.masses) - num_del
            dy_mt = np.zeros((num_atom*dim,
                              num_atom*dim), 
                              dtype=complex)
            
            for index_I, index_J, lattice_vec, vec_int, fc in self.fc:
                mass = np.sqrt(self.structure.masses[index_I] *\
                               self.structure.masses[index_J])
                #find the positions in the matrix after cutting edge/surface atoms
                index_I = self._assign_new_index(index_I)  
                index_J = self._assign_new_index(index_J)
                # phase_factor = np.exp(2j * np.pi * np.vdot(k, vec_int))
                phase_factor = np.exp(2j * np.pi * np.vdot(k, lattice_vec))
                # print(phase_factor)
                row, col = (index_I)*dim, (index_J)*dim
                dy_mt[row:row+dim, col:col+dim] += phase_factor * fc / mass 
            # make dynamical matrix hermitian
            dy_mt = (dy_mt + dy_mt.conj().transpose()) / 2           
        return dy_mt

 
    def _make_k_path(self,
                     k_path: Union[List, np.ndarray],
                     k_num: int = 150
                     ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Interpolates a path in reciprocal space between specified
        nodes.
        """ 
        assert isinstance(k_path, (list, np.ndarray)),\
            "the node coordinates must be given in a list or array"
            
        # number of nodes
        n_nodes = len(k_path)
        
        k_path_temp = []
        # set the non-periodic direction to be zero
        for node in k_path:
            pointer = 0
            temp_k = []
            for j in range(self.dim):
                if j in self.fin_dirc:
                    temp_k.append(0)
                else:
                    temp_k.append(node[pointer])
                    pointer += 1
            k_path_temp.append(np.array(temp_k))
        k_path = np.array(k_path_temp)
        
        # find the length between two nodes
        node_dist = [0]
        for i in range(1, n_nodes):
            dk = k_path[i] - k_path[i-1]
            # print(np.dot(np.dot(self.structure.k_lat,dk), np.dot(self.structure.k_lat, dk)))
            dk_dist = np.sqrt(np.dot(np.dot(self.structure.k_lat,dk),
                                     np.dot(self.structure.k_lat,dk)))
            node_dist.append(dk_dist + node_dist[i-1])
        node_dist = np.array(node_dist)
        
        #assign points to the path
        num_list = []
        for i in range(1, n_nodes):
            num = (node_dist[i] - node_dist[i-1]) / node_dist[-1] * (k_num - 1)
            num_list.append(round(num))
        num_list = np.array(num_list, dtype=int)
        
        # for each interpolated point, find q-point and distance
        k_points = []
        k_dist = []
        for i in range(1, n_nodes):
            dk = k_path[i] - k_path[i-1]
            dist = node_dist[i] - node_dist[i-1]
            num = num_list[i-1]
            n = 0
            while n < num:
                frac = n / num
                k_points.append(k_path[i-1] + dk * frac)
                k_dist.append(node_dist[i-1] + dist * frac)
                n += 1
                
        k_points.append(k_path[-1])
        k_dist.append(node_dist[-1])   
        k_points = np.array(k_points)
        k_dist = np.array(k_dist)
                
        return k_points, k_dist, node_dist    
    
    
    def solve_dynamical_matrix_kpath(self,
                                     k_path: List[List[float]],
                                     k_direction : Union[np.ndarray, None] = None,
                                     k_num: int = 150,
                                     unit: float = 1,
                                     eig_vec: bool = True
                                     ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        convert the input force constants to dynamical matrix on the given k path
        
        Parameters
        ----------
        k_path : list of list
            lists of coordinates of k-point nodes; k-point will be interpolated 
            between two nearby nodes.
        k_direction : ndarray, optional
            the k direction for non-analytical correction calculation; if not 
            specified, the direction is defined automatically by k_path.
        k_num : int, optional
            total number of kpoints interpolated between two nodes
            The default is 150
        convert_unit : boolean, optional
            determine whether convert the unit from VASP to THz.
            The default is True
        eig_vec : boolean, optional
            determine whether eigenvectors are 
        
        Returns
        -------
        the dynamical matrix at each point;
        frequencies of all bands at each k
        eigenvectors of all bands at each k
        
        """

        k_points, _, _ = self._make_k_path(k_path, k_num)
        all_freqs, all_eig_vecs = [], []
        for i, k in enumerate(k_points):
            if self._read_from_phonopy and self.ph_dm.is_nac()\
                and k_direction is None and k_num > 1:
                # setup the direction for nac term
                k_direction = k_points[i+1] - k_points[i]  
            dy_mt = self._make_dynamical_matrix(k, k_direction)
            if np.max(dy_mt - dy_mt.T.conj()) > 1.0E-9:
                raise Exception("dynamical matrix is not hermitian")
            #diagonalize the dynamical matrix
            if eig_vec is True:
                eig_vals, eig_vec = np.linalg.eigh(dy_mt)
                freqs = np.zeros(len(eig_vals), dtype="complex64")
                for i, eig_val in enumerate(eig_vals): 
                    # freqs[i] = csqrt(eig_val)
                    if eig_val >= 0:    
                        freqs[i] = np.sqrt(eig_val)
                    else:
                        freqs[i] = -np.sqrt(-eig_val)   
                all_freqs.append(freqs*unit)
                all_eig_vecs.append(eig_vec) 
                return dy_mt, all_freqs, all_eig_vecs  
            else:
                eig_vals = np.linalg.eigvalsh(dy_mt)
                freqs = np.zeros(len(eig_vals), dtype="complex64")
                for i, eig_val in enumerate(eig_vals): 
                    # freqs[i] = csqrt(eig_val)
                    if eig_val >= 0:    
                        freqs[i] = np.sqrt(eig_val)
                    else:
                        freqs[i] = -np.sqrt(-eig_val)  
                all_freqs.append(freqs*unit)
                return dy_mt, all_freqs
    

    def _get_weight(self,
                    vec: np.ndarray,
                    site_comb: List[List[int]]
                    ) -> np.ndarray:
        """
        compute the weight for each combintaion of sites according to the
        eigenvector 
        """
        num_atom = len(self.structure.masses)
        new_vec = np.zeros(num_atom)
        for i in range(num_atom):
            new_vec[i] = np.linalg.norm(vec[i*self.dim:i*self.dim+self.dim])
        
        #get the projectors for each group
        gw = []
        norm_f = 0
        for i, comb in enumerate(site_comb):
            projector = np.zeros(len(new_vec))
            for j in range(len(projector)):
                if j in comb:
                    projector[j] = 1
            group_weight = np.dot(projector, new_vec)
            gw.append(group_weight)
            norm_f += group_weight
        return np.array(gw, dtype=float) / norm_f
    

    def _make_color(self,
                    colors: List[float],
                    margin_highlight: List[float]
                    ) -> Tuple[float]:
        """
        convert the eigendisplacements to rgb colors 

        """
        if len(colors) == 2:
            return tuple([colors[0],0,colors[1]])
        # if there are three groups, use red and green and blue
        elif len(colors) == 3:
            #exaggerate the contribution from the edge
            if margin_highlight != [0.,0.]:
                factor = 2.5
                diff_1 = min(colors[0]*factor, 1.0) - colors[0]
                diff_2 = min(colors[1]*factor, 1.0) - colors[1]
                return tuple([min(colors[0]*factor, 1.0), 
                              min(colors[1]*factor, 1.0),
                              max(colors[2]-diff_1-diff_2, 0.0)])
            else:
                return tuple(colors)
        # if there are four groups, use cyan, magenta, yellow and black
        elif len(colors) == 4:
            r = (1-colors[0])*(1-colors[3]) 
            g = (1-colors[1])*(1-colors[3]) 
            b = (1-colors[2])*(1-colors[3])
            return tuple([r,g,b])
        
        
    def _make_title(self,
                    ax: matplotlib.axes.Axes):
        """
        make the title for the plot
        """
        if self.structure.atoms is None:
            return
        atom_count = {}
        for atom in self.structure.atoms:
            if atom not in atom_count:
                atom_count[atom] = 1
            else:
                atom_count[atom] += 1
        compound = ""
        if len(atom_count) == 1:
            ax.set_title(self.structure.atoms[0], fontsize=14)
        else:
            for atom, count in atom_count.items():
                if count == 1:
                    compound += atom
                else:
                    compound += "{}$_{}$".format(atom, str(count))
            ax.set_title(compound, fontsize=14)

    
    def _make_legend(self,
                     ax: matplotlib.axes.Axes,
                     site_comb : List[List[float]],
                     margin_highlight: List[float]):
        """
        make the legend for the plot
        """
        from matplotlib.lines import Line2D

        if len(site_comb) == 2:
            lines = [Line2D([0], [0], color='red', lw=2.5),
                Line2D([0], [0], color='blue', lw=2.5)]
            names = ['comb1', 'comb2']
        elif len(site_comb) == 3:
            lines = [Line2D([0], [0], color='red', lw=2.5),
                Line2D([0], [0], color='green', lw=2.5),
                Line2D([0], [0], color='blue', lw=2.5),]
            if margin_highlight != [0., 0.]:
                names = ['bottom', 'top', 'bulk']  
            else:
                names = ['comb1', 'comb2', 'comb3']        
        else:
            lines = [Line2D([0], [0], color='cyan', lw=2.5),
                Line2D([0], [0], color='magenta', lw=2.5),
                Line2D([0], [0], color='yellow', lw=2.5),
                Line2D([0], [0], color='black', lw=2.5),]
            names = ['comb1', 'comb2', 'comb3', 'comb4']
        ax.legend(lines, names)


    def _convert_unit(self,
                      unit: Union[str, int]
                      ) -> float:
        """
        Convert the unit if using VASP interface

        """
        
        if isinstance(unit, str) and unit.lower() in unit_dict:
            # if given a string, convert only if using vasp outputs
            unit_vasp = VASP2THZ * unit_dict[unit.lower()]
            return unit_vasp if self.VASPcal else 1
        elif isinstance(unit, int):
            return unit
        else:
            raise Exception("invalid unit specified")
    
    



    def atom_projected_band(self,
                            nodes: List[List[float]],
                            site_comb: Union[List[List[float]], None] = None,
                            node_names: Union[List[str], None] = None,
                            k_num: int = 150,
                            y_min: int = None,
                            y_max: int = None,
                            margin_highlight: List[float] = [0.,0.],
                            fin_dirc: Union[int, None] = None,
                            unit: Union[str, float] = "THz",
                            qi = None,
                            qi_band = 'all',
                            T=0,
                            int_factor=1,
                            max_size=float('inf')
                            ) -> matplotlib.figure.Figure:
        """
        Make plain or atom-resolved phonon band plot
        
        Parameters
        ----------
        nodes: 2d list or ndarray of float
            a list of kpoints that defines the kpath for band plot
        site_comb : 2d list, optional
            if specified, the plot will be colored based on the magnitude of 
            displacement in each combination; for example, in SrTiO3
            [[1],[2],[3,4,5]] will project the plot onto Sr, Ti and O;
            the number groups can be 2, 3 and 4;
            if not specified, the plain band will be plotted
            The default is None
        node_names : list, optional
            a list of strings of kpoint names, e.g. [r"$\Gamma$", "X", "M", r"$\Gamma$", "R",]
            The default is None
        k_num : int, optional
            total number of kpoints in the plot
            The default is 150
        y_min : int, optional
            lower bound of the plot.
            The default is None
        y_max : int, optional
            upper bound of the plot.
            The default is None
        margin_highlight : list, [float, float], optional
            bands contributed from atoms margin_highlight[0]/[1] closer to the bottom/top will 
            be colored. The value should be given in the unit of lattice vector
            of the primitive cell. For example, edge_highlight=[1,1] will highlight 
            the contribution from the bottom primitive cell and the top primitive cell.
            red: bottom, green: top, blue: bulk 
            The default is [0.0,0.0]
        fin_dirc : int, optional
            the direction of the edge/surface. If not specified, the first value in
            model.fin_dirc will be used
        unit : str or float, optional
            the unit of the plot; can be "thz", "cm-1", "cm^-1", "ev", "mev". 
            The default is "thz"
        Returns
        -------
        A matplotlib.figure.Figure object containing phonon band plot
        """
        
        print("start plotting the atom_resolved band structures...")
        from matplotlib.collections import LineCollection
        # import matplotlib.ticker as ticker
        import matplotlib as mpl
        
        # convert unit 
        unit_num = self._convert_unit(unit)
        
        if node_names is not None:
            assert isinstance(node_names, (list, np.ndarray)),\
                "the names of nodes must be given in a list or array"
            assert len(nodes) == len(node_names),\
                "the lengths of nodes and k_name don't match"
            self.node_names = node_names
        
        #convert edge_highlight to site_comb
        if margin_highlight != [0.,0.]:
            site_comb = [[],[],[]]
            assert len(self.fin_dirc) != 0, "You don't have a slab/ribbon, please \
                call cut_piece function first"
            if fin_dirc is None:
                fin_dirc = self.fin_dirc[0]
            else:
                assert fin_dirc in self.fin_dirc, "Wrong fin_dirc"
            bot_bound = margin_highlight[0] / self.multi
            top_bound = margin_highlight[1] / self.multi
            bot_coord = min(self.structure.prm_dirc[:,fin_dirc])
            top_coord = max(self.structure.prm_dirc[:,fin_dirc])
            names = self.structure.atoms
            for i, coord in enumerate(self.structure.prm_dirc):
                if coord[fin_dirc] - bot_coord < bot_bound:
                    print("will project wfs of {} to bottom state".format(names[i]))
                    site_comb[0].append(i)
                elif top_coord - coord[fin_dirc] < top_bound:
                    print("will project wfs of {} to top state".format(names[i]))
                    site_comb[1].append(i)
                else:
                    site_comb[2].append(i)
        
        k_points, k_dist, node_dist = self._make_k_path(nodes, k_num=k_num)
        num_k = len(k_dist)
        num_band = ((len(self.structure.masses)-len(self.bottom_del)-len(self.top_del)))\
                    * self.dim
        
        # find all the intensities and convert them to rgb
        if qi is not None:
            DW_matrices = self.DW_coefficient(T)
            all_intensities = np.zeros((len(k_points), num_band))
            max_i = 0
            min_i = float('inf')
            for i in range(len(k_points)):
                for j in range(num_band):
                    cur_i = self.intensity(qi, k_points[i], DW_matrices, T=T, branches=[j])[0]
                    # print(cur_i)
                    max_i = max(max_i, cur_i)
                    min_i = min(min_i, cur_i)
                    all_intensities[i][j] = cur_i
            max_i = min(max_size, max_i)
                # print(all_ki)
        # print(max_i)
        
        # band plot
        fig, ax = plt.subplots()
        #starting point
        k_direction = None
        if self._read_from_phonopy and self.ph_dm.is_nac():
            k_direction = k_points[1] - k_points[0]
        # dy_mt = self._make_force_const_matrix(k_points[0], k_direction)
        dy_mt = self._make_dynamical_matrix(k_points[0], k_direction)
        eig_vals_2, eig_vecs_2 = np.linalg.eigh(dy_mt)
        _modify_freq(eig_vals_2)
        y_max_g, y_min_g = max(eig_vals_2), min(eig_vals_2)
        x_min, x_max = k_dist[0], k_dist[-1]
        for i in range(1,len(k_points)):
            if site_comb is not None:
                colors = np.zeros((num_k, num_band, len(site_comb)))
            frequencies = np.zeros((num_band,2))
            # the previous eigenvals and eigenvectors 
            eig_vals_1, eig_vecs_1 = eig_vals_2, eig_vecs_2
            k_direction = None
            if self._read_from_phonopy and self.ph_dm.is_nac():
                k_direction = k_points[i] - k_points[i-1]
            # dy_mt = self._make_force_const_matrix(k_points[i], k_direction)
            dy_mt = self._make_dynamical_matrix(k_points[i], k_direction)
            eig_vals_2, eig_vecs_2 = np.linalg.eigh(dy_mt)
            # set correct frequencies
            _modify_freq(eig_vals_2)
            y_max_g = max(y_max_g, max(eig_vals_2))
            y_min_g = min(y_min_g, min(eig_vals_2))
            frequencies[:,0] = eig_vals_1*unit_num
            frequencies[:,1] = eig_vals_2*unit_num
            
            seg = np.zeros((num_band, 2, 2))
            seg[:, :, 1] = frequencies
            seg[:, 0, 0] = k_dist[i-1]
            seg[:, 1, 0] = k_dist[i]
            
            if qi is not None:
                intensities_at_k = []
                for j in range(num_band):
                    intensity_1 = min((all_intensities[i-1][j]) * int_factor, max_size)
                    intensity_2 = min((all_intensities[i][j]) * int_factor, max_size)
                    # intensities_at_k.append( np.log((intensity_1+intensity_2) / 2) )    
                    intensities_at_k.append( (intensity_1+intensity_2) / 2 )    
                ls = LineCollection(
                    seg,
                    array=np.array(intensities_at_k),
                    linestyles='-',
                    linewidths=2.0,
                    clim=(min_i, max_i),
                    cmap='viridis'
                    )
                ax.add_collection(ls)
                # axcb = fig.colorbar(ls,location='right')
                
            elif site_comb is not None:
                colors = []
                for j in range(num_band):
                    # set the weights of each atom groups
                    colors1 = self._get_weight(eig_vecs_1[:,j], site_comb)
                    colors2 = self._get_weight(eig_vecs_2[:,j], site_comb)
                    colors.append(self._make_color((colors1 + colors2)/2, margin_highlight))
                ls = LineCollection(
                    seg,
                    colors=colors,
                    linestyles='-',
                    linewidths=2.0,
                    )
                self._make_legend(ax, site_comb, margin_highlight)
                ax.add_collection(ls) 

            else:
                ls = LineCollection(seg, linestyles='-', linewidths=2.0)
                ax.add_collection(ls) 

        # set ylim and xlim
        margin = (y_max_g - y_min_g) * 0.05
        if y_max is None:
            y_max = (y_max_g + margin) * unit_num
        if y_min is None:
            y_min = (y_min_g - margin) * unit_num
        y_range = y_max - y_min
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # set ylabel and xlabel
        ax.set_ylabel(_set_ylabel(unit), fontsize=16)
        ax.set_xlabel("Wavevector", fontsize=16)
        
        #show high symmetry point
        ax.vlines(node_dist, y_min-y_range*0.05, y_max+y_range*0.05)
        #set node labels
        if node_names is not None:
            ax.set_xticks(node_dist)
            ax.set_xticklabels(node_names)
        ax.tick_params(labelsize=15)
        if len(self.fin_dirc) == 0:
            self._make_title(ax)
        
        # make a color bar
        if qi is not None:
            cmap = mpl.cm.viridis
            norm = mpl.colors.Normalize(
                vmin=min_i*int_factor,
                vmax=min(max_i,max_size)*int_factor
                )
            # norm = mpl.colors.Normalize(vmin=min_i*int_factor, vmax=max_i*int_factor)
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                orientation='vertical',
                label='Intensity',
                )
        # cbar.set_ticks([i for i in range(int(min_i), int(max_i), int((max_i-min_i)//5))])
        return fig

    
    def _make_supercell(self,
                        multi: int,
                        fin_dirc: int
                        ) -> Structure:
        """
        Build a supercell for edge/surface states calculation
 
        """
        
        old_strc = self.structure
        num = len(old_strc.masses)

        #initialization        
        structure = Structure(old_strc.dim)
        structure.lat = np.zeros((old_strc.dim,old_strc.dim))
        structure.masses = np.zeros(multi*num)  
        structure.prm_dirc = np.zeros((multi*num, old_strc.dim))
        structure.prm_cart = np.zeros((multi*num, old_strc.dim))
        if old_strc.atoms is not None:
            structure.atoms = [0 for _ in range(multi*num)]
        
        new_lat = copy.deepcopy(old_strc.lat)
        new_lat[fin_dirc] *= multi 
        structure.set_lat(new_lat)
        # structure.k_lat = np.delete(structure.k_lat, fin_dirc, axis=0)
        
        # calculate the coordinates of all atoms in the supercell
        for i in range(multi):
            for j in range(num):
                index = i*num + j
                if old_strc.atoms is not None:
                    structure.atoms[index] = old_strc.atoms[j]
                structure.masses[index] = old_strc.masses[j]
                cart = old_strc.prm_cart[j] + i*old_strc.lat[fin_dirc]
                direct = _cartesian_to_direct(cart, structure.lat)
                structure.prm_dirc[index] = direct
                structure.prm_cart[index] = cart
        return structure
    
    
    def cut_piece(self,
                  multi: int,
                  fin_dirc: int,
                  bottom_shift: float = 0.0,
                  top_shift: float = 0.0
                  ) -> 'Model':
        """
        Constructs a (dim-1)-dimensional tight-binding model. This is 
        a phonon version of cut_piece function of pythTB. The extra unit cells
        are stacked on the top of the original cell.
               
        Parameters
        ----------
        multi : int
            number of times the unit cell is multiplied (i.e., thickness of
            slab or ribbon).
        fin_dirc : int 
            direction along which the model is no longer periodic (i.e., the norm 
            of surface/edge).
        bottom_shift : float
            atoms between (the bottom-most atom) and (the bottom-most atom plus bottom_shift) will be deleted;
            unit: lattice parameters in the primitive cell
            used to change the configuration of the bottom edge/surface
        top_shift : float   
            atoms between (the top-most atom) and (the top-most atom minus bottom_shift) will be deleted;
            used to change the configuration of the top edge/surface
        Returns
        -------
        a Model object with force constants of the layered structure 
        """

        num = len(self.structure.masses) #number of atoms in the primitive cell
        structure = self._make_supercell(multi, fin_dirc)
        model = Model(structure) 
        model.org_model = self
        model.VASPcal = self.VASPcal
        model.multi = multi
        model.fin_dirc = self.fin_dirc[:] 
        model.fin_dirc.append(fin_dirc)
        
        # delete atoms on the bottom and the top
        bot_bound = bottom_shift / multi
        top_bound = top_shift / multi
        bot_coord = min(model.structure.prm_dirc[:,fin_dirc])
        top_coord = max(model.structure.prm_dirc[:,fin_dirc])
        for i, coord in enumerate(model.structure.prm_dirc):
            if coord[fin_dirc] - bot_coord < bot_bound:
                model.bottom_del.add(i)
                print("{} at the bottom is removed".format(model.structure.atoms[i]))
            elif top_coord - coord[fin_dirc] < top_bound:
                model.top_del.add(i)
                print("{} at the top is removed".format(model.structure.atoms[i]))

        #modify the force constants in the original model
        #for each repeated cell 
        for c in range(multi):
            #for each force constant in the original model 
            for atom_1, atom_2, vec, vec_int, fc in self.fc:
                #assign new indexes
                atom_1 = atom_1 + c*num
                atom_2 = atom_2 + (c + vec_int[fin_dirc])*num
                if atom_2 < 0 or atom_2 > num * multi - 1:
                    continue
                # remove bottom and top atoms
                if atom_1 in model.bottom_del or atom_1 in model.top_del or\
                atom_2 in model.bottom_del or atom_2 in model.top_del:
                    continue
                
                # the vector in the given directon is not considered any more
                new_vec = copy.deepcopy(vec)
                new_vec[fin_dirc] = 0.0
                
                new_vec_int = copy.deepcopy(vec_int)
                new_vec_int[fin_dirc] = 0.0
                
                model.set_fc(int(atom_1), int(atom_2), new_vec, new_vec_int, fc)
        
        return model
        #change the indexes and the lattice displacements in force constants
    
    
    def _pixel_band(self,
                    k: np.ndarray,
                    edge_atoms: np.ndarray,
                    ylim: np.ndarray,
                    y_res: int,
                    sigma: float,
                    unit: float
                    ) -> np.ndarray:
        """
        make a series of grids at k whose colors depends on edge_atoms
        """
        nb_atoms = len(self.structure.masses)
        y_grid = np.zeros(y_res)
        dy_mt = self._make_dynamical_matrix(k)
        eig_vals, eig_vecs = np.linalg.eigh(dy_mt)
        _modify_freq(eig_vals)
        for i, freq in enumerate(eig_vals):
            # the index of the square for a given freqency
            freq *= unit
            if freq < ylim[0] or freq > ylim[1]:
                continue   
            # convert the eigenvector of length dim*n to a vector of length n
            eig_vec = eig_vecs[:, i]
            new_vec = np.zeros(nb_atoms)
            for i in range(nb_atoms):
                new_vec[i] = np.linalg.norm(eig_vec[i*self.dim:i*self.dim+self.dim])
                
            index = (freq-ylim[0]) // ((ylim[1]-ylim[0])/y_res)
            y_grid[int(index)] += np.dot(new_vec, edge_atoms)
            
        # # gaussian smearing
        y_grid = gaussian_filter1d(y_grid, sigma)
        return y_grid


    def plot_edge(self,
                  nodes: List[List[float]],
                  edge: List[float],
                  y_min: float,
                  y_max: float,
                  node_names: Union[List[str], None] = None,
                  k_num: int = 100,
                  fin_dirc: Union[int, None] = None,
                  sigma: float = 2.0,
                  unit: Union[str, float] = "thz"
                  ) -> matplotlib.figure.Figure:
        """
        plot edge/surface states, smear the bulk bands with gaussian
        
        Parameters
        ----------
        nodes: 2d list or ndarray
            a list of kpoints that defines the kpath for band plot
        edge : list, [float, float]
            bands contributed from atoms margin_highlight[0]/[1] closer to the bottom/top will 
            be colored. The value should be given in the unit of lattice vector
            of the primitive cell. For example, edge_highlight=[1,1] will highlight 
            the contribution from the bottom primitive cell and the top primitive cell.
        y_min : int
            lower bound of the plot.
        y_max : int
            upper bound of the plot.
        k_num : int, optional
            total number of kpoints in the plot
            The default is 100
        fin_dirc : int, optional
            the edges/surfaces normal to the "fin_dirc"th direction will be considered,
            if not specified, the fin_dirc will be the automatically chosen as the first
            element in self.fin_dirc
            The default is None
        sigma : float, optional
            standard deviation for Gaussian kernel
            The default is 2
        unit : str or float, optional
            the unit of the plot; can be "thz", "cm-1", "cm^-1", "ev", "mev". 
            The default is "thz"
        Returns
        -------
        A matplotlib.figure.Figure object containing phonon band plot
        """
        
        ylim = np.array([y_min, y_max])        
        # convert unit 
        unit_num = self._convert_unit(unit)
        # get ylabel
        ylabel = _set_ylabel(unit)
        if self.dim == 2:
            print("start plotting the edge states...")
        elif self.dim == 3:
            print("start plotting the surface states...")

        # determine which atoms are considered as edge atoms
        nb_atoms = len(self.structure.masses)
        edge_atoms = np.zeros(nb_atoms)
        assert len(self.fin_dirc) > 0, "plot_boundary method works for edge/surface states"
        if edge != [0.,0.]:
            assert len(self.fin_dirc) != 0, "You don't have a slab/ribbon, please\
                call cut_piece function first"

            if fin_dirc is None:
                fin_dirc = self.fin_dirc[0]
            else:
                assert fin_dirc in self.fin_dirc, "Wrong fin_dirc"
            bot_bound = edge[0] / self.multi
            top_bound = edge[1] / self.multi
            bot_coord = min(self.structure.prm_dirc[:,fin_dirc])
            top_coord = max(self.structure.prm_dirc[:,fin_dirc])
            for i, coord in enumerate(self.structure.prm_dirc):
                if coord[fin_dirc] - bot_coord < bot_bound or\
                    top_coord - coord[fin_dirc] < top_bound:
                    edge_atoms[i] = 1
        
        y_res = 200
        k_points, k_dist, node_dist = self._make_k_path(nodes, k_num=k_num)
        # at each k, build a fine grids and fill them with numbers according to edge_atoms
        weights = np.zeros((y_res, len(k_points)))
        for i in range(0,len(k_points)):
            weights[:,i] = self._pixel_band(k_points[i],
                                            edge_atoms,
                                            ylim,
                                            y_res,
                                            sigma,
                                            unit_num)     
            # print(max(weights[:,i]))
        y_grid = np.linspace(ylim[0], ylim[1], y_res)
        X, Y = np.meshgrid(k_dist, y_grid)
        fig, ax = plt.subplots()
        ax.pcolormesh(X, Y, weights, cmap='gist_heat')
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("Wavevector", fontsize=16)
        
        #set node labels
        if node_names is not None:
            ax.set_xticks(node_dist)
            ax.set_xticklabels(node_names)
        ax.tick_params(labelsize=15)
        if len(self.fin_dirc) == 0:
            self._make_title(ax)
        
        plt.show()
        
        return fig

    
    def _sample_2d_band(self,
                        band_index: int,
                        center: List[float],
                        xy_range: float,
                        dirc: int,
                        z: float,
                        k_num: int,
                        unit: float
                        ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        compute all wavefunctions on a 2d grid.
        
        """
        step = xy_range/k_num/2
        kx = np.arange(center[0]-xy_range, center[0]+xy_range+step, step)
        ky = np.arange(center[1]-xy_range, center[1]+xy_range+step, step)
        X, Y = np.meshgrid(kx, ky)
        wfs = np.zeros((X.shape[0], X.shape[1]),dtype=complex)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if self.dim == 2:
                    kpt = [X[i,j], Y[i,j]]
                elif self.dim == 3:
                    if dirc == 0:
                        kpt = [z, X[i,j], Y[i,j]]
                    elif dirc == 1:
                        kpt = [Y[i,j], z, X[i,j]]
                    else:
                        kpt = [X[i,j], Y[i,j], z]
                else:
                    raise Exception("To build 3d_band plot, the dim must be 2 or 3")
                _, all_freq =\
                    self.solve_dynamical_matrix_kpath([kpt],
                                                      k_num=1,
                                                      unit=unit,
                                                      eig_vec=False)
                wfs[i,j] = copy.deepcopy(all_freq[0][band_index])
        return X, Y, wfs
    
    
    def plot_3d_band(self,
                     band_indexes: List[int],
                     center: List[float],
                     xy_range: float,
                     dirc: int = 2,
                     z: float = 0.0,
                     k_num: int = 10,
                     tol: float = 0.5,
                     view: Union[List[float], None] = None,
                     unit: Union[str, float] = 'thz'
                     ) -> matplotlib.figure.Figure:

        """
        plot one or two bands on a 2d squared grid, with energies on the z axis
        
        Parameters
        ----------
        band_indexes: list of int
            a list of integers correspoding to band indexes
        center : list, [float, float]
            two floats that specify the coordinates of the center on the 2d square
        xy_range : float
            the edge length of the 2d square
        dirc : int, optional
            the direction that is normal to the 2d grid
            The default is 2
        z : float, optional
            the third coordinate that defines the plane
            The default is 0.0
        k_num : int, optional
            the number of points sampled in both directions
            The default is 10
        tol : float, optional
            if the length of band_indexes is two, the k_point where the energy
            difference is below tol will be dotted. In many cases the value of t
            ol should be increased to see the degenerate points.
            The default is 0.5
        view : list, [float, float], optional
            two numbers that changes the view of the plot; the first number sets
            the elevation (degree above or below the xy-plane) while the second
            number sets the azimuth (degree rotated about z-axis)
            The default is None
        unit : str or float, optional
            the unit of the plot; can be "thz", "cm-1", "cm^-1", "ev", "mev". 
            The default is "thz"
        Returns
        -------
        A matplotlib.figure.Figure object containing phonon band plot
        """

        from mpl_toolkits.mplot3d import Axes3D
        
        # scale the tolerance
        tol *= xy_range / k_num
        # convert unit 
        unit_num = self._convert_unit(unit)

        assert len(band_indexes) in (1,2), "You can specify 1 or 2 bands"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # get zlabel
        zlabel = _set_ylabel(unit)
        ax.set_zlabel(zlabel)
        
        X, Y, wfs_1 = self._sample_2d_band(band_indexes[0],
                                           center,
                                           xy_range,
                                           z=z,
                                           dirc=dirc,
                                           k_num=k_num,
                                           unit=unit_num)
        ax.plot_surface(X, Y, wfs_1, shade=False, alpha=0.4, color='orange')
        if len(band_indexes) == 2:
            X, Y, wfs_2 = self._sample_2d_band(band_indexes[1],
                                               center,
                                               xy_range,
                                               z=z,
                                               dirc=dirc,
                                               k_num=k_num,
                                               unit=unit_num)   
            ax.plot_surface(X, Y, wfs_2, shade=False, alpha=0.6, color='blue')
            # find degenerate points
            print(len(X[1]))
            wf_diff = wfs_1 - wfs_2
            for i in range(len(X[0]-1)):
                for j in range(len(X[1]-1)):
                    if abs(wf_diff[i][j]) < tol:
                        ax.scatter(
                            float(X[i][j]), float(Y[i][j]), float(wfs_1[i][j]),
                            c='black', s=30)
                        print("degenerate point around [{}, {}]".format(X[i][j], Y[i][j]))
        if view is not None:
            ax.view_init(view[0], view[1]) 
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return fig


    def intensity(self,
                  H: List[float],
                  k: List[float],
                  DW_matrices: List[np.array],
                  T: float = 0.0,
                  e_range: Union[List[float], None] = None,
                  de: float = 0.5,
                  branches: Union[str, List[int]]="all"
                  ) -> np.array:
        """
        calculate electron diffuse scattering intensity for a given scattering vector 
        
        Parameters
        ----------
        H : list of int
            Reciprocal lattice vector
        k : list of float
            Phonon wavevector; The scattering vector q is given by H + k
        DW_matrices : list of ndarray
            Debye-Waller coefficient matrix for each atom, which is calculated from DW_coefficient() function
        T : float, optional
            Temperature at which the intensity is calculated in Kelvin
            The default is 300.0
        e_range : list of float, optional
            Intensity is evaluated within e ~ [e_range[0], e_range[1]]. If None, the whole energy range is considered
            The default is None
        de : float, optional
            The intensity is evaluated every de THz
            The default is 0.5
        branches : list of int, optional
            The indices of band included in the intensity calculation. If 'all', all the bands are .
            The default is 'all'
        Returns
        -------
        ndarray contains all the intensities.
        """
        
        H = np.array(H, dtype=complex)
        k = np.array(k, dtype=complex)
        # scattering factor q
        q = H + k
        q_actual = np.matmul(self.structure.k_lat, q) * 10**(-10)   # actual q value in unit of 1/A
        # set a minimum temperature to avoid overflow
        if T < 1:
            T = 1
            
        nb_atom = len(self.structure.atoms)
        if branches == "all":
            nb_band = nb_atom * self.dim
            branches = [i for i in range(nb_band)] 
        else:
            nb_band = len(branches)
            
        #atomic scattering factor
        fs = np.ones(nb_atom)
        # atomic mass
        mius = np.ones(nb_atom)
        # Debye-Waller factor
        ms = np.ones(nb_atom)
        #occupation number 
        njk = np.ones(nb_band)
        # frequencies
        omegajk = np.ones(nb_band, dtype=complex)
        # polarization vector
        ejsk = np.ones((nb_band, nb_atom, self.dim), dtype=complex)
        # positions of atoms 
        pos = np.ones((nb_atom, self.dim))
   
        for s in range(nb_atom):
            # set masses
            mius[s] = masses_dict[self.structure.atoms[s].upper()]

        # set atomic positions
        # pos = self.structure.prm_dirc
            
        # set Debye-Waller Matrix
        for s in range(nb_atom):
             ms[s] = np.matmul(np.matmul(q_actual, DW_matrices[s]), q_actual)
        
        # set scattering factor
        for i in range(len(fs)):
            ai = SCATTER_PAR_A[self.structure.atoms[i]]
            # print(ai)
            bi = SCATTER_PAR_B[self.structure.atoms[i]]
            # c = SCATTER_PAR_C[self.structure.atoms[i]]
            z = ATOMS.index(self.structure.atoms[i])
            f = z
            for j in range(len(ai)):
                # X-ray scattering form factor
                f -= 41.78214 * np.linalg.norm(q)**2 * ai[j] * np.exp(-bi[j] * np.linalg.norm(q)**2)
            # get electron scattering form factors using Mott-Bethe formula
            fs[i] = 0.02393 * (z - f) / np.linalg.norm(q)**2
        
        # get frequencies and eigenvectors at k
        dy_mx, freq, v = self.solve_dynamical_matrix_kpath([k])
        
        if not e_range:
            e_range = [min(freq[0]) * VASP2THZ, max(freq[0]) * VASP2THZ]
            de = max(freq[0]) * VASP2THZ - min(freq[0]) * VASP2THZ
        res = np.zeros(int(round((e_range[1].real - e_range[0].real) / de.real)))
        # print(res)
        # sum over all the bands in e+de                    
        for ne in range(len(res)):
            e_min = (e_range[0] + de * ne) * unit_dict['ev']
            e_max = (e_range[0] + de * ne + de) * unit_dict['ev']
            # print(e_min, e_max)
            intensity = 0
            for j in range(nb_band):
                # get omega_jk and n_jk
                omegajk[j] = np.array(freq[0][branches[j]], dtype=complex) * VASP2THZ * unit_dict['ev']
                # print(freq[0][branches[j]])
                # count only if the band is in e_range
                if not e_min <= omegajk[j] <= e_max:
                    continue
                njk[j] = 1 / (np.exp(omegajk[j] / KB / T) - 1)
                factor = (njk[j] + 1/2) / omegajk[j]
                # print(factor)
                fjq = 0 + 0j
                
                # get polarization vectors
                for s in range(nb_atom):
                    ejsk[j][s] = np.array(v[0][:,branches[j]][s*self.dim : s*self.dim+self.dim],dtype=complex)
                    
                #sum over all the atoms
                for s in range(nb_atom):
                    # phase = np.exp(-1j * (H) * pos[s]) 
                    phase = 1
                    fjq += fs[s] / np.sqrt(mius[s]) * np.exp(-ms[s]) * \
                        np.dot(q, ejsk[j][s]) * phase
                intensity += factor * np.dot(fjq.conjugate(), fjq) 
            res[ne] = intensity
            
        return res


    def DW_coefficient(self,
                       T: float,
                       k_num: int = 10
                       ) -> List[np.array]: 
        '''
        return the Debye Waller coefficient matrix for atom s
        '''
        
        HB = HBARCGS * 10 ** 16  # cm^2gs-1 to A^2gs-1
        
        nb_atom = len(self.structure.atoms)
        nb_band = nb_atom * self.dim
        
        DW_matrices = []
        for s in range(nb_atom):
            DW_matrix = np.zeros((3,3))
            mius = masses_dict[self.structure.atoms[s].upper()] / MOLE   # in unit of g
            # span the whole Brillouin zone
            for kx in range(1,k_num):
                for ky in range(1,k_num):
                    for kz in range(1,k_num):
                        kpt = np.array([kx / k_num, ky / k_num, kz / k_num])
                        dy_mx, freq, v = self.solve_dynamical_matrix_kpath([kpt])
                        
                        # iterate over all the bands
                        for j in range(nb_band):     
                            if abs(freq[0][j]) <= 0.01:
                                return np.ones((3,3))
                            omegajk = np.array(freq[0][j], dtype=complex) * VASP2THZ * 10 ** 12   # THz to Hz
                            akj2 = HB * _coth(HBAR*omegajk/2/KB/T) / freq[0][j]
                            ejsk = v[0][:,j][s*self.dim : s*self.dim+self.dim]
                            # fill the 3x3 matrix
                            for mx in range(3):
                                ealpha = ejsk[mx]
                                for my in range(3):
                                    ebeta = np.conjugate(ejsk[my])
                                    DW_matrix[mx][my] += akj2 * ealpha * ebeta
            DW_matrices.append(DW_matrix / mius)                    
            
        return DW_matrices
                        
    
def read_from_files(path):
    """
    Allows users to build tb-model by reading POSCAR, SPOSCAR and FORCE_CONSTANTS 

    Parameters
    ----------
    path : str
        path of a directory that contains "POSCAR", "SPOSCAR" and "FORCE_CONSTANTS"  

    Returns
    -------
    a Model object
    """
    import os
    # make a structure object
    poscar = os.path.join(path, "POSCAR")
    sposcar = os.path.join(path, "SPOSCAR")
    fcs = os.path.join(path, "FORCE_CONSTANTS")
    structure = Structure(3)
    structure.read_POSCAR(poscar)
    structure.read_supercell(sposcar)
    model = Model(structure)
    model.read_fc(fcs)
    return model