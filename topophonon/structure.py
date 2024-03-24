# -*- coding: utf-8 -*-
"""
This module contains the class that describes the crystal structure related 
properties
"""

from topophonon.units import masses_dict
import numpy as np
import os
import re
import warnings
from typing import List

class Structure():
    
    #regex for finding coordinates
    PATTERN = "(-?\d+\.?\d+)\s*(-?\d+\.?\d+)\s*(-?\d+\.?\d+).*"

    
    def __init__(self,
                 dim: int,
                 lat: List[List[float]] = None,
                 coords: List[List[float]] = None,
                 masses: List[float] = None,
                 atoms: List[str] = None,
                 shift: List[float] = [0.0, 0.0, 0.0]):
        """
        Parameters
        ----------
        dim : int
            dimension of the crystal.
        lat : 2d list or ndarray, optional
            a matrix defines the lattice parameters in cartesian coordinates.
            The default is None
        coords : 2d list or ndarray, optional
            a list of vectors define the coordinates of atoms in the unit cell.
            The default is None
        masses : list, optional
            a list contains the massses of atoms. If not specified, 
            default values will be used based on "atoms"
            The default is None
        atoms : list, optional
            a list contains the name of atoms.
            The default is None.
        """
        
        self.dim = dim
        assert isinstance(self.dim, int) and 1 <= dim <= 3, "dimension must be 1, 2 or 3"
        if lat is None:
            self.lat = np.zeros((self.dim,self.dim))
        else:
            self.lat = np.array(lat,dtype=float)
            # calculate the reciprocal lattice
            # self.k_lat = np.linalg.inv(np.dot(self.lat,self.lat.T))
            self.k_lat = np.linalg.inv(self.lat.T)
            
        if coords is not None:
            for coord in coords:
                if len(coord) != self.dim:
                    raise ValueError("The lenght of a coordinate must match dim")
            self.coords = coords
                
            self.prm_dirc = np.array(coords,dtype=float)
            self.prm_cart = self._direct_to_cartesian(self.prm_dirc, self.lat)
            
        if masses is None:
            self.masses = None
        # set the atoms and default masses
        self._set_atoms(atoms)
        if masses is not None:
            assert isinstance(masses, list), ("the masses must be given in a list")
            if self.masses is not None:
                warnings.warn("masses are already given, will be overwritten")
            self.masses = masses
        self.shift = np.array(shift)
       
        
    def _atoms_to_masses(self,
                         atoms: List[str]):
        """
        convert a list of atoms to their masses based on their names
        """
        masses = []
        for a in atoms:
            assert isinstance(a,str), "the atoms must be given in string type"
            assert a.upper() in masses_dict,\
                "unrecognized atom type {}".format(a)
            masses.append(masses_dict[a.upper()]) 
        self.masses = masses
        
        
    def _set_atoms(self,
                   atoms: List[str]):
        """
        set atom list and mass list
        
        """
        if atoms is None:
            self.atoms = None
            return
        warnings.warn("default masses are used")
        self.atoms = atoms   
        self._atoms_to_masses(atoms)
    
        
    def set_lat(self,
                lat: np.ndarray):
        """
        set real space and reciprocal lattice tensor 
        
        Parameters
        ----------
        lat : 2d ndarray of float
            3 lattice vectors 
        """
        
        self.lat = lat
        self.k_lat = np.linalg.inv(np.dot(lat, lat.T))
    
    
    @staticmethod    
    def _direct_to_cartesian(coord: np.ndarray,
                             lat: np.ndarray) -> np.ndarray :
        """
        convert a direct coordinate to a cartesian coordinate

        """
        return np.dot(coord, lat)
        
    
    @staticmethod  
    def _cartesian_to_direct(coord: np.ndarray,
                             lat: np.ndarray) -> np.ndarray :
        """
        convert a cartesian coordinate to a direct coordinate

        """
        return np.dot(coord, np.linalg.inv(lat))
    

    def read_POSCAR(self, poscar: str):
        """
        read the VASP POSCAR file
    
        Parameters
        ----------
        poscar : str
            the path of POSCAR file
        """
        
        assert self.dim == 3, "To read files from phonopy, the dimension must be 3"
        assert os.path.exists(poscar), "{} not existed".format(poscar)
        
        self.prm_dirc, self.prm_cart = [], []    
        # try:
        with open(poscar) as p:
            # skip the fist line
            p.readline()
            # the factor of lattice
            factor = float(p.readline().strip())
            # read the lattice constants
            for i in range(3):
                r_str = p.readline().strip().split()
                r_float = [float(r) * factor for r in r_str]
                self.lat[i] = np.array(r_float) 
            self.k_lat = np.linalg.inv(self.lat.T) 
            # read atoms
            atoms = p.readline().strip().split()
            counts = p.readline().strip().split()
            all_atoms = []
            for i in range(len(counts)):
                for _ in range(int(counts[i])):
                    all_atoms.append(atoms[i])     
            #convert atoms to masses  
            if self.masses is None:
                self._set_atoms(all_atoms)
            
            # read the mode
            mode = p.readline().strip()
            
            # read all coordinates
            if mode.lower().startswith('c'):     
                pass
            
            elif mode.lower().startswith('d'):
                for i in range(len(self.atoms)):
                    coord_str = re.findall(Structure.PATTERN, p.readline())[0]
                    coord_array = np.array([float(r) for r in coord_str])
                    self.prm_dirc.append(coord_array) 
                    self.prm_cart.append(
                        self._direct_to_cartesian(coord_array, self.lat))                      
                self.prm_dirc = np.array(self.prm_dirc)
                self.prm_cart = np.array(self.prm_cart)
            else:
                raise Exception("unknown mode in POSCAR; can be direct or cartesian")
        # except:
        #     raise ValueError("Something goes wrong with {}".format(poscar))
    
    
    def read_supercell(self, sposcar: str):
        """
        read the supercell file which can be generated by phonopy
        
        Parameters
        ----------
        sposcar : str
            the path of SPOSCAR file
    
        """
        
        assert os.path.exists(sposcar), "{} not existed".format(sposcar)
        
        #convert the original coordinates to cartesian coordinates
        # org_coords_cart = []
        # for coord in self.prm_dirc:
        #     org_coords_cart.append(self._direct_to_cartesian(coord, self.lat))
        # org_coords_cart = np.array(org_coords_cart)
            
        self.super_dirc, self.super_cart = [], []
        self.super_lat = np.zeros((self.dim,self.dim))
        #Read information in the supercell
        try:
            with open(sposcar) as p:
                p.readline()
                factor = float(p.readline().strip())
                # read the lattice constants
                for i in range(3):
                    r_str = p.readline().strip().split()
                    r_float = [float(r) * factor for r in r_str]
                    self.super_lat[i] = np.array(r_float)     
                p.readline()
                #total number of atoms in the supercell
                nums = p.readline().strip().split()
                nums = [int(num) for num in nums]
                total = sum(nums)
                # read the mode
                mode = p.readline().strip()
                
                # read all coordinates
                if mode.lower().startswith('c'):     
                    pass
                elif mode.lower().startswith('d'):
                    
                    for i in range(total):
                        coord_str = re.findall(Structure.PATTERN, p.readline())[0]
                        coord = np.array([float(r) for r in coord_str])
                        self.super_dirc.append(coord)
                        self.super_cart.append(
                            self._direct_to_cartesian(coord, self.super_lat))
                    self.super_dirc = np.array(self.super_dirc)
                    self.super_cart = np.array(self.super_cart)
        except:
            raise ValueError("Something goes wrong with {}". format(sposcar))


    @staticmethod  
    def _write_structure(file, lat, atoms, count, coords):
        with open (file, 'w') as f:
            f.write('Output from topoPhonon\n')
            f.write('  {:.6f} \n'.format(1))
            f.write("    {:.6f}    {:.6f}    {:.6f}\n    {:.6f}    {:.6f}    {:.6f}\n    {:.6f}    {:.6f}    {:.6f}\n"\
                          .format(lat[0][0], lat[0][1], lat[0][2],
                                  lat[1][0], lat[1][1], lat[1][2],
                                  lat[2][0], lat[2][1], lat[2][2],))
            for atom in atoms:
                f.write("    {}".format(atom))       
            f.write('\n')
            for c in count:
                f.write("    {}".format(c))
            f.write('\n')
            f.write('Direct')
            f.write('\n')
            for coord in coords:
                f.write('    {:.6f}    {:.6f}    {:.6f}\n'.format(coord[0], coord[1], coord[2])) 

   
    def write_poscar(self, file):
        """
        output the structure in VASP POSCAR format
        
        Parameters
        ----------
        file : str
            the path of output POSCAR file
    
        """
        lat = self.lat
        atoms, count = np.unique(self.atoms,return_counts=True)
        coords = self.prm_dirc
        self._write_structure(file, lat, atoms, count, coords)