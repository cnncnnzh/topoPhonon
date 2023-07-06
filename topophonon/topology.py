# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:10:49 2022

@author: zhuhe
"""

from topophonon.utils import _make_k_grid

from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import copy
import numpy as np
import warnings
import functools
from copy import deepcopy

class Topology():
    """
    Class that contains topology-related properties and methods  
    """
    
    
    def __init__(self, model, dim=None):
        self.model = model
        self.nb = len(model.structure.masses) * model.dim
        if dim is None:
            self.dim = model.dim
        else:
            self.dim = dim    

    def _wfs_at_kpt(self, kpt):
        """
        Return the modified the phonon eigenvectors at kpt 
        """
        modified_wf = []
        dy_mt, all_freqs, eig_vecs =\
            self.model.solve_dynamical_matrix_kpath([kpt], k_num=1)
        dim = len(all_freqs[0])
        for i in range(dim):
            # convert the row vector to column vector
            # old_wf = eig_vecs[0][:,i].reshape((-1,1))
            # freq = all_freqs[0][i]
            # new_wf = np.zeros(dim*2, dtype=complex)
            # new_wf[:dim] = np.dot(sqrtm(dy_mt), old_wf).flatten()
            # new_wf[dim:] = (-1j * freq * old_wf).flatten()
            # modified_wf.append(new_wf)
            old_wf = np.array(eig_vecs[0][:,i])
            modified_wf.append(old_wf)
        return modified_wf

        
    def _gen_k_loop_bz(self, k_along, k_start, k_num=50):
        """
        Generate a closed k loop for wilson_loop function

        """
        step = np.array([0 for _ in range(self.dim)],dtype=float)
        step[k_along] = 1/(k_num-1)
        # self._back_to_bz(k_start)
        k_grids = [copy.deepcopy(k_start)]
        for i in range(k_num-1):
            k_start += step
            # print(k_start)
            # self._back_to_bz(k_start)
            # print(k_start)
            temp = copy.deepcopy(k_start)
            k_grids.append(temp)
        return np.array(k_grids)  
    

    def gen_BZ_wfs(self, k_along, k_start, k_num=60):
        """
        Generate the phonon eigenvectors along a closed line over the whole brillioun zone.
        
        Parameters
        ----------
        k_along : int
            the direction of the line
        k_start : list of float
            the starting (and the ending) point of the line 
        k_num : int, optional
            total number of kpoints on the line
            The default is 60
        Returns
        -------
        an array of eigenvectors along the line
        """
        k_start = np.array(k_start, dtype=float)
        k_grids = self._gen_k_loop_bz(k_along, k_start, k_num)
        self.k_grids = k_grids
        all_wfs = []
        for kpt in k_grids:
            modified_wf = self._wfs_at_kpt(kpt)
            all_wfs.append(modified_wf)
        return np.array(all_wfs)   
     
    
    def gen_circle_wfs(self, center, dirc=2, z=0, r=0.05, k_num=60):
        """
        Generate the eigenvectors along a circle.
        
        Parameters
        ----------
        center : list, [float, float]
            two floats that specify the coordinates of the center
        dirc : int, optional
            the direction that is normal to the circle
            The default is 2
        z : float, optional
            the third coordinate that defines the plane of the circle; for example,
            if the dirc=0 and z=0.5, the wfs are generated on x=0.5 plane. 
            The default is 0.0
        r : float, optional
            the radius of the circle
            The default is 0.05
        k_num : int, optional
            total number of kpoints on the line
            The default is 60
        Returns
        -------
        an array of eigenvectors on the circle
        """

        wfs = []
        for i in range(k_num):
            theta = 2 * np.pi * i / k_num
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            if self.dim == 2:
                wfs.append(self._wfs_at_kpt(np.array([x,y])))
            else:
                if dirc == 0:
                    wfs.append(self._wfs_at_kpt(np.array([z,x,y])))
                elif dirc == 1:
                    wfs.append(self._wfs_at_kpt(np.array([y,z,x])))
                elif dirc == 2:
                    wfs.append(self._wfs_at_kpt(np.array([x,y,z])))
                else:
                    raise Exception("dirc must be 0, 1 or 2 if dim of the model is 3")
        # make sure the last wf is equal to the first one
        wfs.append(wfs[0])
        return np.array(wfs)

    
    def _gen_orbit_wfs(self, center, r, theta, dirc, k_num=60):
        """
        Generate an orbit around for a given center 
        """
        wfs = []
        for i in range(k_num+1):
            phi = 2 * np.pi * i / k_num
            x = center[0] + r * np.sin(theta) * np.cos(phi)
            y = center[1] + r * np.sin(theta) * np.sin(phi)
            z = center[2] + r * np.cos(theta)
            if dirc == 0:
                kpt = np.array([y,z,x])
            elif dirc == 1:
                kpt = np.array([z,x,y])
            elif dirc == 2:
                kpt = np.array([x,y,z])
            else:
                raise Exception("dirc must be 0, 1 or 2")
            # print(kpt)
            wfs.append(self._wfs_at_kpt(kpt))
        return np.array(wfs)


    @staticmethod
    def wilson_loop(band_indices, all_wfs,):
        """
        Compute the berry phase of a given set of eigenvectors, the first eigenvector
        must be equal to the last eigenvector
        
        Parameters
        ----------
        band_indices : list of ints
            list of band indices for which the wilson loop is calculated.
        all_wfs : ndarray of complex
            an array of eigenvectors around a closed path.
        
        Returns
        -------
        berry phase calculated with wilson loop method
        """

        if isinstance(band_indices, int):
            band_indices = [band_indices]
        num_bands = len(band_indices)
        k_num = len(all_wfs)
        prod = np.identity(num_bands, dtype=complex)
        det = 1
        #iterate over all kpoints on the loop
        for i in range(k_num-1):
            #container for overlap matrix
            ovlp = np.zeros([num_bands,num_bands],dtype=complex)
            #iterate over all bands for overlap matrix
            for j in range(num_bands):
                for k in range(num_bands):
                    wf_1 = all_wfs[i][band_indices[j]] 
                    wf_2 = all_wfs[i+1][band_indices[k]] 
                    ovlp[j,k] = np.dot(wf_1.conjugate(),wf_2)  
        #     det *= np.linalg.det(ovlp)
        #     print(np.linalg.det(ovlp),det)
        # phase = -np.imag(np.log(det)) / np.pi          
            # calculate determinant for each overlap matrix and then multiply?
            # multiply all overlap matrices
            prod = np.dot(prod,ovlp)
            # print(prod)
        # compute the determinant
        det = np.linalg.det(prod)
        # the flux for this small plaquette
        phase = (-1.0)*np.angle(det, deg=False)/np.pi   
        return phase
    
    
    def wcc_evol_sphere(self, band_indices, center, r=0.001, dirc=2, num=60,):
        """
        Generate a sphere and plot the evolution of wannier centers around that sphere,
        for 3d only. The sphere is sliced into multiple orbitals. Wilson loop
        method is applied to calculate the wannier center on each orbit. 
        
        Parameters
        ----------
        band_indices : list of int
            list of band indices on which the wannier centers are calculated.
        center : list, [float, float, float]
            three floats that specify the coordinates of the sphere center
        r : float, optional
            the radius of the sphere
            The default is 0.001
        dirc : int, optional
            the direction along which theta evolves; in other words, the orbitals 
            are perpendicular to those orbitals
            The default is 2
        num : int, optional
            number of slices and number of k points on the loop
            The default is 60
        
        Returns
        -------
        berry phase calculated with wilson loop method
        """
        
        if isinstance(band_indices, int):
            band_indices = [[band_indices]]
        if isinstance(band_indices[0],int):
            band_indices = [band_indices]
        print("calculating wcc charge center evolution around {}...".format(center))
        polar_angles = []
        all_wccs = [[] for _ in range(len(band_indices))]
        for i in range(num+1):
            theta = np.pi * i / num
            polar_angles.append(theta/np.pi)
            wfs = self._gen_orbit_wfs(center, r, theta, dirc, k_num=num)
            for j in range(len(band_indices)):
                w = self.wilson_loop(band_indices[j], wfs)
                all_wccs[j].append(w)
        # self.polar_angles = polar_angles
        # self.wccs = wccs
        for i, wccs in enumerate(all_wccs):
            plt.figure()
            plt.scatter(polar_angles, wccs)
            plt.ylim(-1, 1)
            plt.xlim(0, 1)
            plt.title("band {}".format(str(band_indices[i])))
        # return polar_angles, all_wccs 


    def berry_curvature(self, kpt, band_indices, delta=1e-9):
        """
        Compute the berry curvature for one or more bands on a given k point,
        using Kubo formulation.
        Works for 2d and 3d cases.
        
        Parameters
        ----------
        kpt : list of floats
            the coordinates of the kpoint; the length must be equal to self.dim
        band_indices : list of ints
            list of band indices on which the berry curvature is calculated.
        delta : float, optional
            a small distance between two points for differentiating the Hamiltonian.
            The default is 1e-9
            
        Returns
        -------
        a 3/1-component vector for 3-d/2-d models
        """

        assert len(kpt) == self.dim, ("wrong dimension of kpt")
        # build a wilson loop adjacent to the point
        d, freqs, eig_vecs =\
            self.model.solve_dynamical_matrix_kpath([kpt],
                                                    k_num=1)  
        num_deg = len(band_indices)
        num_bands = len(freqs[0])
        f0 = [freqs[0][i] for i in band_indices]
        wfs = [eig_vecs[0][:,i] for i in band_indices] 

        if self.dim == 2:
            # total = np.array(0+0j)  
            kxp = kpt + np.array([delta,0])
            kyp = kpt + np.array([0,delta])
            d_x, _, _ = self.model.solve_dynamical_matrix_kpath([kxp],
                                                           k_num=1,)
            d_y, _, _ = self.model.solve_dynamical_matrix_kpath([kyp],
                                                           k_num=1,)
            delta_x = (d_x - d)/delta
            delta_y = (d_y - d)/delta
            # build the berry curvature matrix B_ij
            total = np.zeros((num_deg, num_deg),dtype=complex)
            for i in range(num_deg):
                for j in range(num_deg):
                    fi, fj = f0[i], f0[j]
                    wfi, wfj = wfs[i], wfs[j]
                    # iterate over all bands
                    for m in range(num_bands):
                        if m in band_indices:
                            continue
                        wfm = np.array(eig_vecs[0][:,m])
                        fm = freqs[0][m]
                        if abs(fi - fm) < 1e-11:
                            raise Exception("band {} and {} are degenerate"\
                                            .format(i, m))
                        if abs(fj - fm) < 1e-11:
                            raise Exception("band {} and {} are degenerate"\
                                            .format(j, m))
                        prod1, prod2 = np.zeros(2, dtype=complex), np.zeros(2, dtype=complex)
                        prod1[0] = np.dot(wfi.conjugate(),np.dot(delta_x, wfm))
                        prod1[1] = np.dot(wfi.conjugate(),np.dot(delta_y, wfm))
                        prod2[0] = np.dot(wfm.conjugate(),np.dot(delta_x, wfj))
                        prod2[1] = np.dot(wfm.conjugate(),np.dot(delta_y, wfj))
                        # total[i][j] += np.cross(prod1, prod2) / (fi-fm) / (fj-fm)
                        total[i][j] += np.cross(prod1, prod2) / (fi-fm) / (fj-fm)
        elif self.dim == 3:
            # total = np.zeros(3, dtype=complex)
            kxp = kpt + np.array([delta,0,0])
            kyp = kpt + np.array([0,delta,0])
            kzp = kpt + np.array([0,0,delta])
            d_x, _, _ = self.model.solve_dynamical_matrix_kpath([kxp],
                                                           k_num=1,)
            d_y, _, _ = self.model.solve_dynamical_matrix_kpath([kyp],
                                                           k_num=1,)
            d_z, _, _ = self.model.solve_dynamical_matrix_kpath([kzp],
                                                           k_num=1,)
            delta_x = (d_x - d)/delta
            delta_y = (d_y - d)/delta
            delta_z = (d_z - d)/delta

            # build the berry curvature matrix B_ij
            total = np.zeros((num_deg, num_deg, 3),dtype=complex)
            for i in range(num_deg):
                for j in range(num_deg):
                    fi, fj = f0[i], f0[j]
                    wfi, wfj = wfs[i], wfs[j]
                    # iterate over all bands
                    for m in range(num_bands):
                        if m in band_indices:
                            continue
                        wfm = np.array(eig_vecs[0][:,m])
                        fm = freqs[0][m]
                        if abs(fi - fm) < 1e-15:
                            warnings.warn("band {} and {} are likely to be degenerate at {}"\
                                            .format(band_indices[i], m, kpt))
                        if abs(fj - fm) < 1e-15:
                            warnings.warn("band {} and {} are likely to be degenerate at {}"\
                                            .format(band_indices[j], m, kpt))
                        prod1, prod2 = np.zeros(3, dtype=complex), np.zeros(3, dtype=complex)
                        prod1[0] = np.dot(wfi.conjugate(),np.dot(delta_x, wfm))
                        prod1[1] = np.dot(wfi.conjugate(),np.dot(delta_y, wfm))
                        prod1[2] = np.dot(wfi.conjugate(),np.dot(delta_z, wfm))
                        prod2[0] = np.dot(wfm.conjugate(),np.dot(delta_x, wfj))
                        prod2[1] = np.dot(wfm.conjugate(),np.dot(delta_y, wfj))
                        prod2[2] = np.dot(wfm.conjugate(),np.dot(delta_z, wfj))
                        total[i][j] += np.cross(prod1, prod2) / (fi-fm) / (fj-fm)
        else:
            raise Exception("To calculate berry curvatures, the dimension must\
                            be 2 or 3")
        return np.real(1j * np.trace(total)) 
        
    
    # @staticmethod    
    # def _log_scale(x):
    #     # scale the vector logarithmically
    #     return np.sign(x)*np.log10(1+np.abs(x))
    
    
    # @staticmethod
    # def _normalization(x, y):
    #     return x/np.sqrt(x**2 + y**2), y/np.sqrt(x**2 + y**2)
    
    
    def berry_curvature_proj(self,
                            band_indices,
                            dirc,
                            kz,
                            center=[0,0],
                            xy_range=0.5,
                            num=10,):

        """
        plot 3d berry curvature for one or more bands projected on a 2D plane, 
        "dirc" is normal to the plane, "center" specify the center of the map; 
        "kz" determines the position of the plane on "dirc" axis. for example:
        dirc=2, kz=0.5, center=[0,0], the berry curvature is plotted on xy 
        plane with kz=0.5 around [0,0,0.5] point.
        
        Parameters
        ----------
        band_indices : list of ints
            list of band indices on which the berry curvature is calculated.
        dirc : int
            the direction that is normal to the plane on which the berry curvatures
            are projected.
        kz : float
            the third coordinate that defines the plane of the circle; for example,
            if the dirc=0 and z=0.5, the berry curvatures will be plotted on x=0.5 plane. 
        center : list, [float, float], optional
            coordinate of the point around which the berry curvatures are plotted;
            specify only two component on the plane; the third coordinate is specified by kz. 
        xy_range : float
            the range of the grid, i.e., the plot spans [center[0]-xy_range, 
            center[0]+xy_range] and [center[1]-xy_range, center[1]+xy_range]
            The default is 0.5
        num : int, optional
            number of points sampled on the plane is given by num * num
            The default is 10

        """

        print("start plotting the berry curvature field map...")
        if isinstance(band_indices, int):
            band_indices = [band_indices]
        # # a1 = np.arange(-xy_range[0], -interval/2, interval)        
        # # a2 = np.arange(interval/2, xy_range[0], interval) 
        # # a = np.hstack((a1,a2))
        # interval_0, interval_1 = xy_range[0]/num, xy_range[1]/num
        # a = np.arange(-xy_range[0], xy_range[0], interval_0)
        # # b1 = np.arange(-xy_range[1], -interval/2, interval)        
        # # b2 = np.arange(interval/2, xy_range[1], interval)
        # # b = np.hstack((b1,b2))
        # b = np.arange(-xy_range[1], xy_range[1], interval_1)
        # kx,ky = np.meshgrid(a,b)
            kx, ky = _make_k_grid(xy_range, num)
        kx0, ky0 = center[0], center[1]
        #the x and y component of berry curvature
        # for group in band_indices:

        u = np.zeros((len(kx), len(kx[0])), dtype=float)  
        v = np.zeros((len(kx), len(kx[0])), dtype=float) 
        plt.figure()   
        for i in range(len(kx)):
            for j in range(len(kx[0])):
                if dirc == 0:
                    kpt = np.array([kz,kx0+kx[i][j],ky0+ky[i][j]])
                    berry_curv = self.berry_curvature(kpt, band_indices)
                    norm = np.sqrt(berry_curv[1]**2 + berry_curv[2]**2)
                    if norm == 0:
                        u[i][j] = None
                        v[i][j] = None
                    else:
                        u[i][j] = berry_curv[1]/norm
                        v[i][j] = berry_curv[2]/norm
                elif dirc == 1:
                    kpt = np.array([ky0+ky[i][j],kz,kx0+kx[i][j]])
                    berry_curv = self.berry_curvature(kpt, band_indices)
                    norm = np.sqrt(berry_curv[2]**2 + berry_curv[0]**2)
                    if norm == 0:
                        u[i][j] = None
                        v[i][j] = None
                    else:
                        u[i][j] = berry_curv[2]/norm
                        v[i][j] = berry_curv[0]/norm
                elif dirc == 2:
                    kpt = np.array([kx0+kx[i][j],ky0+ky[i][j],kz])
                    berry_curv = self.berry_curvature(kpt, band_indices)
                    norm = np.sqrt(berry_curv[0]**2 + berry_curv[1]**2)
                    # print(norm)
                    if norm == 0:
                        u[i][j] = None
                        v[i][j] = None
                    else:
                        u[i][j] = berry_curv[0]/norm
                        v[i][j] = berry_curv[1]/norm
                else:
                    raise Exception("dirc must be 0, 1 or 2")     
        # plt.quiver(kx,ky,self._log_scale(u),self._log_scale(v), scale_units='xy')    
        # print(u, v)
        plt.quiver(kx+kx0, ky+ky0, u, v, scale_units='xy')    
        plt.title("band {}".format(str(band_indices)))