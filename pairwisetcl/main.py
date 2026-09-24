import numpy as np
import re
from pyscf.tools import molden
from pathlib import Path
from .output import *
from .tcltools import *

class GenerateCoordinates:
    
    '''
    Class for generating coordinates of the molecule and (optional) parameters for a random hydrogen spin bath.
    
    Args:
        coordinates (dict or str): from where to generate the coordinates of the molecule.
            Can be either a dictionary of number indexed atom types as keys and ndarrays of shape (3,) as the values
            or .xyz file
            or an ORCA .out or .log file
    
    Optional Method:
        set_bath_parameters: set the parameters for a random hydrogen spin bath 
            args: 
                density (float): density of random spin bath in cm^-3.
                box_length (int): edge length of cubic box.
                number_configurations (int): number of configurations of the random bath to average over
            optional args:
                center (ndarray with shape (3,)): position of center of box, usually taken to be the position of the electron.
                                                  If not supplied, set to the origin.
                atom_filter (float): filters the randomly placed particles such that none are closer to any molecular atom 
                                     than this value.
                                     Defaults to 2 angstroms if not supplied.
                spin_dens_filter (float): filters the randomly placed particles such thatn none are closer 
                                          to the electron position (center of spin density) than this value.
                                          Defaults to 7 angstroms if not supplied.
    
    Attributes:
        bath_density: defined in set_bath_parameters
        bath_box_length: defined in set_bath_parameters
        bath_number_configurations: defined in set_bath_parameters
        bath_center: defined in set_bath_parameters
        bath_max_spins (int): the maximum amount of randomly placed spins generated from the random bath function. 
                              after the filters are implemented, the actual number of spins for each configuration
                              will be less than this value.
        bath_atom_filter: defined in set_bath_parameters
        bath_spin_dens_filter: defined in set_bath_parameters
        molecule_atoms: a dictionary of indexed atom types as keys and cartesian coordinates in angstroms as values
        molecule_spins: a dictionary of indexed atoms and cartesian coordinates as values, only including hydrogen spins
                        found on the molecule
    '''
    
    def __init__(self,coordinates):
        
        self.bath_density = None 
        self.bath_box_length = None
        self.bath_number_configurations = None
        self.bath_center = None  
        self.bath_max_spins = None
        self.bath_atom_filter = None
        self.bath_spin_dens_filter = None
        
        if isinstance(coordinates, dict):
            self.molecule_atoms = coordinates
            
        elif isinstance(coordinates, str):
            
            file_extension = coordinates.split('.')[-1]
            if file_extension == 'xyz':
                self.atom_labels,atom_coordinates = get_atom_labels_coordinates_xyzfile(coordinates)
                self.molecule_atoms = {}
                atom_count = 0
                for key,val in zip(self.atom_labels,atom_coordinates):
                    self.molecule_atoms[f'{atom_count}{key}'] = val
                    atom_count += 1
            
            elif file_extension == 'log' or file_extension == 'out':
                self.atom_labels, atom_coordinates = get_atom_labels_coordinates_orca(coordinates)
                self.molecule_atoms = {}
                atom_count = 0
                for key,val in zip(self.atom_labels,atom_coordinates):
                    self.molecule_atoms[f'{atom_count}{key}'] = val
                    atom_count += 1
        
        else:
             raise ValueError('coordinates must be a dictionary with numbered atoms as keys and coordinates in angstrom as values\nor must be a string containing the path to a .xyz file or orca output file that contains the coordinates')
        
        self.molecule_spins = {}
        spin_active_count = 1 # count starts from one, eventually the electron spin will be indexed at 0
        for key,val in zip(list(self.molecule_atoms.keys()),list(self.molecule_atoms.values())):
            if re.findall(r'[a-zA-Z]+|\d+', key)[-1] == 'H':
                self.molecule_spins[f'{spin_active_count}H'] = val         
                spin_active_count += 1
        
    def set_bath_parameters(self,density,box_length,number_configurations,center=None,atom_filter=None,spin_dens_filter=None,seed=None): 
        
        self.bath_density = density
        self.bath_box_length = box_length
        self.bath_number_configurations = number_configurations 
        self.bath_center = center
        self.bath_atom_filter = atom_filter
        self.bath_spin_dens_filter = spin_dens_filter
        self.bath_seed = seed
        
        self.bath_max_spins = len(random_bath_generator(self.bath_box_length,self.molecule_atoms,atom_filter_distance=0.0,spin_dens_filter_distance=0.0,
                                     density=self.bath_density, density_units='cm-3',center=self.bath_center,seed=self.bath_seed))

class DephasingAnalysis:
    
    '''
    Class for performing the electron doublet spin Hahn-echo dephasing due to pairwise nuclear spin flip flops.
    
    Args:
        time_space (ndarray of shape (t,) where t is the number of time points): the time space to calculate the dephasing.
        parent (Class Object): the parent TclAnalysis class - included to ensure inheritance 
                               of important attributes of TclAnalysis
    
    Methods:
        analyze_molecule_pairs:
            args:
        analyze_solvent_pairs:
            args:
        get_total_dephasing:
            args:
    
    Attributes:
        time_space (ndarray of shape (t,)): defined in Args
        molecule_hfcs (dict): dictionary containing pre-computed hyperfine couplings, inherited from parent class
        coordinates (attribute): coordinates attribute inherited from parent class
        charge (int): charge of the molecule, inherited from parent class
        basis_set (dict): 
        
        molecule_alpha_map

    '''
    
    def __init__(self,time_space,parent):
        
        self.time_space = time_space
        self.molecule_hfcs = parent.molecule_hfcs 
        self.coordinates = parent.coordinates 
        self.charge = parent.charge
        self.basis_set = parent.basis_set
        self.spin_density = parent.spin_density
        self.molecule_alpha_map = None
        self.solvent_alpha_map = None
        self.molecule_solvent_alpha_map = None
        self.molecule_tcl2 = None
        self.solvent_tcl2 = None
    
    def e_n_n_point_dipole_dephasing(self,vk,vl,vk_indices,vl_indices,verbose=True,sort_alpha=False):
        
        if isinstance(self.coordinates.bath_center,np.ndarray):
            electron_coordinate = self.coordinates.bath_center
        else:
            if verbose:
                print('electron coordinates not supplied; setting to the origin')
            electron_coordinate = np.array([0,0,0])
        
        MU0 = np.pi*4e-7
        PLANCK = 6.626e-34
        GYRO_E = -17608.59705
        GYRO_H = 26.75221824
    
        # e-nk
        
        rek = electron_coordinate - vk
        mag_rek = np.linalg.norm(rek,axis=1,keepdims=True)
        
        ak = (1 - (3*rek[:,[2]]**2)/(mag_rek**2)) * (((MU0*GYRO_E*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rek**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
        
        # e-nl
        
        rel = electron_coordinate - vl
        mag_rel = np.linalg.norm(rel,axis=1,keepdims=True)
    
        al = (1 - (3*rel[:,[2]]**2)/(mag_rel**2)) * (((MU0*GYRO_E*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rel**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
    
        # nk-nl
    
        rkl = vk - vl
        mag_rkl = np.linalg.norm(rkl,axis=1,keepdims=True)
    
        bkl = (1 - (3*rkl[:,[2]]**2)/(mag_rkl**2)) * (((MU0*GYRO_H*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rkl**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
    
        # broadcasting over the prescribed time space for dynamics
        
        wkl_tcl2 = (4 * (bkl**2*(ak-al)**2) / (bkl**2 + (ak-al)**2)**2) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((ak-al)**2+bkl**2))**4
        wkl_tcl4 = (8 * (bkl**4*(ak-al)**4) / (bkl**2 + (ak-al)**2)**4) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((ak-al)**2+bkl**2))**8
    
        exp_tcl2 = np.exp(-np.sum(wkl_tcl2,axis=0))
        exp_tcl4 = np.exp(-np.sum(wkl_tcl2,axis=0)-3/2*np.sum(wkl_tcl4,axis=0))
        
        # broadcasting to get the alpha map
        
        alpha_map_size = int((1 + np.sqrt(1+8*len(vk)))/2)
        
        alpha = (4 * (bkl**2*(ak-al)**2) / (bkl**2 + (ak-al)**2)**2)
        
        alpha = alpha.squeeze()
        mag_rkl = mag_rkl.squeeze()
        mag_rek = mag_rek.squeeze()
        mag_rel = mag_rel.squeeze()
        
        if sort_alpha:

            # also sorting based on average e-n distance for n-n distance > 1 and < 3
            
            avg_mag_rekl = 1/2 * (mag_rek + mag_rel)
            
            avg_rekl_mask = (mag_rkl > 1.0) & (mag_rkl < 3.0)
            
            filtered_alpha = np.where(avg_rekl_mask,alpha,0.0)
            filtered_rkl = np.where(avg_rekl_mask,mag_rkl,0.0)
            filtered_rekl = np.where(avg_rekl_mask,avg_mag_rekl,np.inf) 
            
            filtered_sort_order = np.argsort(filtered_rekl)
            
            filtered_alpha_sorted = filtered_alpha[filtered_sort_order]
            filtered_rekl_sorted = filtered_rekl[filtered_sort_order]
            
            filtered_alpha_sorted_map = np.zeros([alpha_map_size,alpha_map_size])
            filtered_alpha_sorted_map[vk_indices,vl_indices] = filtered_alpha_sorted
            filtered_alpha_sorted_map[vl_indices,vk_indices] = filtered_alpha_sorted
            
            filtered_rekl_sorted_map = np.zeros([alpha_map_size,alpha_map_size])
            filtered_rekl_sorted_map[vk_indices,vl_indices] = filtered_rekl_sorted
            filtered_rekl_sorted_map[vl_indices,vk_indices] = filtered_rekl_sorted
            
            # now sorting based on rkl - unfiltered
            
            sort_order = np.argsort(mag_rkl)
            mag_rkl = mag_rkl[sort_order]
            alpha = alpha[sort_order]
            
            # recompute size of returnable matrix
            
        else:
            
            filtered_alpha_sorted_map = None
            filtered_rekl_sorted_map = None
       
        alpha_map = np.zeros([alpha_map_size,alpha_map_size])
        alpha_map[vk_indices,vl_indices] = alpha
        alpha_map[vl_indices,vk_indices] = alpha
        
        rkl_map = np.zeros([alpha_map_size,alpha_map_size])
        rkl_map[vk_indices,vl_indices] = mag_rkl
        rkl_map[vl_indices,vk_indices] = mag_rkl
        
    
        return exp_tcl2,exp_tcl4,alpha_map,rkl_map,filtered_alpha_sorted_map,filtered_rekl_sorted_map 
    
    def e_n_n_from_orca_dephasing(self,vk,vl,vk_indices,vl_indices,ak,al,sort_alpha=False):
        
        if isinstance(self.coordinates.bath_center,np.ndarray):
            electron_coordinate = self.coordinates.bath_center
        else:
            electron_coordinate = np.array([0,0,0])
        
        MU0 = np.pi*4e-7
        PLANCK = 6.626e-34
        GYRO_E = -17608.59705
        GYRO_H = 26.75221824
    
        # nk-nl
    
        rkl = vk - vl
        mag_rkl = np.linalg.norm(rkl,axis=1,keepdims=True)
    
        bkl = (1 - (3*rkl[:,[2]]**2)/(mag_rkl**2)) * (((MU0*GYRO_H*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rkl**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
    
        # broadcasting over the prescribed time space for dynamics
        
        wkl_tcl2 = (4 * (bkl**2*(ak-al)**2) / (bkl**2 + (ak-al)**2)**2) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((ak-al)**2+bkl**2))**4
        wkl_tcl4 = (8 * (bkl**4*(ak-al)**4) / (bkl**2 + (ak-al)**2)**4) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((ak-al)**2+bkl**2))**8
    
        exp_tcl2 = np.exp(-np.sum(wkl_tcl2,axis=0))
        exp_tcl4 = np.exp(-np.sum(wkl_tcl2,axis=0)-3/2*np.sum(wkl_tcl4,axis=0))
        
        # broadcasting to get the alpha map
        
        alpha = (4 * (bkl**2*(ak-al)**2) / (bkl**2 + (ak-al)**2)**2)
                        
        alpha_map_size = int((1 + np.sqrt(1+8*len(vk)))/2) 
        
        alpha = alpha.squeeze()
        bkl = bkl.squeeze()
            
        mag_rkl = mag_rkl.squeeze()
        mag_rel = np.linalg.norm(electron_coordinate-vl,axis=1,keepdims=True).squeeze()
        mag_rek = np.linalg.norm(electron_coordinate-vk,axis=1,keepdims=True).squeeze()
            
        if sort_alpha:
            
            # also sorting based on average e-n distance for n-n distance > 1 and < 3
            
            avg_mag_rekl = 1/2 * (mag_rek + mag_rel)
            
            avg_rekl_mask = (mag_rkl > 1.0) & (mag_rkl < 3.0)
            
            filtered_alpha = np.where(avg_rekl_mask,alpha,0.0)
            filtered_rkl = np.where(avg_rekl_mask,mag_rkl,0.0)
            filtered_rekl = np.where(avg_rekl_mask,avg_mag_rekl,np.inf) 
            
            filtered_sort_order = np.argsort(filtered_rekl)
            
            filtered_alpha_sorted = filtered_alpha[filtered_sort_order]
            filtered_rekl_sorted = filtered_rekl[filtered_sort_order]
            
            filtered_alpha_sorted_map = np.zeros([alpha_map_size,alpha_map_size])
            filtered_alpha_sorted_map[vk_indices,vl_indices] = filtered_alpha_sorted
            filtered_alpha_sorted_map[vl_indices,vk_indices] = filtered_alpha_sorted
            
            filtered_rekl_sorted_map = np.zeros([alpha_map_size,alpha_map_size])
            filtered_rekl_sorted_map[vk_indices,vl_indices] = filtered_rekl_sorted
            filtered_rekl_sorted_map[vl_indices,vk_indices] = filtered_rekl_sorted
                        
            # unfiltered sorting of rkl
            
            sort_order = np.argsort(mag_rkl)
            
            mag_rkl = mag_rkl[sort_order]
            alpha = alpha[sort_order]
            
        else:
            
            filtered_alpha_sorted_map = None
            filtered_rekl_sorted_map = None
            
        alpha_map = np.zeros([alpha_map_size,alpha_map_size])
        alpha_map[vk_indices,vl_indices] = alpha
        alpha_map[vl_indices,vk_indices] = alpha
        
        rkl_map = np.zeros([alpha_map_size,alpha_map_size])
        rkl_map[vk_indices,vl_indices] = mag_rkl
        rkl_map[vl_indices,vk_indices] = mag_rkl
    
        return exp_tcl2,exp_tcl4,alpha_map,rkl_map,filtered_alpha_sorted_map,filtered_rekl_sorted_map
    
    def e_n_n_molecule_solvent(self,vk,vl,ak=None,verbose=False,sort_alpha=False):
    
        # if ak != None, uses orca hfcs for the molecule spins
        # if ak == None, uses point dipole for everything
        
        if isinstance(self.coordinates.bath_center,np.ndarray):
            electron_coordinate = self.coordinates.bath_center
        else:
            if verbose:
                print('electron coordinates not supplied; setting to the origin')
            electron_coordinate = np.array([0,0,0])
        
        MU0 = np.pi*4e-7
        PLANCK = 6.626e-34
        GYRO_E = -17608.59705
        GYRO_H = 26.75221824
        
        # ak
        
        if isinstance(ak,dict):
            
            ak = np.array(list(ak.values()))[:,2,2]
            ak = ak.reshape(-1,1)
            
        else:
            
            rek = electron_coordinate - vk
            mag_rek = np.linalg.norm(rek,axis=1,keepdims=True)
    
            ak = (1 - (3*rek[:,[2]]**2)/(mag_rek**2)) * (((MU0*GYRO_E*GYRO_H*PLANCK**2) / 
                                 (16*np.pi**3*mag_rek**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
        
        rel = electron_coordinate - vl
        mag_rel = np.linalg.norm(rel,axis=1,keepdims=True)
    
        al = (1 - (3*rel[:,[2]]**2)/(mag_rel**2)) * (((MU0*GYRO_E*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rel**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
        
        
        # broadcast ak-al to get the correct shape
        
        delta_a = ak[:,None,:] - al[None,:,:]
        
        # bkl
        
        rkl = vk[:,None,:] - vl[None,:,:]
        
        mag_rkl = np.linalg.norm(rkl,axis=2,keepdims=True)
    
        bkl = (1 - (3*rkl[:,:,[2]]**2)/(mag_rkl**2)) * (((MU0*GYRO_H*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rkl**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
        
        # broadcast t to get the correct shape
        
        t = self.time_space[None,None,None,:]
        
        wkl_tcl2 = (4 * (bkl**2*(delta_a)**2) / (bkl**2 + (delta_a)**2)**2) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((delta_a)**2+bkl**2))**4
        wkl_tcl4 = (8 * (bkl**4*(delta_a)**4) / (bkl**2 + (delta_a)**2)**4) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((delta_a)**2+bkl**2))**8
    
        exp_tcl2 = np.exp(-np.sum(wkl_tcl2,axis=(0,1)))
        exp_tcl4 = np.exp(-np.sum(wkl_tcl2,axis=(0,1))-3/2*np.sum(wkl_tcl4,axis=(0,1)))
        
        # alpha map
        
        alpha = (4 * (bkl**2*(delta_a)**2) / (bkl**2 + (delta_a)**2)**2)
       
        alpha_map = np.squeeze(alpha,axis=2)
        bkl_map = np.squeeze(bkl,axis=2)
        rkl_map = np.squeeze(mag_rkl,axis=2)
    
        if sort_alpha:
            
            #sort_indices = np.argsort(-bkl_map,axis=1)
            sort_indices = np.argsort(rkl_map,axis=1)
            rkl_map = np.take_along_axis(rkl_map,sort_indices,axis=1)
            alpha_map = np.take_along_axis(alpha_map,sort_indices,axis=1)
        
        return exp_tcl2,exp_tcl4,alpha_map,rkl_map
    
    def e_n_n_molecule_solvent_spin_density(self,vk,vl,ak=None,al=None,verbose=False,sort_alpha=False):
    
        # if ak != None, uses orca hfcs for the molecule spins
        # if ak == None, uses point dipole for molecule spins
        # al=None is purely to retain the correct ordering - with this function it will always be the solvent hfcs computed from spin density
        
        if isinstance(self.coordinates.bath_center,np.ndarray):
            electron_coordinate = self.coordinates.bath_center
        else:
            if verbose:
                print('electron coordinates not supplied; setting to the origin')
            electron_coordinate = np.array([0,0,0])
        
        #if al == None:
        #    raise ValueError('al is returning None. This is a bug.')
        
        MU0 = np.pi*4e-7
        PLANCK = 6.626e-34
        GYRO_E = -17608.59705
        GYRO_H = 26.75221824
        
        # ak
        
        if isinstance(ak,dict):
            
            ak = np.array(list(ak.values()))[:,2,2]
            ak = ak.reshape(-1,1)
            
        else:
            
            rek = electron_coordinate - vk
            mag_rek = np.linalg.norm(rek,axis=1,keepdims=True)
    
            ak = (1 - (3*rek[:,[2]]**2)/(mag_rek**2)) * (((MU0*GYRO_E*GYRO_H*PLANCK**2) / 
                                 (16*np.pi**3*mag_rek**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
        
        al = al.reshape(-1,1)
        
        # broadcast ak-al to get the correct shape
        
        delta_a = ak[:,None,:] - al[None,:,:]
        
        # bkl
        
        rkl = vk[:,None,:] - vl[None,:,:]
        
        mag_rkl = np.linalg.norm(rkl,axis=2,keepdims=True)
    
        bkl = (1 - (3*rkl[:,:,[2]]**2)/(mag_rkl**2)) * (((MU0*GYRO_H*GYRO_H*PLANCK**2) / 
                             (16*np.pi**3*mag_rkl**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2)))
        
        # broadcast t to get the correct shape
        
        t = self.time_space[None,None,None,:]
        
        wkl_tcl2 = (4 * (bkl**2*(delta_a)**2) / (bkl**2 + (delta_a)**2)**2) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((delta_a)**2+bkl**2))**4
        wkl_tcl4 = (8 * (bkl**4*(delta_a)**4) / (bkl**2 + (delta_a)**2)**4) * np.sin(2*np.pi*self.time_space/4 * np.sqrt((delta_a)**2+bkl**2))**8
    
        exp_tcl2 = np.exp(-np.sum(wkl_tcl2,axis=(0,1)))
        exp_tcl4 = np.exp(-np.sum(wkl_tcl2,axis=(0,1))-3/2*np.sum(wkl_tcl4,axis=(0,1)))
        
        # alpha map
        
        alpha = (4 * (bkl**2*(delta_a)**2) / (bkl**2 + (delta_a)**2)**2)
        
        alpha_map = np.squeeze(alpha,axis=2)
        bkl_map = np.squeeze(bkl,axis=2)
        rkl_map = np.squeeze(mag_rkl,axis=2)
    
        if sort_alpha:
            
            #sort_indices = np.argsort(-bkl_map,axis=1)
            sort_indices = np.argsort(rkl_map,axis=1)
            rkl_map = np.take_along_axis(rkl_map,sort_indices,axis=1)
            
            alpha_map = np.take_along_axis(alpha_map,sort_indices,axis=1)
        
        return exp_tcl2,exp_tcl4,alpha_map,rkl_map
    
    def analyze_molecule_pairs(self):
        
        vectors = np.array(list(self.coordinates.molecule_spins.values()))
        vk_indices,vl_indices = np.triu_indices(len(vectors),k=1)
        vk = vectors[vk_indices]
        vl = vectors[vl_indices]
        
        if self.molecule_hfcs == None:
            
            self.molecule_tcl2,self.molecule_tcl4,self.molecule_alpha_map,self.molecule_rkl_map,self.molecule_filtered_alpha_map,self.molecule_filtered_rekl_map = self.e_n_n_point_dipole_dephasing(vk,vl,vk_indices,vl_indices)
            
        else:
                        
            hfcs = np.array(list(self.molecule_hfcs.values()))[:,2,2]
            ak_indices,al_indices = np.triu_indices(len(hfcs),k=1)
            ak = hfcs[ak_indices]
            al = hfcs[al_indices]
            
            ak = ak.reshape(-1,1)
            al = al.reshape(-1,1)
            
            self.molecule_tcl2,self.molecule_tcl4,self.molecule_alpha_map,self.molecule_rkl_map,self.molecule_filtered_alpha_map,self.molecule_filtered_rekl_map = self.e_n_n_from_orca_dephasing(vk,vl,vk_indices,vl_indices,ak,al)
       
    def analyze_solvent_pairs(self,verbose=False):
    
        if None in (self.coordinates.bath_density,self.coordinates.bath_box_length,self.coordinates.bath_number_configurations):
            raise ValueError('random bath parameters not set. If you want to include a random bath,\nuse the coordinates.set_bath_parameters(bath_density,bath_box_length,bath_number_configurations) method.\n\nAlternatively, just use analyze_molecule_pairs() to look at only the molecule.')
        
        if not isinstance(self.molecule_alpha_map,np.ndarray):
            print('molecule spin analysis not found')
            self.analyze_molecule_pairs()
            print('molecule spin analysis done. moving onto solvent analysis')
        
        self.solvent_alpha_map = np.zeros([self.coordinates.bath_max_spins,self.coordinates.bath_max_spins])
        self.solvent_rkl_map = np.zeros([self.coordinates.bath_max_spins,self.coordinates.bath_max_spins])
        self.solvent_filtered_alpha_map = np.zeros([self.coordinates.bath_max_spins,self.coordinates.bath_max_spins])
        self.solvent_filtered_rekl_map = np.zeros([self.coordinates.bath_max_spins,self.coordinates.bath_max_spins])
        
        self.molecule_solvent_alpha_map = np.zeros([len(self.coordinates.molecule_spins),self.coordinates.bath_max_spins])
        self.molecule_solvent_rkl_map = np.zeros([len(self.coordinates.molecule_spins),self.coordinates.bath_max_spins])
            
        self.solvent_tcl2 = np.zeros([len(self.time_space)])
        self.molecule_solvent_tcl2 = np.zeros([len(self.time_space)])
        self.solvent_tcl4 = np.zeros([len(self.time_space)])
        self.molecule_solvent_tcl4 = np.zeros([len(self.time_space)])
        if verbose:
            print('averaging over configurations of the random bath\n')
            
        for n in range(self.coordinates.bath_number_configurations):
                
            random_bath = random_bath_generator(self.coordinates.bath_box_length,self.coordinates.molecule_atoms,atom_filter_distance=self.coordinates.bath_atom_filter,spin_dens_filter_distance=self.coordinates.bath_spin_dens_filter,
                                 density=self.coordinates.bath_density, density_units='cm-3',center=self.coordinates.bath_center,seed=self.coordinates.bath_seed)
                        
            if len(random_bath) == 0:
                self.solvent_tcl2 += 1
                self.solvent_tcl4 += 1
                self.molecule_solvent_tcl2 += 1
                self.molecule_solvent_tcl4 += 1
                continue
            
            vk_indices,vl_indices = np.triu_indices(len(random_bath),k=1)
            vk = random_bath[vk_indices]
            vl = random_bath[vl_indices]
            
            if not isinstance(self.spin_density,np.ndarray):
                
                solvent_tcl2,solvent_tcl4,solvent_alpha_map,solvent_rkl_map,filtered_alpha_map,filtered_rekl_map = self.e_n_n_point_dipole_dephasing(vk,vl,vk_indices,vl_indices,verbose=False,sort_alpha=True) 
                
                self.solvent_tcl2 += solvent_tcl2
                self.solvent_tcl4 += solvent_tcl4
                
                self.solvent_alpha_map[:(solvent_alpha_map.shape[0]),:(solvent_alpha_map.shape[1])] += solvent_alpha_map
                self.solvent_rkl_map[:(solvent_rkl_map.shape[0]),:(solvent_rkl_map.shape[1])] += solvent_rkl_map
                self.solvent_filtered_alpha_map[:(filtered_alpha_map.shape[0]),:(filtered_alpha_map.shape[1])] += filtered_alpha_map
                self.solvent_filtered_rekl_map[:(filtered_rekl_map.shape[0]),:(filtered_rekl_map.shape[1])] += filtered_rekl_map
                
                molecule_solvent_tcl2,molecule_solvent_tcl4,molecule_solvent_alpha_map,molecule_solvent_rkl_map = self.e_n_n_molecule_solvent(np.array(list(self.coordinates.molecule_spins.values())),random_bath,ak=self.molecule_hfcs,verbose=False,sort_alpha=True) 
                
                self.molecule_solvent_tcl2 += molecule_solvent_tcl2
                self.molecule_solvent_tcl4 += molecule_solvent_tcl4
                
                self.molecule_solvent_alpha_map[:(molecule_solvent_alpha_map.shape[0]),:(molecule_solvent_alpha_map.shape[1])] += molecule_solvent_alpha_map
                self.molecule_solvent_rkl_map[:(molecule_solvent_rkl_map.shape[0]),:(molecule_solvent_rkl_map.shape[1])] += molecule_solvent_rkl_map
                
                
            else:
                
                solvent_hfcs = hyperfines_from_spin_density(self.spin_density, self.basis_set, random_bath, (np.array(list(self.coordinates.molecule_atoms.values())), self.coordinates.atom_labels, self.charge))
                solvent_hfcs = solvent_hfcs[:,2,2]
                
                ak_indices,al_indices = np.triu_indices(len(solvent_hfcs),k=1)
                ak = solvent_hfcs[ak_indices]
                al = solvent_hfcs[al_indices]
            
                ak = ak.reshape(-1,1)
                al = al.reshape(-1,1)
                
                solvent_tcl2,solvent_tcl4,solvent_alpha_map,solvent_rkl_map,filtered_alpha_map,filtered_rekl_map = self.e_n_n_from_orca_dephasing(vk,vl,vk_indices,vl_indices,ak,al,sort_alpha=True)
                
                self.solvent_tcl2 += solvent_tcl2
                self.solvent_tcl4 += solvent_tcl4
                
                self.solvent_alpha_map[:(solvent_alpha_map.shape[0]),:(solvent_alpha_map.shape[1])] += solvent_alpha_map
                self.solvent_rkl_map[:(solvent_rkl_map.shape[0]),:(solvent_rkl_map.shape[1])] += solvent_rkl_map
                self.solvent_filtered_alpha_map[:(filtered_alpha_map.shape[0]),:(filtered_alpha_map.shape[1])] += filtered_alpha_map
                self.solvent_filtered_rekl_map[:(filtered_rekl_map.shape[0]),:(filtered_rekl_map.shape[1])] += filtered_rekl_map
                
                molecule_solvent_tcl2,molecule_solvent_tcl4,molecule_solvent_alpha_map,molecule_solvent_rkl_map = self.e_n_n_molecule_solvent_spin_density(np.array(list(self.coordinates.molecule_spins.values())),random_bath,ak=self.molecule_hfcs,al=solvent_hfcs,verbose=False,sort_alpha=True)
                
                self.molecule_solvent_tcl2 += molecule_solvent_tcl2
                self.molecule_solvent_tcl4 += molecule_solvent_tcl4
                
                self.molecule_solvent_alpha_map[:(molecule_solvent_alpha_map.shape[0]),:(molecule_solvent_alpha_map.shape[1])] += molecule_solvent_alpha_map
                self.molecule_solvent_rkl_map[:(molecule_solvent_rkl_map.shape[0]),:(molecule_solvent_rkl_map.shape[1])] += molecule_solvent_rkl_map
            
            if verbose:
                print(f'{n+1} out of {self.coordinates.bath_number_configurations} done')
            
        self.solvent_tcl2 /= self.coordinates.bath_number_configurations
        self.solvent_tcl4 /= self.coordinates.bath_number_configurations
        self.molecule_solvent_tcl2 /= self.coordinates.bath_number_configurations
        self.molecule_solvent_tcl4 /= self.coordinates.bath_number_configurations
        
        self.solvent_alpha_map /= self.coordinates.bath_number_configurations
        self.solvent_rkl_map /= self.coordinates.bath_number_configurations
        self.molecule_solvent_alpha_map /= self.coordinates.bath_number_configurations
        self.molecule_solvent_rkl_map /= self.coordinates.bath_number_configurations
        self.solvent_filtered_alpha_map /= self.coordinates.bath_number_configurations
        self.solvent_filtered_rekl_map /= self.coordinates.bath_number_configurations
    
    def get_total_dephasing(self,verbose=False):
        
        if not isinstance(self.molecule_tcl2,np.ndarray):
            self.analyze_molecule_pairs()
            self.analyze_solvent_pairs(verbose=verbose)
            
        elif isinstance(self.molecule_tcl2,np.ndarray) and not isinstance(self.solvent_tcl2,np.ndarray):
            self.analyze_solvent_pairs(verbose=verbose)
        
        self.total_tcl2 = self.molecule_tcl2 * self.solvent_tcl2 * self.molecule_solvent_tcl2
        self.total_tcl4 = self.molecule_tcl4 * self.solvent_tcl4 * self.molecule_solvent_tcl4 
        
class TclAnalysis: 
    
    '''
    Class for initiating a TCL2 and TCL4 analysis of electron doublet spin Hahn-echo dephasing due to pairwise
    nuclear spin flip flops.
      
    Methods:
        get_coordinates:
            args:
                coordinates: see GenerateCoordinates
        get_orca_hfcs:
            args:
                output_file: output file generated from ORCA containing computed HFCs for the hydrogen spins on 
                             the molecule. If not used, all couplings will be calculated from the point dipole
        get_orca_spin_density: TO DO
        get_orca_basis_set: TO DO
        get_dephasing_analysis: TO DO
    
    Attributes:
        TO DO
    '''
    
        
    def __init__(self):
        
        self.coordinates = None
        self.molecule_hfcs = None
        self.basis_set = None
        self.spin_density = None
        self.charge = None
    
    def get_coordinates(self,coordinates):
    
        self.coordinates = GenerateCoordinates(coordinates)
        
    def get_orca_hfcs(self,output_file):
        
        '''
        parses ORCA output file to get computed hyperfine tensors.
        only looks for hydrogen hyperfine tensors, since that's all our theory is good for right now
        '''
    
        self.molecule_hfcs = get_hyperfine_from_orca_output(output_file)        
        self.molecule_hfcs = {f"{i+1}H": value for i, (key, value) in enumerate(self.molecule_hfcs.items())}
        
    def get_orca_spin_density(self,output_file):
               
        with open(output_file,'r') as f:
            
            input_file_flag = False
            casscf_flag = False
            spindensity_flag = False
            
            for line in f:
                if 'INPUT FILE' in line:
                    input_file_flag = True
                    continue
                elif 'END OF INPUT' in line:
                    input_file_flag = False
                    continue
                elif input_file_flag:
                    if 'casscf' in line:
                        casscf_flag = True
                        spindensity_flag = True # CAS automatically prints the spin density                        
                        
                    if 'P_SpinDensity' in line:
                        spindensity_flag = True
                    if 'xyzfile' in line:
                        self.charge = int(line.split()[4])
        
        if not spindensity_flag:
            raise RuntimeError('Spin density matrix not printed in output file.\nEnsure verbosity is at the correct level in ORCA.\nIf you are running a single-reference calculation, include\n%output\nPrint[P_SpinDensity] 1\nend\nin your input file.')
            
        self.spin_density = spin_density_from_orca_output(output_file,casscf=casscf_flag)        
                
        if casscf_flag: 
         
            directory = '/'.join(output_file.split('/')[:-1]) + '/'
            basename = output_file.split('/')[-1].split('.')[0]
        
            if not any(Path(directory).glob(f'{basename}.molden')):
                raise RuntimeError('.molden file not found. As implemented, using a casscf spin density also requires a .molden file to be in the directory with the .log file.')
    
            self.spin_density,active_indices = self.spin_density
            
            _,_,mo_coeff,_,_,_ = molden.load(f'{directory}' + '/' + f'{basename}' + '.molden')
            
            active_mo_coeffs = mo_coeff[:,active_indices[0]:(active_indices[1]+1)]
            self.spin_density = active_mo_coeffs @ self.spin_density @ active_mo_coeffs.T
            
            self.spin_density = transform_spin_density(self.spin_density,self.basis_set,self.coordinates.atom_labels)
        
        else:
            
            self.spin_density = self.spin_density[0]
            self.spin_density = transform_spin_density(self.spin_density,self.basis_set,self.coordinates.atom_labels)    
    
    def get_orca_basis_set(self,output_file):
                
        with open(output_file,'r') as f:
            
            input_file_flag = False
            printbasis_found = False
            auxiliary_basis = True # right now this is hard coded as True since there's some issue with the ORCA logic in determining if it's there
            
            for line in f:
                if 'INPUT FILE' in line:
                    input_file_flag = True
                    continue
                elif 'END OF INPUT' in line:
                    input_file_flag = False
                    continue
                elif input_file_flag:
                    if '! PrintBasis' in line:
                        printbasis_found = True
                    if 'rijcosx' in line or 'RI' in line or '/C' in line or '/J' in line or 'RIJCOSX' in line:
                        auxiliary_basis = True
        
        if not printbasis_found:
            raise RuntimeError('Basis set information not found in ORCA output file.\nInclude ! PrintBasis in the header of your input file to ORCA to generate the basis set information.')
        
        basis_set_string = get_basis_set_string(output_file, auxiliary_basis=auxiliary_basis)
        self.basis_set = parse_basis_set_string(basis_set_string)
                
    def get_dephasing_analysis(self,time_space):
        
        self.dephasing_analysis = DephasingAnalysis(time_space,self)

if __name__ == "__main__":
    test = TclAnalysis()
    test.get_coordinates({'0H':np.array([3.9,0,0.8]),'1H':np.array([4.1,0,-0.8])})
    time_space = np.linspace(0,0.02,100)
    test.get_dephasing_analysis(time_space)
    test.dephasing_analysis.analyze_molecule_pairs()
