import numpy as np
import re
from pyscf.tools import molden
from pathlib import Path
from .output import *
from .tcltools import *

class GenerateCoordinates:
    
    '''
    This needs to have a generator for the random bath parameters... 
    which will be averaged over in the pairwise and dephasing module - not sure of the best way to go about this
    methods could be added to interface with diferent electronic structure outputs... thinking of Gaussian
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
        
    def set_bath_parameters(self,density,box_length,number_configurations,center=None,atom_filter=None,spin_dens_filter=None): 
        
        '''
        This function needs to generate the couplings averaged over all the number of configurations
        and can be inherited for dephasing and pairwise analysis
        So the output ought to be m-s and s-s tcl contributions and
        '''
        
        self.bath_density = density
        self.bath_box_length = box_length
        self.bath_number_configurations = number_configurations 
        self.bath_center = center
        self.bath_atom_filter = atom_filter
        self.bath_spin_dens_filter = spin_dens_filter
        
        self.bath_max_spins = len(random_bath_generator(self.bath_box_length,self.molecule_atoms,atom_filter_distance=0.0,spin_dens_filter_distance=0.0,
                                     density=self.bath_density, density_units='cm-3',center=self.bath_center))
            
    # Could add a method to calculate density on the fly given a solvent molecule, temperature, mixture, etc

class DephasingAnalysis: # this may need to be explicitly a child class, I'll find out as I go
    
    '''
    Calculate pairwise analysis for the entire system
    Couplings should be calculated on the fly
    Should have a way of calculating couplings on the fly if they are not prefigured from ORCA or something else
    How to handle spin densities, hfcs, etc?
    pair analysis is calculated always at the 2nd order tcl level, but dynamics can be calculated at 2nd or 4th order
    2nd order will be default, to add 4th order will be a method call
    '''
    
    def __init__(self,time_space,parent):
        
        self.time_space = time_space
        self.molecule_hfcs = parent.molecule_hfcs # this could be None if there aren't any being imported from ORCA
        self.coordinates = parent.coordinates # this inherits the bath parameters generated with the Coordinates class
        self.charge = parent.charge
        self.basis_set = parent.basis_set
        self.spin_density = parent.spin_density
        self.molecule_alpha_map = None
        self.solvent_alpha_map = None
        self.molecule_solvent_alpha_map = None
        self.molecule_tcl2 = None
        self.solvent_tcl2 = None
    
    def e_n_n_point_dipole_dephasing(self,vk,vl,vk_indices,vl_indices,verbose=True):
        
        # this function can be used for the solvent stuff too
        
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
        
        alpha_map = np.zeros([alpha_map_size,alpha_map_size])
        alpha_map[vk_indices,vl_indices] = alpha.squeeze()
        alpha_map[vl_indices,vk_indices] = alpha.squeeze()
    
        return exp_tcl2,exp_tcl4,alpha_map
    
    def e_n_n_from_orca_dephasing(self,vk,vl,vk_indices,vl_indices,ak,al):
        
        # this function can be used for the solvent stuff too
        
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
            
        alpha_map = np.zeros([alpha_map_size,alpha_map_size])
        alpha_map[vk_indices,vl_indices] = alpha.squeeze()
        alpha_map[vl_indices,vk_indices] = alpha.squeeze()
    
        return exp_tcl2,exp_tcl4,alpha_map
    
    def e_n_n_molecule_solvent(self,vk,vl,ak=None,verbose=False):
    
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
        
        return exp_tcl2,exp_tcl4,alpha_map
    
    def e_n_n_molecule_solvent_spin_density(self,vk,vl,ak=None,al=None,verbose=False):
    
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
        
        return exp_tcl2,exp_tcl4,alpha_map
    
    def analyze_molecule_pairs(self):
        
        '''
        attempting a rewrite for this function to be faster
        not bothering to store imap, it's not a good idea
        '''
        
        # don't need to initialize either of these objects here
        
        #self.molecule_alpha_map = np.zeros([len(self.coordinates.molecule_spins),len(self.coordinates.molecule_spins)])
        #self.molecule_dephasing = np.zeros([2,len(self.time_space)])
            
        # self.get_magnetic_couplings() - this should be used somewhere in the numpy vectorization
        # it is a method of this class since the class will have information about hfcs that can be used
        
        vectors = np.array(list(self.coordinates.molecule_spins.values()))
        vk_indices,vl_indices = np.triu_indices(len(vectors),k=1)
        vk = vectors[vk_indices]
        vl = vectors[vl_indices]
        
        if self.molecule_hfcs == None:
            
            self.molecule_tcl2,self.molecule_tcl4,self.molecule_alpha_map = self.e_n_n_point_dipole_dephasing(vk,vl,vk_indices,vl_indices)
            
        else:
                        
            hfcs = np.array(list(self.molecule_hfcs.values()))[:,2,2]
            ak_indices,al_indices = np.triu_indices(len(hfcs),k=1)
            ak = hfcs[ak_indices]
            al = hfcs[al_indices]
            
            ak = ak.reshape(-1,1)
            al = al.reshape(-1,1)
            
            self.molecule_tcl2,self.molecule_tcl4,self.molecule_alpha_map = self.e_n_n_from_orca_dephasing(vk,vl,vk_indices,vl_indices,ak,al)
       
    def analyze_solvent_pairs(self,verbose=False):
    
        # this function will need to interface with the orca spin densities
        # should have different behavior depending on if self.spin_density and self.basis_set are None
     
        if None in (self.coordinates.bath_density,self.coordinates.bath_box_length,self.coordinates.bath_number_configurations):
            raise ValueError('random bath parameters not set. If you want to include a random bath,\nuse the coordinates.set_bath_parameters(bath_density,bath_box_length,bath_number_configurations) method.\n\nAlternatively, just use analyze_molecule_pairs() to look at only the molecule.')
        
        if not isinstance(self.molecule_alpha_map,np.ndarray):
            print('molecule spin analysis not found')
            self.analyze_molecule_pairs()
            print('molecule spin analysis done. moving onto solvent analysis')
        
        # will need to think of a clever way to do the averaging
        # for now the logic simply needs to be compute with point dipole or with spin densities
        # but that should be further in the logic so the code is more compact
            
        # will need to initialize some stuff I think, probably alpha map and dynamics arrays
            
        self.solvent_alpha_map = np.zeros([self.coordinates.bath_max_spins,self.coordinates.bath_max_spins])
        self.molecule_solvent_alpha_map = np.zeros([len(self.coordinates.molecule_spins),self.coordinates.bath_max_spins])
            
        self.solvent_tcl2 = np.zeros([len(self.time_space)])
        self.molecule_solvent_tcl2 = np.zeros([len(self.time_space)])
        self.solvent_tcl4 = np.zeros([len(self.time_space)])
        self.molecule_solvent_tcl4 = np.zeros([len(self.time_space)])
        if verbose:
            print('averaging over configurations of the random bath\n')
            
        for n in range(self.coordinates.bath_number_configurations):
                
            random_bath = random_bath_generator(self.coordinates.bath_box_length,self.coordinates.molecule_atoms,atom_filter_distance=self.coordinates.bath_atom_filter,spin_dens_filter_distance=self.coordinates.bath_spin_dens_filter,
                                 density=self.coordinates.bath_density, density_units='cm-3',center=self.coordinates.bath_center)
            
            # random_bath has a shape (N,3) where N is the number of random bath spins. 
            # N might be different for each iteration since they are randomly placed
            # and then filtered by distance from the molecular atoms  
            
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
                
                solvent_tcl2,solvent_tcl4,solvent_alpha_map = self.e_n_n_point_dipole_dephasing(vk,vl,vk_indices,vl_indices,verbose=False)
                
                self.solvent_tcl2 += solvent_tcl2
                self.solvent_tcl4 += solvent_tcl4
                
                # need to add in the m-s pair logic - needs different calls
                
                molecule_solvent_tcl2,molecule_solvent_tcl4,molecule_solvent_alpha_map = self.e_n_n_molecule_solvent(np.array(list(self.coordinates.molecule_spins.values())),random_bath,ak=self.molecule_hfcs,verbose=False) # check use of self.molecule_hfcs here
                
                # needs to know if orca, pdip, or spin density for the molecule part of molecule-solvent pairs
                
                self.molecule_solvent_tcl2 += molecule_solvent_tcl2
                self.molecule_solvent_tcl4 += molecule_solvent_tcl4
                
                # need something here probably to sort the alpha_map...
                
            else:
                
                solvent_hfcs = hyperfines_from_spin_density(self.spin_density, self.basis_set, random_bath, (np.array(list(self.coordinates.molecule_atoms.values())), self.coordinates.atom_labels, self.charge))
                solvent_hfcs = solvent_hfcs[:,2,2]
                
                ak_indices,al_indices = np.triu_indices(len(solvent_hfcs),k=1)
                ak = solvent_hfcs[ak_indices]
                al = solvent_hfcs[al_indices]
            
                ak = ak.reshape(-1,1)
                al = al.reshape(-1,1)
                
                solvent_tcl2,solvent_tcl4,solvent_alpha_map = self.e_n_n_from_orca_dephasing(vk,vl,vk_indices,vl_indices,ak,al)
                
                self.solvent_tcl2 += solvent_tcl2
                self.solvent_tcl4 += solvent_tcl4
                
                # need molecule-solvent logic now... then it's all done
                
                molecule_solvent_tcl2,molecule_solvent_tcl4,molecule_solvent_alpha_map = self.e_n_n_molecule_solvent_spin_density(np.array(list(self.coordinates.molecule_spins.values())),random_bath,ak=self.molecule_hfcs,al=solvent_hfcs,verbose=False)
                
                self.molecule_solvent_tcl2 += molecule_solvent_tcl2
                self.molecule_solvent_tcl4 += molecule_solvent_tcl4
            
            if verbose:
                print(f'{n+1} out of {self.coordinates.bath_number_configurations} done')
            
        self.solvent_tcl2 /= self.coordinates.bath_number_configurations
        self.solvent_tcl4 /= self.coordinates.bath_number_configurations
        self.molecule_solvent_tcl2 /= self.coordinates.bath_number_configurations
        self.molecule_solvent_tcl4 /= self.coordinates.bath_number_configurations
    
    def get_total_dephasing(self,verbose=False):
        
        if not isinstance(self.molecule_tcl2,np.ndarray):
            self.analyze_molecule_pairs()
            self.analyze_solvent_pairs(verbose=verbose)
            
        elif isinstance(self.molecule_tcl2,np.ndarray) and not isinstance(self.solvent_tcl2,np.ndarray):
            self.analyze_solvent_pairs(verbose=verbose)
        
        self.total_tcl2 = self.molecule_tcl2 * self.solvent_tcl2 * self.molecule_solvent_tcl2
        self.total_tcl4 = self.molecule_tcl4 * self.solvent_tcl4 * self.molecule_solvent_tcl4 
        
#
#
### Parent Class
#
#

class TclAnalysis: 
    
    '''
    parent class - contains everything needed to calculate dynamics and do the pairwise analysis
    
    should contain
    coordinates
    spin density
    hfcs
    basis set information
    
    if the electronic structure parameters are not present in the parent class, should just do everything from point dipole
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
        
        # look through input file to determine if it is CASSCF
        # look through input file to make sure the
        # %output
        # Print[P_SpinDensity] 1
        # end
        # section is there, ONLY if it's not CAS. if it's CAS, it'll automatically print the MO spin density
        
        # RuntimeError should be implemented here if self.basis_set is None. Will need it for the transformed spin density
        
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
        
        # code to transform spin density to correct format
        
        # need a molden file for casscf - there is probably a way around this, but it's the only way I've done this so far
        
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
            
            #self.charge and mo_coeff are both accessible here
        
        else:
            
            self.spin_density = self.spin_density[0]
            self.spin_density = transform_spin_density(self.spin_density,self.basis_set,self.coordinates.atom_labels)    
    
    def get_orca_basis_set(self,output_file):
        
        # look at input file to determine if ! PrintBasis is in the header
        # also determine if an auxiliary basis was used - not going to be too thorough with this, 
        # ORCA has too many ways to initiate density fitting
        
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
