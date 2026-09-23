import numpy as np
import re
from pyscf import gto

def transform_spin_density(spin_density,basis_set,atom_list):
    
    '''
    Transforms spin density from ORCA AO ordering to PySCF AO ordering
    assumes first ao is an s-like orbital, which should be a safe assumption...
    '''
    
    transformation_sub_matrices = {0:np.array([[1]]),
                                   1:np.array([[0,1,0],
                                               [0,0,1],
                                               [1,0,0]]),
                                   2:np.array([[0,0,0,0,1],
                                               [0,0,1,0,0],
                                               [1,0,0,0,0],
                                               [0,1,0,0,0],
                                               [0,0,0,1,0]]),
                                   3:np.array([[0,0,0,0,0,0,1],
                                               [0,0,0,0,1,0,0],
                                               [0,0,1,0,0,0,0],
                                               [1,0,0,0,0,0,0],
                                               [0,1,0,0,0,0,0],
                                               [0,0,0,1,0,0,0],
                                               [0,0,0,0,0,1,0]]),
                                   4:np.array([[0,0,0,0,0,0,0,0,1],
                                               [0,0,0,0,0,0,1,0,0],
                                               [0,0,0,0,1,0,0,0,0],
                                               [0,0,1,0,0,0,0,0,0],
                                               [1,0,0,0,0,0,0,0,0],
                                               [0,1,0,0,0,0,0,0,0],
                                               [0,0,0,1,0,0,0,0,0],
                                               [0,0,0,0,0,1,0,0,0],
                                               [0,0,0,0,0,0,0,1,0]])}
    
    transformation_matrix = np.array([])
    
    for atom in atom_list:
        for basis in basis_set[atom]:
            angular_momentum = basis[0]
            if transformation_matrix.size == 0:
                transformation_matrix = np.array([[1]])
            else:
                transformation_matrix = np.pad(transformation_matrix,((0,2*angular_momentum+1),(0,2*angular_momentum+1)),mode='constant',constant_values=0)
                transformation_matrix[-(2*angular_momentum+1):,-(2*angular_momentum+1):] = transformation_sub_matrices[angular_momentum]
                
    return transformation_matrix @ spin_density @ transformation_matrix.T

def point_dipolar_coupling(type_1,type_2,vector_1,vector_2):
    
    # constants
    
    MU0 = np.pi*4e-7
    PLANCK = 6.626e-34
    GYRO_E = -17608.59705
    GYRO_H = 26.75221824
    
    type_dict = {'e':GYRO_E,'H':GYRO_H}
    
    # coupling calculation
    
    r12 = vector_1 - vector_2
    magnitude_r12 = np.linalg.norm(vector_1-vector_2)
    
    coupling_constant = ((MU0*type_dict[type_1]*type_dict[type_2]*PLANCK**2) / 
                         (16*np.pi**3*magnitude_r12**3))*(1000/(PLANCK*(1e-10)**3*(1e-4)**2))

    anisotropy = 1 - (3*r12[2]**2)/(magnitude_r12**2)
    
    return coupling_constant * anisotropy
    
def random_bath_generator(size, molecule_atoms, atom_filter_distance=None, spin_dens_filter_distance=None, number=1000, density=None, types=None, density_units="cm-3", center=None, seed=None):
    
    if atom_filter_distance is None:
        atom_filter_distance = 2.0
    else:
        atom_filter_distance = atom_filter_distance
        
    if spin_dens_filter_distance is None:
        spin_dens_filter_distance = 7.0
    else:
        spin_dens_filter_distance = spin_dens_filter_distance
    
    _remove_digits = str.maketrans("", "", "+-^1234567890")
    
    size = np.asarray(size)
    unit_conversion = {"a": 1, "cm": 1e-8, "m": 1e-10}

    if size.size == 1:
        size = np.array([size, size, size])

    elif size.size > 3:
        raise RuntimeError("Wrong size format")

    if center is None:
        center = size / 2

    if density is not None:
        du = density_units.lower().translate(_remove_digits)
        power = re.findall(r"\d+", density_units.lower())

        sc = np.count_nonzero(size != 0)

        if power:
            powa = int(power[0])
            if sc != powa:
                warnings.warn(f"size dimensions {sc} do not agree with density units {density_units}", stacklevel=2)
        else:
            powa = sc

        density = np.asarray(density) * unit_conversion[du] ** powa

        number = np.rint(density * np.prod(size[size != 0])).astype(np.int32)

    else:
        number = np.asarray(number, dtype=np.int32)

    total_number = np.sum(number)

    # Generate the coordinates
    generator = np.random.default_rng(seed=seed)

    bath_spins = generator.random((total_number,3)) * size - center # bath centering
    
    # need to do the filter here to ensure nothing is too close to the molecular atoms or the center of spin density
        
    distance_to_center_spin_dens = np.linalg.norm((bath_spins - center),axis=1)
    too_close_spin_dens_mask = distance_to_center_spin_dens <= spin_dens_filter_distance
    
    dif_vec_bath_molecule_spins = bath_spins[:,None,:] - np.array(list(molecule_atoms.values()))[None,:,:]
    distance_bath_molecule_spins = np.linalg.norm(dif_vec_bath_molecule_spins,axis=2)
    too_close_atoms_mask = np.any(distance_bath_molecule_spins < atom_filter_distance, axis=1)
    
    to_remove = too_close_spin_dens_mask | too_close_atoms_mask
    to_keep = ~to_remove
    
    filtered_bath_spins = bath_spins[to_keep]
    
    return filtered_bath_spins

def hyperfines_from_spin_density(spin_density,basis_set,nuclear_positions,mol_params):
    
    '''
    spin_density comes from an ORCA calculation, transformed into PySCF notation using transform_spin_density
    basis_set is the raw basis set from an ORCA calculation
    nuclear_positions is an array of arrays containing the Cartesian coordinates of hydrogens
    mol_params contains the atoms, elements, and charge of the system under consideration
    '''
    
    # arguments - self.spin_density, self.basis_set, random_bath, (np.array(list(self.coordinates.molecule_atoms.values())), self.coordinates.atom_labels, self.charge)
    
    mol_coordinates = mol_params[0] # np.array(list(self.coordinates.molecule_atoms.values()))
    mol_types = mol_params[1] # self.coordinates.atom_labels
    mol_charge = mol_params[2] # self.charge
    hyperfines = np.zeros([len(nuclear_positions),3,3],dtype=np.complex128)
    
    for number,n in enumerate(nuclear_positions): # this could probably be faster than explicitly looping
        
        mol_extended = gto.Mole()
        
        new_coords = np.concatenate((mol_coordinates,np.array([n])),axis=0)
        new_elements = mol_types.copy()
        new_elements.append('H')
        
        mol_extended.atom = [[element,coord] for element,coord in zip(new_elements,new_coords)]
        mol_extended.charge = mol_charge
        mol_extended.spin = 0 # this may be the issue?
        mol_extended.unit = 'angstrom'
        mol_extended.basis = basis_set
        
        mol_extended.build()
        
        with mol_extended.with_rinv_origin(mol_extended.atom_coord(len(mol_types))): # code snippet from PySCF to calculate hyperfine fermi contact and spin-dipole using atomic orbital integrals
            ipipv = mol_extended.intor('int1e_ipiprinv', 9).reshape(3,3,mol_extended.nao,mol_extended.nao)
            ipvip = mol_extended.intor('int1e_iprinvip', 9).reshape(3,3,mol_extended.nao,mol_extended.nao)
            h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
            h1ao = h1ao + h1ao.transpose(0,1,3,2)
            trace = h1ao[0,0] + h1ao[1,1] + h1ao[2,2]
            idx = np.arange(3)
            h1ao[idx,idx] -= trace
    
        pad_width = h1ao.shape[2]-spin_density.shape[0]
        padded_spin_density = np.pad(spin_density, ((0,pad_width),(0,pad_width)), mode='constant', constant_values=0)
        
        fcsd = np.einsum('xyij,ji->xy', h1ao, padded_spin_density)
        
        hyperfines[number,:,:] = fcsd*533.5514*1000
         
    return hyperfines.astype(np.float64)
