import numpy as np
import re

def get_atom_labels_coordinates_orca(output_file):
    
    coordinates_start_flag = 'CARTESIAN COORDINATES (ANGSTROEM)'
    coordinates_end_flag = 'CARTESIAN COORDINATES (A.U.)'
    
    atom_labels = []
    atom_coordinates = []
    
    with open(output_file,'r') as f:
        
        collect_atom_labels = False
        
        for line in f:
            if coordinates_start_flag in line:
                collect_atom_labels = True
                continue
            elif coordinates_end_flag in line:
                collect_atom_labels = False
                continue
            elif collect_atom_labels:
                if len(line.split()) == 4:
                    atom_labels.append(line.split()[0])
                    atom_coordinates.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
                    
    return atom_labels,np.array(atom_coordinates)

def get_atom_labels_coordinates_xyzfile(xyzfile):
    
    atom_labels = []
    atom_coordinates = []
    with open(xyzfile,'r') as f:
        for line in f:
            if len(line.split()) == 4:
                atom_labels.append(line.split()[0])
                atom_coordinates.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    
    return atom_labels,np.array(atom_coordinates)

def get_hyperfine_from_orca_output(output_file):
    
    with open(output_file,'r') as f:
        for line in f:
            if 'Program Version' in line:
                verscheck = line.split()
                if verscheck[2] == '5.0.4':
                    hfflag = 'Raw HFC matrix'
                elif verscheck[2] == '5.0.3':
                    hfflag = 'Raw HFC matrix'
                elif verscheck[2] == '6.0.1':
                    hfflag = 'Total HFC matrix'  
                elif verscheck[2] == '6.1.1':
                    hfflag = 'Total HFC matrix'
    
    nuccount = 0

    # getting hyperfines
    
    hfdict = {}    
    
    soi = ['H']
    
    with open(output_file,'r') as f:
        HF = False
        for line in f:
            if 'Nucleus ' in line and 'Finite Nucleus Model' not in line:
                
                key = line.split(':')[0].split()[1]
                nuc = re.findall(r'[a-zA-Z]+', key)[0]
                if nuc not in soi:
                    HF = False
                    continue
                elif nuc in soi:
                    HF = True
                    continue
            elif 'A(FC)' in line:
                HF = False
                continue
            elif HF:
                if hfflag in line:
                    hfarr = np.zeros([3,3])
                    dim2count = 0
                    for i in range(4):
                        next_ = next(f).split()
                        if len(next_) == 3:
                            for j,elem in enumerate(next_): 
                                hfarr[dim2count,j] = float(elem)
                            dim2count += 1
                    hfdict[('0',key)] = hfarr*1000                   
    
    return hfdict

def get_basis_set_string(output_file,auxiliary_basis=False):
    
    #defining Boolean flags
    basis_start_flag = 'BASIS SET IN INPUT FORMAT'
    if auxiliary_basis:
        basis_end_flag = 'AUXILIARY/J BASIS SET INFORMATION'
    else:
        basis_end_flag = 'ORCA STARTUP CALCULATIONS'
        
    element_start_flag = 'NewGTO'
    element_end_flag = 'end;'

    basis_set = ''
    
    with open(output_file,'r') as f:
        
        parse_output = False
        element_flag = False
        
        for line in f:
            if basis_start_flag in line and len(line.split()) == 5:
                parse_output = True
                continue
            elif basis_end_flag in line:
                parse_output = False
                continue
            elif parse_output:
                if element_start_flag in line:
                    element_flag = not element_flag 
                elif element_end_flag in line:
                    basis_set += line # adding 'end;' to the end the string
                    element_flag = not element_flag
                
                if element_flag:
                    basis_set += line
    
    return basis_set

def parse_basis_set_string(basis_set_string):
    
    ANGULAR_MOMENTUM_MAP = {'S':0,'P':1,'D':2,'F':3,'G':4}
    
    split_string = basis_set_string.split('end;')
    
    processed_basis_set = {}
    
    for element in split_string[:-1]:
        key = element.split()[1]
        basis = []
        for item in element.split()[2:]:
            boolean_array = [item.isdigit(),item.isalpha()]
            if boolean_array == [False,True] or boolean_array == [False,False]:
                basis.append(item)
                
        processed_list = []
        current_group = None
        
        for item in basis:
            try:
                val = float(item) # if this does not raise a ValueError, the item is a float
                if len(current_group) == 1 or len(current_group[-1]) == 2:
                    current_group.append([val])
                else:
                    current_group[-1].append(val)
            except ValueError:
                current_group = [ANGULAR_MOMENTUM_MAP[item]]
                processed_list.append(current_group)
            
        processed_basis_set[key] = processed_list
        
    return processed_basis_set

def spin_density_from_orca_output(output_file,casscf=False):

    if casscf:
        spin_density_start_flag = 'SPIN-DENSITY MATRIX'
        spin_density_end_flag = 'Trace of the spin density'
        
        with open(output_file,'r') as f:
            for line in f:
                if 'Active       ' in line:
                    n_act_orbs = int(line.split()[5])
                    active_indices = (int(line.split()[1]),int(line.split()[3]))
        
        spin_density = np.zeros([n_act_orbs,n_act_orbs])
    
    else:
        spin_density_start_flag = 'SPIN DENSITY'
        spin_density_end_flag = 'MULLIKEN POPULATION ANALYSIS'
    
        with open(output_file,'r') as f:
            for line in f:
                if 'Number of basis functions' in line:
                    nao = int(line.split()[-1])
   
        spin_density = np.zeros([nao,nao])

    with open(output_file,'r') as f: 
        
        collect_spin_density = False
        
        for line in f:
            if spin_density_start_flag in line:
                collect_spin_density = True
                continue
            elif spin_density_end_flag in line:
                collect_spin_density = False
                continue    
            elif collect_spin_density:
                
                if len(line.split()) > 1:
                    types = [i.isdigit() for i in line.split()]
                    if all(types):
                        columns = [int(i) for i in line.split()]
                    else:
                        row = int(line.split()[0])
                        matrix_elements = [float(i) for i in line.split()[1:]]
                    
                        spin_density[row,columns] = matrix_elements
                        
    if casscf:
        return (spin_density,active_indices)
    else:
        return (spin_density,)
