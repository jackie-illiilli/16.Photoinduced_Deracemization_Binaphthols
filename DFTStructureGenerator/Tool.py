# 一些常用工具的集合文档
from inspect import BoundArguments
import math, os, shutil
import numpy as np
import pandas as pd
from rdkit import Chem


def find_first_line(fileline, find_str, find_type="start"):
    assert find_type in ['start', 'end', 'all', 'in']
    if find_type == "start":
        line_function = lambda x, y: x.startswith(y)
    elif find_type == 'end':
        line_function = lambda x, y: x.endswith(y)
    elif find_type == 'all':
        line_function = lambda x, y: x == y
    else:
        line_function = lambda x, y: y in x

    for idx, line in enumerate(fileline):
        if line_function(line, find_str):
            return idx, line
    return None, None

def remove_same(lists):
    """Remove duplicates from a list of lists.

    Args:
        lists (list): A list of lists.

    Returns:
        list: A list of lists without duplicates.
    """
    seen = set()
    return_list = []
    for each in lists:
        # Convert each list to a tuple so it can be hashed
        t = tuple(each)
        # Check if the tuple is already in the set
        if t not in seen:
            # Add it to the set and the return list
            seen.add(t)
            return_list.append(each)
    return return_list

def save_load(file_name, smiles_lists = None):
    if smiles_lists != None and len(smiles_lists) != 0:
        with open(file_name, "wt") as f:
            for each in smiles_lists:
                f.write(each + "\n")
        return None
    else:
        smiles_lists = []
        with open(file_name, "rt") as f:
            for eachline in f.readlines():
                smiles_lists.append(eachline.strip("\n"))
        print(len(smiles_lists))
        smiles_lists_ = smiles_lists
        smiles_lists = []
        return smiles_lists_

def clean_nan(input_list):
    return np.nan_to_num(input_list, nan=0)

def get_array_cos(array1, array2):
    return array1 @ array2.T / (np.sqrt(array1 @ array1.T)
                              * np.sqrt(array2 @ array2.T))

def get_atoms_distance(atom_positionA, atom_positionB):
    """[summary]

    Args:
        atom_positionA (array): [description]
        atom_positionA (array): [description]

    Returns:
        array: distance
    """
    return np.sqrt(sum((atom_positionA - atom_positionB) ** 2))

def get_max_min_bond(positions, atom_lists):
    results = []
    for atom_list in atom_lists:
        results.append(np.max([get_atoms_distance(positions[atom_list[0][0]], positions[atom_list[0][1]]), get_atoms_distance(positions[atom_list[1][0]], positions[atom_list[1][1]])]))
        results.append(np.min([get_atoms_distance(positions[atom_list[0][0]], positions[atom_list[0][1]]), get_atoms_distance(positions[atom_list[1][0]], positions[atom_list[1][1]])]))
    return results

def get_bond_angle_deg(atom_positionA, atom_positionB, atom_positionC):
    conf = Chem.rdchem.Conformer(3)
    all_positions = [atom_positionA, atom_positionB, atom_positionC]
    for i in range(3):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    return bond_angle

def get_max_min_angle(positions, atom_lists):
    results = []
    for atom_list in atom_lists:
        results.append(np.max([get_bond_angle_deg(positions[atom_list[0][0]], positions[atom_list[0][1]], positions[atom_list[0][2]]), get_bond_angle_deg(positions[atom_list[1][0]], positions[atom_list[1][1]], positions[atom_list[1][2]])]))
        results.append(np.min([get_bond_angle_deg(positions[atom_list[0][0]], positions[atom_list[0][1]], positions[atom_list[0][2]]), get_bond_angle_deg(positions[atom_list[1][0]], positions[atom_list[1][1]], positions[atom_list[1][2]])]))
    return results

def get_torsion(A, B, C, D):
    """计算A-B-C-D二面角的cos值

    Args:
        A (array): points
        B (array): 
        C (array): 
        D (array): 

    Returns:
        cos: _description_
    """    
    AB_AC = np.cross((B - A), (C - A))
    DB_DC = np.cross((B - D), (C - D))
    cos0 = get_array_cos(AB_AC, DB_DC)

    return cos0
def get_torsion_(A, B, C, D):
    conf = Chem.rdchem.Conformer(4)
    all_positions = [A, B, C, D]
    for i in range(4):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
    return bond_angle


def GetSpinMultiplicity(Mol, CheckMolProp = True):
    """From RDKitUtil.py
    Get spin multiplicity of a molecule. The spin multiplicity is either
    retrieved from 'SpinMultiplicity' molecule property or calculated from
    from the number of free radical electrons using Hund's rule of maximum
    multiplicity defined as 2S + 1 where S is the total electron spin. The
    total spin is 1/2 the number of free radical electrons in a molecule.

    Arguments:
        Mol (object): RDKit molecule object.
        CheckMolProp (bool): Check 'SpinMultiplicity' molecule property to
            retrieve spin multiplicity.

    Returns:
        int : Spin multiplicity.

    """
    
    Name = 'SpinMultiplicity'
    if (CheckMolProp and Mol.HasProp(Name)):
        return int(float(Mol.GetProp(Name)))

    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
    NumRadicalElectrons = 0
    for Atom in Mol.GetAtoms():
        NumRadicalElectrons += Atom.GetNumRadicalElectrons()

    TotalElectronicSpin = NumRadicalElectrons/2
    SpinMultiplicity = 2 * TotalElectronicSpin + 1
    
    return int(SpinMultiplicity)


def stablize_smileses(smiles_list):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(each)) for each in smiles_list]

def Calc_areas_(mol, excluded_atoms_ids=[], num_per_axis=20, radius=8, count_per_axis = [2,2,2]):
    # 非均匀格点积分
    table = Chem.rdchem.GetPeriodicTable()
    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    atomicnum_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    geom = mol_conformers[0].GetPositions()

    # 正方体边长和总点数
    num = num_per_axis
    radius /= 2
    cube_length = 2 * radius
    total_points = num * num * num 
    count_num = count_per_axis[0] * count_per_axis[1] * count_per_axis[2]
    counts = np.zeros(count_num, dtype=np.int32)
    counts_num_each = np.zeros(count_num, dtype=np.int32)

    # 生成均匀的网格点
    x = np.linspace(0.1 -radius, radius - 0.1, num)
    y = x; z = x
    # 生成点
    points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    points_inside = np.zeros(total_points, dtype=bool)
    # 计算每个点到每个原子中心的距离
    for atom_id, (atomic_num, sphere_center) in enumerate(zip(atomicnum_list, geom)):
        if atom_id in excluded_atoms_ids:
            continue
        radii = table.GetRvdw(atomic_num)
        translated_points = points - sphere_center
        distances = np.linalg.norm(translated_points, axis=1)
        points_inside = points_inside | (distances <= radii)
    if count_per_axis == None:
        return points_inside
    for i in range(count_num):
        x_id = i // (count_per_axis[1] * count_per_axis[2]) % count_per_axis[0]
        y_id = i // (count_per_axis[2]) % count_per_axis[1]
        z_id = i % count_per_axis[2]
        x_range =  [- radius + 2 * x_id * radius / count_per_axis[0], - radius + 2 * (x_id + 1) * radius / count_per_axis[0]]
        y_range =  [- radius + 2 * y_id * radius / count_per_axis[1], - radius + 2 * (y_id + 1) * radius / count_per_axis[1]]
        z_range =  [- radius + 2 * z_id * radius / count_per_axis[2], - radius + 2 * (z_id + 1) * radius / count_per_axis[2]]
        counts[i] = np.sum(points_inside[np.all([points[:, 0] >= x_range[0], points[:, 1] >= y_range[0], points[:, 2] >= z_range[0], points[:, 0] <= x_range[1], points[:, 1] <= y_range[1], points[:, 2] <= z_range[1]], axis=0)])
        counts_num_each[i] = np.sum(np.all([points[:, 0] >= x_range[0], points[:, 1] >= y_range[0], points[:, 2] >= z_range[0], points[:, 0] <= x_range[1], points[:, 1] <= y_range[1], points[:, 2] <= z_range[1]], axis=0))
    counts = counts / counts_num_each
    
    return counts.tolist()