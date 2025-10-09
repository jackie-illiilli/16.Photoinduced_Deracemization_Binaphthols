# 一些常用工具的集合文档
from inspect import BoundArguments
import math, os, shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

# Draw Figures
def draw_heatmap(x_labels, y_labels, values, title="None", figure_size=(40, 6), min_value = 0.0, max_value = 1):
    """Draws a heatmap using seaborn"""
    import seaborn as sns
    sns.set()
    # Set font to Arial

    plt.rcParams['font.sans-serif']='Arial'

    uniform_data = values # Set 2D matrix
    f, ax = plt.subplots(figsize=figure_size)
    annot_kws = {"fontsize": 30}
    # Heatmap parameters: vmin/vmax for colorbar range, annot for values, linewidths for grid
    sns.heatmap(uniform_data, ax=ax,vmin=min_value,vmax=max_value,cmap='Blues',linewidths=2,cbar=True, annot=True,annot_kws=annot_kws, fmt='.3f')

    ax.set_title(title, fontsize=40) # Set title
    ax.set_xticklabels(x_labels, fontsize=30)
    ax.set_yticklabels(y_labels, fontsize=30)
    # Set label rotations
    label_y =  ax.get_yticklabels()
    plt.setp(label_y, rotation=0, horizontalalignment='right')
    label_x =  ax.get_xticklabels()
    plt.setp(label_x, rotation=0, horizontalalignment='center')
    plt.savefig('test.svg', format='svg')
    plt.show()
    return plt

def calc_distribution2(y, eachsize=0.01, title=None, xlab=None, ylab="Count", y_max=None, y_min=None, color="green"):
    if y_max == None:    y_max = np.max(y)
    if y_min == None:    y_min = np.min(y)
    X = np.arange(y_min, y_max + eachsize, eachsize)
    des = [0 for each in X]
    z = (y - y_min)/eachsize
    for each in z:
        try:
            des[int(each)] += 1
        except:
            continue
    des = np.array(des)
    # des = des / len(y)
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    plt.bar(X, des, width=eachsize/2, color=color)
    plt.xlim(y_min - eachsize, y_max + eachsize)
    plt.ylim(0, np.max(des) * 1.2)
    plt.xlabel(xlab, fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if title != None:
        plt.title = title
    plt.tight_layout()
    plt.savefig("test.png", format="png", dpi=300, bbox_inches='tight')
    plt.show()
    return des

def plot_scatter_with_metrics(x, y, title=None, min_=-10, max_=60, figsize = (5,5)):
    """
    绘制散点图并显示回归性能指标
    
    参数：
    x: 一维数组类型，表示x轴数据。
    y: 一维数组类型，表示y轴数据。
    title: 字符串类型，表示图的标题。
    
    返回值：
    None
    
    """
    # 计算回归性能指标
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    mse = mean_squared_error(x, y)

    # 绘制散点图
    plt.figure(figsize=figsize, facecolor='white')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.xlabel("Real", fontsize=18)
    # plt.ylabel("Prediction", fontsize=18)
    if title != None:
        plt.title("%s\nR2:%.3f, MAE:%.3f, MSE:%.3f" % (title, r2, mae, mse), fontsize=24)
        plt.title("%s"%title,fontsize=24)
    z = np.linspace(min_, max_, 10000)
    plt.plot(z, z)

    # plt.scatter(x, y, marker="*", c="g")
    sns.kdeplot(x=x, y=y, cmap="Blues", shade=True, bw_adjust=1, thresh=0.01)
    # 添加回归性能指标到图像的第二行
    
    # 显示图像
    plt.savefig("test.png", format="png", dpi=300, bbox_inches='tight')
    plt.show()