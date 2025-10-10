# Collection of utility functions for molecular calculations and plotting
from inspect import BoundArguments
import math, os, shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def find_first_line(fileline, find_str, find_type="start"):
    """
    Find the first line in a list that matches the specified condition.

    Args:
        fileline (list): List of strings representing lines from a file.
        find_str (str): The string to search for.
        find_type (str): Type of match: 'start' (startswith), 'end' (endswith), 'all' (exact match), or 'in' (contains).

    Returns:
        tuple: (index, line) if found, otherwise (None, None).
    """
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

def clean_nan(input_list):
    """
    Replace NaN values in a numpy array with 0.

    Args:
        input_list (np.array): Input array that may contain NaN values.

    Returns:
        np.array: Array with NaNs replaced by 0.
    """
    return np.nan_to_num(input_list, nan=0)

def get_array_cos(array1, array2):
    """
    Compute the cosine similarity between two arrays (vectors).

    Args:
        array1 (np.array): First input array.
        array2 (np.array): Second input array.

    Returns:
        np.array: Cosine similarity scores.
    """
    return array1 @ array2.T / (np.sqrt(array1 @ array1.T)
                              * np.sqrt(array2 @ array2.T))

def get_atoms_distance(atom_positionA, atom_positionB):
    """
    Calculate the Euclidean distance between two atom positions.

    Args:
        atom_positionA (np.array): 3D coordinates of the first atom.
        atom_positionB (np.array): 3D coordinates of the second atom.

    Returns:
        float: Distance between the two atoms.
    """
    return np.sqrt(sum((atom_positionA - atom_positionB) ** 2))

def get_max_min_bond(positions, atom_lists):
    """
    Compute the maximum and minimum bond lengths for pairs of atoms across two conformations.

    Args:
        positions (np.array): Array of atom positions for two conformations.
        atom_lists (list): List of two sublists, each containing pairs of atom indices for a bond.

    Returns:
        list: List of max and min bond lengths for each bond pair.
    """
    results = []
    for atom_list in atom_lists:
        bond1 = get_atoms_distance(positions[atom_list[0][0]], positions[atom_list[0][1]])
        bond2 = get_atoms_distance(positions[atom_list[1][0]], positions[atom_list[1][1]])
        results.append(np.max([bond1, bond2]))
        results.append(np.min([bond1, bond2]))
    return results

def get_bond_angle_deg(atom_positionA, atom_positionB, atom_positionC):
    """
    Calculate the bond angle in degrees between three atom positions using RDKit.

    Args:
        atom_positionA (np.array): 3D coordinates of the first atom.
        atom_positionB (np.array): 3D coordinates of the central atom.
        atom_positionC (np.array): 3D coordinates of the third atom.

    Returns:
        float: Bond angle in degrees.
    """
    conf = Chem.rdchem.Conformer(3)
    all_positions = [atom_positionA, atom_positionB, atom_positionC]
    for i in range(3):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    return bond_angle

def get_max_min_angle(positions, atom_lists):
    """
    Compute the maximum and minimum bond angles for triplets of atoms across two conformations.

    Args:
        positions (np.array): Array of atom positions for two conformations.
        atom_lists (list): List of two sublists, each containing triplets of atom indices for an angle.

    Returns:
        list: List of max and min angles for each angle triplet.
    """
    results = []
    for atom_list in atom_lists:
        angle1 = get_bond_angle_deg(positions[atom_list[0][0]], positions[atom_list[0][1]], positions[atom_list[0][2]])
        angle2 = get_bond_angle_deg(positions[atom_list[1][0]], positions[atom_list[1][1]], positions[atom_list[1][2]])
        results.append(np.max([angle1, angle2]))
        results.append(np.min([angle1, angle2]))
    return results

def get_torsion_(A, B, C, D):
    """
    Calculate the dihedral (torsion) angle in degrees between four atom positions using RDKit.

    Args:
        A (np.array): 3D coordinates of the first atom.
        B (np.array): 3D coordinates of the second atom.
        C (np.array): 3D coordinates of the third atom.
        D (np.array): 3D coordinates of the fourth atom.

    Returns:
        float: Dihedral angle in degrees.
    """
    conf = Chem.rdchem.Conformer(4)
    all_positions = [A, B, C, D]
    for i in range(4):
        conf.SetAtomPosition(i, all_positions[i][:3])
    bond_angle = Chem.rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
    return bond_angle

def GetSpinMultiplicity(Mol, CheckMolProp=True):
    """
    Get the spin multiplicity of a molecule. Retrieves from 'SpinMultiplicity' property if available,
    otherwise calculates using Hund's rule based on the number of radical electrons.

    Args:
        Mol (rdkit.Chem.Mol): RDKit molecule object.
        CheckMolProp (bool): Whether to check the molecule property first.

    Returns:
        int: Spin multiplicity (2S + 1).
    """
    Name = 'SpinMultiplicity'
    if CheckMolProp and Mol.HasProp(Name):
        return int(float(Mol.GetProp(Name)))

    # Calculate spin multiplicity using Hund's rule of maximum multiplicity
    NumRadicalElectrons = 0
    for Atom in Mol.GetAtoms():
        NumRadicalElectrons += Atom.GetNumRadicalElectrons()

    TotalElectronicSpin = NumRadicalElectrons / 2
    SpinMultiplicity = 2 * TotalElectronicSpin + 1
    
    return int(SpinMultiplicity)

# Draw Figures
def draw_heatmap(x_labels, y_labels, values, title="None", figure_size=(40, 6), min_value=0.0, max_value=1):
    """
    Draw a heatmap using seaborn for visualizing a 2D matrix of values.

    Args:
        x_labels (list): Labels for the x-axis.
        y_labels (list): Labels for the y-axis.
        values (np.array): 2D array of numerical values.
        title (str): Title of the plot.
        figure_size (tuple): Size of the figure (width, height).
        min_value (float): Minimum value for color scaling.
        max_value (float): Maximum value for color scaling.

    Returns:
        matplotlib.pyplot: The plot object.
    """
    import seaborn as sns
    sns.set()
    # Set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    uniform_data = values  # Set 2D matrix
    f, ax = plt.subplots(figsize=figure_size)
    annot_kws = {"fontsize": 30}
    # Heatmap parameters: vmin/vmax for colorbar range, annot for values, linewidths for grid
    sns.heatmap(uniform_data, ax=ax, vmin=min_value, vmax=max_value, cmap='Blues', linewidths=2, cbar=True, annot=True, annot_kws=annot_kws, fmt='.3f')

    ax.set_title(title, fontsize=40)  # Set title
    ax.set_xticklabels(x_labels, fontsize=30)
    ax.set_yticklabels(y_labels, fontsize=30)
    # Set label rotations
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=0, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=0, horizontalalignment='center')
    plt.savefig('test.svg', format='svg')
    plt.show()
    return plt

def calc_distribution2(y, eachsize=0.01, title=None, xlab=None, ylab="Count", y_max=None, y_min=None, color="green"):
    """
    Compute and plot a histogram distribution for a 1D array of values.

    Args:
        y (np.array): 1D array of numerical values.
        eachsize (float): Bin width for the histogram.
        title (str): Title of the plot.
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        y_max (float): Maximum value for x-axis (auto if None).
        y_min (float): Minimum value for x-axis (auto if None).
        color (str): Color for the bars.

    Returns:
        np.array: Histogram counts.
    """
    if y_max is None:
        y_max = np.max(y)
    if y_min is None:
        y_min = np.min(y)
    X = np.arange(y_min, y_max + eachsize, eachsize)
    des = [0 for each in X]
    z = (y - y_min) / eachsize
    for each in z:
        try:
            des[int(each)] += 1
        except:
            continue
    des = np.array(des)
    # des = des / len(y)
    
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    plt.bar(X, des, width=eachsize / 2, color=color)
    plt.xlim(y_min - eachsize, y_max + eachsize)
    plt.ylim(0, np.max(des) * 1.2)
    plt.xlabel(xlab, fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig("test.png", format="png", dpi=300, bbox_inches='tight')
    plt.show()
    return des

def plot_scatter_with_metrics(x, y, title=None, min_=-10, max_=60, figsize=(5, 5)):
    """
    Plot a scatter plot with kernel density estimation and display regression metrics.

    Args:
        x (np.array): 1D array for x-axis (e.g., real values).
        y (np.array): 1D array for y-axis (e.g., predicted values).
        title (str): Title of the plot.
        min_ (float): Minimum limit for axes.
        max_ (float): Maximum limit for axes.
        figsize (tuple): Size of the figure (width, height).

    Returns:
        None
    """
    # Calculate regression metrics
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    mse = mean_squared_error(x, y)

    # Draw scatter plot with KDE
    plt.figure(figsize=figsize, facecolor='white')
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.xlabel("Real", fontsize=18)
    # plt.ylabel("Prediction", fontsize=18)
    if title is not None:
        plt.title("%s\nR2:%.3f, MAE:%.3f, MSE:%.3f" % (title, r2, mae, mse), fontsize=24)
    z = np.linspace(min_, max_, 10000)
    plt.plot(z, z, color='red', linestyle='--')  # Diagonal line

    # plt.scatter(x, y, marker="*", c="g")
    sns.kdeplot(x=x, y=y, cmap="Blues", shade=True, bw_adjust=1, thresh=0.01)
    # Add regression metrics to the plot (already in title)
    
    # Show and save plot
    plt.savefig("test.png", format="png", dpi=300, bbox_inches='tight')
    plt.show()

def calc_distribution_line(ys, eachsize=0.1, title=None, xlab=None, ylab="Freq", return_result=False, labels=None, colors=None, useSVG=False, save_name='test', figure_size=(5, 4), xlimit=[]):
    """
    Plot overlaid bar distributions (normalized histograms) for multiple 1D datasets.

    Args:
        ys (list): List of 1D arrays, each representing a dataset.
        eachsize (float): Bin width for the histograms.
        title (str): Title of the plot.
        xlab (str): Label for the x-axis.
        ylab (str): Label for the y-axis.
        return_result (bool): Whether to return the last histogram counts.
        labels (list): Labels for each dataset (auto if None).
        colors (list): Colors for each dataset (auto if None).
        useSVG (bool): Save as SVG if True, else PNG.
        save_name (str): Base name for the saved file.
        figure_size (tuple): Size of the figure (width, height).
        xlimit (list): [min, max] for x-axis limits (auto if empty).

    Returns:
        np.array or None: Histogram counts for the last dataset if return_result=True.
    """
    from scipy.interpolate import make_interp_spline
    fig = plt.figure(figsize=figure_size)
    y_max = np.max([max(each) for each in ys])
    y_min = np.min([min(each) for each in ys])
    # y_max = 50
    # y_min = 0
    if labels is None:
        labels = [None] * len(ys)
    if colors is None:
        colors = ["blue"] * len(ys)
    X = np.arange(y_min, y_max + eachsize, eachsize)
    all_max = []
    for idx, y in enumerate(ys):
        des = [0 for each in X]
        z = (y - y_min) / eachsize
        for each in z:
            try:
                assert int(each) < len(X)
                des[int(each)] += 1
            except:
                continue
        des = np.array(des)
        des = des / len(y)
        # Commented out spline interpolation and fill; using bars instead
        # x = np.linspace(y_min - eachsize, y_max + eachsize * 2, 1000)
        # model = make_interp_spline(X, des)
        # ys_interp = model(x)
        # all_max.append(max(ys_interp))
        # print(x[np.argmax(ys_interp)])
        # plt.plot(x, ys_interp, color=colors[idx])
        # plt.fill_between(x, ys_interp, 0, where=(ys_interp > 0), interpolate=True, color=colors[idx], alpha=0.3, label=labels[idx])
        plt.bar(X + idx * eachsize / len(ys) / 2, des, width=eachsize / len(ys) / 2, color=colors[idx], label=labels[idx])
    if xlimit != []:
        plt.xlim(xlimit[0], xlimit[1])
    else:
        plt.xlim(y_min - eachsize, y_max + eachsize)
    # plt.ylim(0, 1.1 * max(all_max))
    plt.xlabel(xlab, fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if labels[0] is not None:
        plt.legend()
    if title is not None:
        plt.title(title)
    if useSVG:
        plt.savefig(f"{save_name}.svg", bbox_inches='tight', format='svg')
    else:
        plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.show()  
    if return_result:
        return des