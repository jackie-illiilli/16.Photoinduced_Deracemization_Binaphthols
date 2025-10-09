import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def move(a):
    """
    Converts a 3D translation vector to a 4D homogeneous transformation matrix.

    Args:
        a (iterable): 3D translation vector (x, y, z).

    Returns:
        np.array: 4x4 homogeneous translation matrix.
    """
    x, y, z = a[:3]
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def rotation(a, sin, cos):
    """
    Generates a 4D homogeneous rotation matrix for rotation around axis a by angle defined by sin and cos.

    Args:
        a (np.array): 3D unit axis vector.
        sin (float): Sine of the rotation angle.
        cos (float): Cosine of the rotation angle.

    Returns:
        np.array: 4x4 homogeneous rotation matrix.
    """
    a = np.array(a)
    a = a / np.sqrt(a @ a.T)
    u, v, w = a[:3]
    return np.array([[u * u + (1 - u * u) * cos, u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin, 0],
                     [u * v * (1 - cos) + w * sin, v * v + (1 - v * v) * cos, v * w * (1 - cos) - u * sin, 0],
                     [u * w * (1 - cos) - v * sin, v * w * (1 - cos) + u * sin, w * w + (1 - w * w) * cos, 0],
                     [0, 0, 0, 1]])


def trfm_rot(a, b, c, position=[], center_point=np.array([0, 0, 0])):
    """
    Applies transformations to align points a, b, c such that the vector ab is parallel to the x-axis,
    points a, b, c lie in the xoy plane, and a, b are symmetric about the center_point.

    Args:
        a (np.array): 3D coordinate of point a (will be placed on negative x-axis).
        b (np.array): 3D coordinate of point b (will be placed on positive x-axis).
        c (np.array): 3D coordinate of point c (y-coordinate positive, z-coordinate 0 after transformation).
        position (list, optional): List of positions to transform. If provided, returns transformed positions.
                                  Defaults to [].
        center_point (np.array, optional): Symmetry center point. Defaults to np.array([0, 0, 0]).

    Returns:
        tuple: If position is empty, returns (tr_array, rot1_array, rot2_array, tr_array2).
               Otherwise, returns transformed positions as np.array.
    """
    zero_point = np.array([0, 0, 0])
    x, y, z = np.array(a), np.array(b), np.array(c)
    x_axis = np.array([1, 0, 0])
    mid_point = (x + y) / 2
    tr_array_3d = zero_point - mid_point
    tr_array = move(tr_array_3d)
    x += tr_array_3d
    y += tr_array_3d
    z += tr_array_3d
    tr_array2 = move(center_point - zero_point)
    xy = y - x
    xy = xy / np.sqrt((xy @ xy))
    law_axis = np.cross(xy, x_axis)
    if not (law_axis == np.array([0, 0, 0])).all():
        law_axis /= np.sqrt((law_axis @ law_axis))
    sin = np.sqrt(xy[1] * xy[1] + xy[2] * xy[2])
    cos = xy[0]
    rot1_array = rotation(law_axis, sin, cos)
    oz = z - zero_point
    oz = rot1_array[:3, :3] @ oz
    oz[0] = 0
    oz = oz / np.sqrt((oz @ oz))
    sin = oz[2]
    cos = oz[1]
    rot2_array = rotation(-1 * x_axis, sin, cos)
    if len(position) == 0:
        return tr_array, rot1_array, rot2_array, tr_array2
    else:
        if len(position[0]) == 3:
            up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        else:
            up_position = position.T
        up_position = tr_array @ up_position
        up_position = rot1_array @ up_position
        up_position = rot2_array @ up_position
        up_position = tr_array2 @ up_position
        up_position = up_position.T
        return up_position


def rot_mol(mol, axis=np.array([0, 1, 0]), sin=0, cos=-1):
    """
    Rotates all conformers of a molecule around the specified axis by the given angle (default 180 degrees around y-axis).

    Args:
        mol (Chem.Mol): RDKit molecule object with conformers.
        axis (np.array, optional): Rotation axis vector. Defaults to np.array([0, 1, 0]) (y-axis).
        sin (float, optional): Sine of the rotation angle. Defaults to 0.
        cos (float, optional): Cosine of the rotation angle. Defaults to -1 (180 degrees).

    Returns:
        Chem.Mol: Deep copy of the rotated molecule.
    """
    # Create a deep copy to avoid modifying the original
    react1 = copy.deepcopy(mol)
    for conformer in react1.GetConformers():
        position = conformer.GetPositions()
        up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        matrix = rotation(axis, sin, cos)
        up_position = (matrix @ up_position).T
        for i, c in enumerate(up_position):
            conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
    return react1


def move_mol(mol, array=np.array([0, 0, 1.5])):
    """
    Translates all conformers of a molecule by the specified vector (default +1.5 units along z-axis).

    Args:
        mol (Chem.Mol): RDKit molecule object with conformers.
        array (np.array, optional): Translation vector (dx, dy, dz). Defaults to np.array([0, 0, 1.5]).

    Returns:
        Chem.Mol: Deep copy of the translated molecule.
    """
    # Create a deep copy to avoid modifying the original
    react1 = copy.deepcopy(mol)
    for conformer in react1.GetConformers():
        position = conformer.GetPositions()
        up_position = np.insert(position, 3, np.ones(len(position)), 1).T
        matrix = move(array)
        up_position = (matrix @ up_position).T
        for i, c in enumerate(up_position):
            conformer.SetAtomPosition(i, Point3D(c[0], c[1], c[2]))
    return react1


def smiles2mol(smiles, conf_num=20):
    """
    Converts a SMILES string to an RDKit molecule object, adding hydrogens and generating 3D conformers.

    Args:
        smiles (str): SMILES representation of the molecule.
        conf_num (int, optional): Number of conformers to generate. Defaults to 20.

    Returns:
        Chem.Mol: RDKit molecule with hydrogens and optimized 3D conformers, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles} can't be read")
        return None
    Hmol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(Hmol, numConfs=conf_num, maxAttempts=100)
    try:
        AllChem.MMFFOptimizeMolecule(Hmol)
    except:
        pass  # Ignore optimization failures
    return Hmol