import os
from rdkit.Chem import AllChem

from . import Tool


def mol_to_xyz(mol, atom_list=None, position_list=None, file_dir="test.xyz", title=None):
    """
    Convert a molecule or atom/position lists to one or more XYZ files.

    This function generates XYZ files from either an RDKit molecule object or
    explicit lists of atom symbols and positions. If a molecule is provided,
    it extracts atoms and conformer positions. Multiple conformers result in
    multiple XYZ files.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object. If None, atom_list and position_list must be provided.
        atom_list (list, optional): List of atom symbols (e.g., ['C', 'O']). Required if mol is None.
        position_list (list, optional): List of position arrays (e.g., [[x1,y1,z1], [x2,y2,z2], ...]). Required if mol is None.
        file_dir (str, optional): Base filename for output XYZ files (without extension). Defaults to "test".
        title (str, optional): Title line for the XYZ file(s). Defaults to None.

    Returns:
        list: List of generated XYZ filenames, or None if input is invalid.
    """
    if mol is None and (atom_list is None or position_list is None):
        print("mol, atom_list/position_list must contain one")
        return None

    file_dir = file_dir.split(".")[0]
    if mol is not None:
        atom_num = mol.GetNumAtoms()
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
        position_list = [conf.GetPositions() for conf in mol.GetConformers()]
    else:
        atom_num = len(atom_list)

    file_names = []
    for i, position in enumerate(position_list):
        file_name = "%s_%s.xyz" % (file_dir, str(i))
        with open(file_name, "wt") as f:
            f.write(" %d\n" % atom_num)
            f.write("%s\n" % title)
            for j, atom in enumerate(atom_list):
                f.write("%s %.8f %.8f %.8f\n" % (atom, position[j][0], position[j][1], position[j][2]))
        file_names.append(file_name)
    return file_names


def mol_to_gjf(mol, file_dir="test_data/mol2gjf.gjf", charge=None, SpinMultiplicity=None, title="Title", method="opt freq b3lyp/6-311g(d,p)", confid=0, ignore_warning=False):
    """
    Convert an RDKit molecule to a Gaussian input file (.gjf).

    This function generates a Gaussian .gjf file from an RDKit molecule,
    including route section, charge/multiplicity, and atomic coordinates.
    It adds hydrogens and embeds a conformer if none exists.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
        file_dir (str, optional): Output file path for the .gjf file. Defaults to "test_data/mol2gjf.gjf".
        charge (int, optional): Net charge of the molecule. If None, computed from formal charges. Defaults to None.
        SpinMultiplicity (int, optional): Spin multiplicity. If None, computed using Tool.GetSpinMultiplicity. Defaults to None.
        title (str, optional): Title for the Gaussian job. Defaults to "Title".
        method (str, optional): Gaussian route section method (e.g., "opt freq b3lyp/6-311g(d,p)"). Defaults to "opt freq b3lyp/6-311g(d,p)".
        confid (int, optional): Index of the conformer to use. Defaults to 0.
        ignore_warning (bool, optional): Suppress warning if spin multiplicity != 1. Defaults to False.

    Returns:
        None: Writes the .gjf file to disk.
    """
    if charge is None:
        charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    if SpinMultiplicity is None:
        SpinMultiplicity = Tool.GetSpinMultiplicity(mol)
    dir = os.path.split(file_dir)[0]
    if not os.path.isdir(dir):
        os.mkdir(dir)
    with open(file_dir, "wt") as f:
        f.write("%nprocshared=28\n%mem=56GB\n#p")
        f.write(" %s\n\n" % method)
        f.write("$$$$%s####%d????\n\n" % (title, charge))
        f.write("%d %d\n" % (int(charge), SpinMultiplicity))
        if SpinMultiplicity != 1 and not ignore_warning:
            print("%s 's SpinMultiplicity != 1, check it" % file_dir)
        if len(mol.GetConformers()) == 0:
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=1)
        positions = mol.GetConformer(confid).GetPositions()
        for i, atom in enumerate(mol.GetAtoms()):
            f.write(" %s                 " % atom.GetSymbol())
            f.write("%f %f %f\n" % tuple(positions[i]))
        f.write("\n\n")


def block_to_gjf(symbol_list, positions, file="test_data/mol2gjf.gjf", charge=0, multiplicity=1, title="Title", method="opt freq b3lyp/6-311g(d,p)", freeze=[], difreeze=[], savechk=None, readchk=None, final_line=None):
    """
    Convert atom symbols and positions to a Gaussian input file (.gjf) with optional constraints.

    This function generates a Gaussian .gjf file from lists of atom symbols and 3D positions.
    Supports checkpoint files, freezing bonds/dihedrals (e.g., for TS optimization), and custom final lines.

    Args:
        symbol_list (list): List of atom symbols (e.g., ['C', 'O']).
        positions (list): List of 3D position tuples/arrays (e.g., [[x1,y1,z1], [x2,y2,z2], ...]).
        file (str, optional): Output file path for the .gjf file. Defaults to "test_data/mol2gjf.gjf".
        charge (int, optional): Net charge of the system. Defaults to 0.
        multiplicity (int, optional): Spin multiplicity. Defaults to 1.
        title (str, optional): Title for the Gaussian job. Defaults to "Title".
        method (str, optional): Gaussian route section method (e.g., "opt freq b3lyp/6-311g(d,p)"). Defaults to "opt freq b3lyp/6-311g(d,p)".
        freeze (list, optional): List of bond freeze tuples (atom1, atom2). Defaults to [].
        difreeze (list, optional): List of dihedral freeze tuples (atom1, atom2, atom3, atom4). Defaults to [].
        savechk (str, optional): Name of output checkpoint file (without .chk). Defaults to None.
        readchk (str, optional): Name of input checkpoint file (without .chk). Defaults to None.
        final_line (str, optional): Custom line to append before the blank lines (e.g., for LINK1). Defaults to None.

    Returns:
        None: Writes the .gjf file to disk.
    """
    file_dir, filename = os.path.split(file)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    assert len(symbol_list) == len(positions)
    with open(file, "wt") as f:
        if savechk is not None:
            f.write("%%chk=%s.chk\n" % savechk)
        if readchk is not None:
            f.write("%%oldchk=%s.chk\n" % readchk)
        f.write("%nprocshared=28\n%mem=56GB\n#p")
        f.write(" %s\n\n" % method)
        f.write("$$$$%s####%d????\n\n" % (title, int(charge)))
        f.write("%d %d\n" % (int(charge), int(multiplicity)))
        for i, atom in enumerate(symbol_list):
            f.write(" %s                 " % atom)
            f.write("%f %f %f\n" % tuple(positions[i][:3]))
        f.write("\n")
        for each in freeze:
            f.write("B %d %d F\n" % tuple(each))
        for each in difreeze:
            f.write("D %d %d %d %d F\n" % tuple(each))
        if final_line is not None:
            f.write(final_line)
            f.write("\n")
        f.write("\n\n")