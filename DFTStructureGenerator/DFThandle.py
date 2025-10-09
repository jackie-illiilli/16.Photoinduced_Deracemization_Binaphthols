import glob, os, shutil
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from . import logfile_process, FormatConverter, xtb_process

def Single_Xtb(root_file, result_file, shift_to_sugan=False):
    """Produces xtb optimization files for single molecules

    Args:
        root_file (str): Path to the root directory
        result_file (str): Path to the CSV file storing molecule information
        shift_to_sugan (bool, optional): Whether to shift to sugan format. Defaults to False.
    """    
    result_file = pd.read_csv(result_file)
    mol_xtb_file = os.path.join(root_file, 'Mol_xtb')
    mol_file = os.path.join(root_file, 'Mols')
    if not os.path.isdir(mol_xtb_file):
        os.mkdir(mol_xtb_file)
    all_mols = []
    all_names = []
    for ix, row in result_file.iterrows():
        mol_idx = row['Index']
        mol_name = f"{mol_idx:05}_r"
        mol = Chem.MolFromMolFile(os.path.join(mol_file, f"{mol_idx:05}.mol"), removeHs=False)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            print(mol_idx)
        all_mols.append(mol)                                                                                
        all_names.append(mol_name)
    xtb_process.xtb_main(all_names, all_mols, dir_path=mol_xtb_file, core=60)
    if shift_to_sugan:
        xtb_process.shift_to_sugan(mol_xtb_file)

def smiles_DFT_calc(root_dir='first_xtb', 
                    mol_dir='mol', 
                    dft_dir='mol_dft', 
                    method="opt freq b3lyp/6-31g* em=gd3bj",
                    conf_limit=3,
                    rmsd_limit=1.5,
                    SpinMultiplicity = None
                    ):
    """Generates Gaussian optimization input files from XTB results

    Args:
        root_dir (str, optional): Root directory for XTB results. Defaults to 'first_xtb'.
        mol_dir (str, optional): Directory for molecule files. Defaults to 'mol'.
        dft_dir (str, optional): Directory to store Gaussian input files. Defaults to 'mol_dft'.
        method (str, optional): Gaussian method string. Defaults to "opt freq b3lyp/6-31g* em=gd3bj".
        conf_limit (int, optional): Conformation limit for XTB structures. Defaults to 3.
        rmsd_limit (float, optional): RMSD limit for XTB structures. Defaults to 1.5.
        SpinMultiplicity (int, optional): Specified spin multiplicity. Defaults to None.
    """                    
    all_files = glob.glob(root_dir + "/*/*/*")
    for xtb_file in all_files:
        try:
            if ("crest.out" in xtb_file) or ("best" in xtb_file) or ("crest_conf" in xtb_file):
                pass
            else:
                if os.path.isdir(xtb_file):
                    shutil.rmtree(xtb_file)
                else:
                    os.remove((xtb_file))
        except:
            continue
    xtb_dirs = glob.glob(root_dir + "/*/*")
    for i, xtb_dir in enumerate(xtb_dirs):
        mol_name = os.path.split(xtb_dir)[-1].split("_")[0]
        mol_file = mol_dir + f"/{mol_name}.mol" 
        mol = Chem.MolFromMolFile(mol_file, removeHs=False, sanitize=False)
        title = "Singlemol"
        charge = int(os.path.split(xtb_dir)[0].split("_")[-2])
        # if charge == 0:
        #     continue
        xtb_process.after_xtb(mol,xtb_dir=xtb_dir, save_dir=dft_dir, xtb_title=title, method=method, conf_limit=conf_limit, rmsd_limit=rmsd_limit, SpinMultiplicity=SpinMultiplicity)


def SPE_DFT_calc(target_dir, opt_name="Reactants", eng_name='Reactants_eng', save_chk=None, method="b3lyp/6-311+g(d,p) em=gd3bj"):
    """Generates single-point energy Gaussian input files from optimization logs

    Args:
        target_dir (str): Target directory containing logs and mols
        opt_name (str, optional): Name of optimization directory. Defaults to "Reactants".
        eng_name (str, optional): Name of energy directory. Defaults to 'Reactants_eng'.
        save_chk (str, optional): Checkpoint file option. Defaults to None.
        method (str, optional): Gaussian method for single-point. Defaults to "b3lyp/6-311+g(d,p) em=gd3bj".
    """
    opt_file_dir = os.path.join(target_dir, opt_name)
    eng_dir = os.path.join(target_dir, eng_name)
    mol_files = glob.glob(target_dir + "/Mols/*.mol")
    for mol_file in mol_files:
        log_files = glob.glob(opt_file_dir + "/" + os.path.split(mol_file)[-1].split(".")[0] + "*.log")
        if len(log_files) == 0:
            continue 
        for log_file in log_files:
            new_log_name = eng_dir + "/" + os.path.split(log_file)[-1].split('.')[0] + ".gjf" 
            opt_log = logfile_process.Logfile(log_file, mol_file_dir=mol_file)
            assert len(opt_log.running_positions) != 0
            title, charge, symbol_list, position,= opt_log.title, opt_log.charge, opt_log.symbol_list, opt_log.running_positions[-1]
            title = " ".join(str(each) for each in title)
            if save_chk:
                savechk = os.path.split(new_log_name.strip(".gjf"))[-1]
            else:
                savechk = None
            FormatConverter.block_to_gjf(symbol_list, position, new_log_name, charge, opt_log.multiplicity, title,
                        method=method, savechk=savechk)


def SPE_DFT_calc_wfn(target_dir, opt_name="Reactants", eng_name='Reactants_eng', save_chk=None, method="b3lyp/6-311+g(d,p) em=gd3bj"):
    """Generates single-point energy Gaussian input files with WFN output from optimization logs

    Args:
        target_dir (str): Target directory containing logs and mols
        opt_name (str, optional): Name of optimization directory. Defaults to "Reactants".
        eng_name (str, optional): Name of energy directory. Defaults to 'Reactants_eng'.
        save_chk (str, optional): Checkpoint file option. Defaults to None.
        method (str, optional): Gaussian method for single-point. Defaults to "b3lyp/6-311+g(d,p) em=gd3bj".
    """
    opt_file_dir = os.path.join(target_dir, opt_name)
    eng_dir = os.path.join(target_dir, eng_name)
    mol_files = glob.glob(target_dir + "/Mols/*.mol")
    for mol_file in tqdm(mol_files):
        log_files = glob.glob(opt_file_dir + "/" + os.path.split(mol_file)[-1].split(".")[0] + "*.log")
        if len(log_files) == 0:
            continue 
        for log_file in log_files:
            new_log_name = eng_dir + "/" + os.path.split(log_file)[-1].split('.')[0] + ".gjf" 
            opt_log = logfile_process.Logfile(log_file, mol_file_dir=mol_file)
            assert len(opt_log.running_positions) != 0
            wfn_name = os.path.split(log_file)[-1].split('.')[0] + ".wfn"
            title, charge, symbol_list, position,= opt_log.title, opt_log.charge, opt_log.symbol_list, opt_log.running_positions[-1]
            title = " ".join(str(each) for each in title)
            FormatConverter.block_to_gjf(symbol_list, position, new_log_name, charge, opt_log.multiplicity, title,
                        method=method, final_line=wfn_name)

def error_improve(target_dir, mol_dir, file_name, dust_bin='dust_bin', improve_dir='improve', bond_attach_std='mol', maxcycles=0, method=None, bond_addition_function=None, bond_ignore_list=None, Inv_dir = 'Inv3', yqc_dir = 'yqc'):
    """
    Works in conjunction with the log_process module to identify and modify errors in Gaussian output files,
    handling imaginary frequency issues. Modified input files and output files with calculation errors or
    imaginary frequencies will be moved to specified folders.

    Args:
        target_dir (str): Root directory
        mol_dir (str): Directory corresponding to the Mol molecule
        file_name (str): Folder in the root directory containing Gaussian output files
        dust_bin (str, optional): Trash bin for irreparable output files. Defaults to 'dust_bin'.
        improve_dir (str, optional): Directory for storing modified Gaussian input files. Defaults to 'improve'.
        bond_attach_std (str, optional): Standard for bond attachment. Defaults to 'mol'.
        maxcycles (int, optional): Maximum number of optimization cycles. Defaults to 0.
        method (str, optional): Optimization method. Defaults to None.
        bond_addition_function (callable, optional): Function for adding bonds. Defaults to None.
        bond_ignore_list (list, optional): List of bonds to ignore. Defaults to None.
        Inv_dir (str, optional): Directory for inversion calculations. Defaults to 'Inv3'.
        yqc_dir (str, optional): Directory for YQC calculations. Defaults to 'yqc'.
    """
    # Construct paths for directories
    opt_file_dir = target_dir + "/" + file_name
    dust_bin_dir = target_dir + "/" + dust_bin
    improve_dir_ = target_dir + "/" + improve_dir
    Inv_dir_ = target_dir + "/" + Inv_dir
    yqc_dir_ = target_dir + "/" + yqc_dir
    mol_files = glob.glob(mol_dir + "/*.mol")
    
    for mol_file in mol_files:
        try:
            print("process %s" % mol_file, end='\r')
            # Find corresponding log files for the mol file
            log_files = glob.glob(opt_file_dir + "/" + os.path.split(mol_file)[-1].split(".")[0] + "*.log")
            if len(log_files) == 0:
                continue 
            for log_file in log_files:
                fail = 0
                with open(log_file, "r") as f:
                    lines = f.readlines()
                if len(lines) <= 15:
                    # File is too short, likely incomplete
                    fail = 2
                else:
                    # Process the log file using logfile_process
                    opt_log = logfile_process.Logfile(log_file, mol_file_dir=mol_file, bond_attach_std=bond_attach_std, bond_addition_function=bond_addition_function, bond_ignore_list=bond_ignore_list)
                    if opt_log.multiplicity <= 0:  # or opt_log.S_2 < 0:
                        # Invalid multiplicity
                        fail = 1
                    elif not opt_log.bond_attach:
                        # Bond attachment failed
                        fail = 1
                    elif opt_log.file_type == "OM" and opt_log.unreal_freq == 0:
                        print("%s may not be a right OM for unreal freq num of %d" % (opt_log.file_dir, opt_log.unreal_freq))
                        # fail = 1
                    elif opt_log.file_type == "TS":
                        if opt_log.unreal_freq >= 0 and opt_log.unreal_freq != 1:  # or not opt_log.is_right_ts:
                            print("%s is not a right TS for unreal freq num of %d" % (opt_log.file_dir, opt_log.unreal_freq))
                            if opt_log.unreal_freq == 0:
                                fail = 1
                            else:
                                opt_log.is_normal_end = 0
                                opt_log.error_reason = ""
                    if opt_log.file_type == "IRC":
                        if opt_log.irc_result == False:
                            # IRC calculation failed
                            fail = 1
                        else:
                            continue
                    # except:
                    #     fail = 2 
                    
                if fail == 1:
                    # Move failed files to dust bin
                    new_log_name = dust_bin_dir + "/" + os.path.split(log_file)[-1] 
                    # new_log_name =  new_log_name.split(".")[0] + "%s.log" % opt_log.file_type
                    if not os.path.isdir(dust_bin_dir):
                        os.mkdir(dust_bin_dir) 
                    shutil.move(log_file, new_log_name)
                    continue
                if fail == 2:
                    # Move incomplete files to improve directory for manual fixing
                    new_log_name = improve_dir_ + "/" + os.path.split(log_file)[-1] 
                    if not os.path.isdir(improve_dir_):
                        os.mkdir(improve_dir_) 
                    shutil.move(log_file, new_log_name)
                    gjf_file = log_file.split(".")[0] + ".gjf"
                    new_gjf_name = improve_dir_ + "/" + os.path.split(gjf_file)[-1] 
                    shutil.move(gjf_file, new_gjf_name)
                    continue
                new_log_name = target_dir + '/' + improve_dir
                savechk = None
                readchk = None
                # if opt_log.file_type == "OM":
                #     savechk = os.path.split(log_file)[-1].split(".")[0]
                # if opt_log.file_type == "TS":
                #     readchk = os.path.split(log_file)[-1].split(".")[0]
                if not opt_log.normal_end:
                    # Solve errors in the log file and generate improved version
                    opt_log.solve_error_logfile(new_log_name, Inv_dir=Inv_dir_, yqc_dir=yqc_dir_, savechk=savechk, readchk=readchk, maxcycles=maxcycles, method=method)
                elif opt_log.unreal_freq and opt_log.file_type not in ["OM", "TS"]:
                    # Handle imaginary frequencies for non-OM/TS files
                    opt_log.unreal_freq_improve(new_log_name, savechk=savechk, readchk=readchk, method=method)
        except:
            # Skip on any exception during processing
            continue


def load_and_prepare_mol(ligand_idx: int, conf_id: int, mol_dir: str, dft_dir: str):
    """
    Loads molecule file, updates geometry from log file, and performs Kekulize processing.
    
    :param ligand_idx: Ligand index
    :param conf_id: Conformation ID
    :param mol_dir: Molecule file directory
    :param dft_dir: DFT log file directory
    :return: Prepared RDKit Mol object and log object
    """
    mol_path = os.path.join(mol_dir, f"{ligand_idx:05}.mol")
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    
    log_path = os.path.join(dft_dir, f"{ligand_idx:05}_r_{conf_id:04}.log")
    log = logfile_process.Logfile(log_path)
    symbol_list, positions = log.symbol_list, log.running_positions[-1:]
    mol = xtb_process.xtb_to_mol(mol, [symbol_list], positions, 1)
    
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol, log


def prepare_ligand_for_cu(mol: Chem.Mol, sites: list[int]) -> Chem.Mol:
    """
    Prepares Ligand-type molecule for Cu coordination: adds dummy O atom, sets charges, and optimizes.
    
    :param mol: Input molecule
    :param sites: List of coordination site indices
    :return: Optimized molecule (with dummy O)
    """
    rwmol = Chem.RWMol(mol)
    new_atom_id = rwmol.AddAtom(Chem.Atom("O"))
    
    for site in sites:
        rwmol.AddBond(site, new_atom_id, Chem.BondType.SINGLE)
        rwmol.GetAtomWithIdx(site).SetFormalCharge(1)
    
    rwmol.GetAtomWithIdx(new_atom_id).SetFormalCharge(len(sites) - 2)
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    # Initial constrained optimization: constrain distances between sites
    ff = AllChem.UFFGetMoleculeForceField(mol)
    for site_id, site in enumerate(sites[1:]):
        ff.UFFAddDistanceConstraint(sites[site_id], site, False, 2.0, 2.8, 1000)
    ff.Initialize()
    ff.Minimize()
    
    # Move O far away and global optimization
    conf = mol.GetConformer(0)
    center_position = np.mean(conf.GetPositions()[sites], axis=0)
    conf.SetAtomPosition(new_atom_id, center_position + np.array([10, 10, 10]))
    AllChem.UFFOptimizeMolecule(mol)
    
    return mol


def prepare_binol_for_cu(mol: Chem.Mol, sites: list[int]) -> Chem.Mol:
    """
    Prepares Binol-type molecule for Cu coordination: removes H atoms, adds dummy O atom, sets charges, and optimizes.
    
    :param mol: Input molecule
    :param sites: List of coordination site indices
    :return: Optimized molecule (with dummy O)
    """
    rwmol = Chem.RWMol(mol)
    
    # Remove H neighbors from each site
    for site in sites:
        neighbors = [neighbor.GetIdx() for neighbor in rwmol.GetAtomWithIdx(site).GetNeighbors() 
                     if neighbor.GetSymbol() == "H"]
        if neighbors:
            rwmol.RemoveAtom(neighbors[0])
    
    new_atom_id = rwmol.AddAtom(Chem.Atom("O"))
    
    for site in sites:
        rwmol.AddBond(site, new_atom_id, Chem.BondType.SINGLE)
        rwmol.GetAtomWithIdx(site).SetFormalCharge(0)
    
    rwmol.GetAtomWithIdx(new_atom_id).SetFormalCharge(len(sites) - 2)
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    # Move O far away and global optimization
    conf = mol.GetConformer(0)
    center_position = np.mean(conf.GetPositions()[sites], axis=0)
    conf.SetAtomPosition(new_atom_id, center_position + np.array([10, 10, 10]))
    AllChem.UFFOptimizeMolecule(mol)
    
    return mol


def convert_to_cu_complex(mol: Chem.Mol, sites: list[int]) -> Chem.Mol:
    """
    Converts dummy O to Cu atom, removes bonds, resets charges, and sanitizes.
    
    :param mol: Molecule with dummy O
    :param sites: List of coordination site indices
    :return: Cu complex molecule
    """
    rwmol = Chem.RWMol(mol)
    new_atom_id = rwmol.GetNumAtoms() - 1  # Assume new atom is the last one
    
    rwmol.GetAtomWithIdx(new_atom_id).SetFormalCharge(2)
    rwmol.GetAtomWithIdx(new_atom_id).SetAtomicNum(29)  # Cu
    
    for site in sites:
        rwmol.RemoveBond(site, new_atom_id)
        rwmol.GetAtomWithIdx(site).SetFormalCharge(0)
    
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def add_distance_constraint(ff: AllChem.ForceField, sites: list[int], cu_idx: int, min_dist: float, max_dist: float, force_const: float):
    """
    Adds Cu-coordination site distance constraints.
    
    :param ff: Force field object
    :param sites: List of coordination site indices
    :param cu_idx: Cu atom index
    :param min_dist: Minimum distance
    :param max_dist: Maximum distance
    :param force_const: Force constant
    """
    # First site: only distance constraint
    ff.UFFAddDistanceConstraint(sites[0], cu_idx, False, min_dist, max_dist, force_const)
    
    # Other sites: distance and angle constraints
    for site_id, site in enumerate(sites[1:]):
        ff.UFFAddDistanceConstraint(site, cu_idx, False, min_dist, max_dist, force_const)
        ff.UFFAddAngleConstraint(sites[site_id], cu_idx, site, False, 80, 110 if force_const < 1000 else 100, force_const)


def perform_cu_optimization_stage(mol: Chem.Mol, sites: list[int], cu_idx: int, min_dist: float, max_dist: float, force_const: float):
    """
    Performs one optimization stage for Cu complex: adds constraints and minimizes.
    
    :param mol: Cu complex molecule
    :param sites: List of coordination site indices
    :param cu_idx: Cu atom index
    :param min_dist: Minimum distance
    :param max_dist: Maximum distance
    :param force_const: Force constant
    """
    ff = AllChem.UFFGetMoleculeForceField(mol)
    add_distance_constraint(ff, sites, cu_idx, min_dist, max_dist, force_const)
    ff.Initialize()
    ff.Minimize()


def optimize_cu_complex(mol: Chem.Mol, sites: list[int], M_L_DIST):
    """
    Performs multi-stage UFF optimization on Cu complex.
    
    :param mol: Cu complex molecule
    :param sites: List of coordination site indices
    :param M_L_DIST: Metal-ligand distance constant
    """
    cu_idx = mol.GetNumAtoms() - 1  # Assume Cu is the last atom
    
    # Stage 1: Loose constraints
    perform_cu_optimization_stage(mol, sites, cu_idx, 1.3, 1.9, 10)
    
    # Stage 2: Medium constraints
    perform_cu_optimization_stage(mol, sites, cu_idx, 1.6, 1.9, 100)
    
    # Stage 3: Tight constraints
    perform_cu_optimization_stage(mol, sites, cu_idx, 1.95, M_L_DIST, 1000)


def save_cu_structure(mol: Chem.Mol, ligand_idx: int, cu_mol_dir: str, cu_dft_dir: str, log_charge: int, charge_offset: int = 0, OPT_METHOD: str = "opt freq b3lyp/def2svpp em=gd3bj nosymm"):
    """
    Saves Cu complex as mol file and gjf file.
    
    :param mol: Cu complex molecule
    :param ligand_idx: Ligand index
    :param cu_mol_dir: Cu mol output directory
    :param cu_dft_dir: Cu gjf output directory
    :param log_charge: Charge from log
    :param charge_offset: Charge offset (Ligand: +2, Binol: 0)
    :param OPT_METHOD: Gaussian optimization method string
    """
    mol_path = os.path.join(cu_mol_dir, f"{ligand_idx:05}.mol")
    Chem.MolToMolFile(mol, mol_path)
    
    symbol_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer(0).GetPositions()
    gjf_path = os.path.join(cu_dft_dir, f"{ligand_idx:05}.gjf")
    total_charge = log_charge + charge_offset
    FormatConverter.block_to_gjf(
        symbol_list, positions, gjf_path, charge=total_charge, multiplicity=2,
        title="Title", method=OPT_METHOD, freeze=[], difreeze=[],
        savechk=None, readchk=None, final_line=None
    )


def generate_ligand_cu_guesses(target_csv: str, mol_dir: str, dft_dir: str, cu_mol_dir: str, cu_dft_dir: str, M_L_DIST: float = 1.96, OPT_METHOD: str = "opt freq b3lyp/def2svpp em=gd3bj nosymm"):
    """
    Generates initial guesses for Ligand-Cu structures.
    
    :param target_csv: Target CSV file path
    :param mol_dir: Molecule file directory
    :param dft_dir: DFT log directory
    :param cu_mol_dir: Cu mol output directory
    :param cu_dft_dir: Cu gjf output directory
    :param M_L_DIST: Metal-ligand distance constant
    :param OPT_METHOD: Gaussian optimization method string
    """
    smiles_csv = pd.read_csv(target_csv)
    smiles_csv = smiles_csv.loc[(smiles_csv['Type'] == "Ligand_Box") | (smiles_csv['Type'] == "Ligand_Other")]
    
    for _, row in smiles_csv.iterrows():
        ligand_idx = row["Index"]
        conf_id = row["conf_id"]
        sites = [int(each) for each in row['Sites'].split()]
        
        mol, log = load_and_prepare_mol(ligand_idx, conf_id, mol_dir, dft_dir)
        mol = prepare_ligand_for_cu(mol, sites)
        mol = convert_to_cu_complex(mol, sites)
        optimize_cu_complex(mol, sites, M_L_DIST)
        save_cu_structure(mol, ligand_idx, cu_mol_dir, cu_dft_dir, log.charge, charge_offset=2, OPT_METHOD=OPT_METHOD)


def generate_binol_cu_guesses(target_csv: str, mol_dir: str, dft_dir: str, cu_mol_dir: str, cu_dft_dir: str, M_L_DIST: float = 1.96, OPT_METHOD: str = "opt freq b3lyp/def2svpp em=gd3bj nosymm"):
    """
    Generates initial guesses for Binol-Cu structures.
    
    :param target_csv: Target CSV file path
    :param mol_dir: Molecule file directory
    :param dft_dir: DFT log directory
    :param cu_mol_dir: Cu mol output directory
    :param cu_dft_dir: Cu gjf output directory
    :param M_L_DIST: Metal-ligand distance constant
    :param OPT_METHOD: Gaussian optimization method string
    """
    smiles_csv = pd.read_csv(target_csv)
    smiles_csv = smiles_csv.loc[smiles_csv['Type'] == "Binol"]
    
    for _, row in smiles_csv.iterrows():
        ligand_idx = row["Index"]
        conf_id = row["conf_id"]
        sites = [int(each) for each in row['Sites'].split()]
        
        mol, log = load_and_prepare_mol(ligand_idx, conf_id, mol_dir, dft_dir)
        mol = prepare_binol_for_cu(mol, sites)
        mol = convert_to_cu_complex(mol, sites)
        optimize_cu_complex(mol, sites, M_L_DIST)
        save_cu_structure(mol, ligand_idx, cu_mol_dir, cu_dft_dir, log.charge, charge_offset=0, OPT_METHOD=OPT_METHOD)

def Calc_areas_(mol, excluded_atoms_ids=[], num_per_axis=20, radius=8, count_per_axis = [2,2,2]):
    """Calculates areas using non-uniform grid integration"""
    # Non-uniform grid integration
    table = Chem.rdchem.GetPeriodicTable()
    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    atomicnum_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    geom = mol_conformers[0].GetPositions()

    # Cube edge length and total points
    num = num_per_axis
    radius /= 2
    cube_length = 2 * radius
    total_points = num * num * num 

    # Generate uniform grid points
    x = np.linspace(0.1 -radius, radius - 0.1, num)
    y = x; z = x
    # Generate points
    points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    points_inside = np.zeros(total_points, dtype=bool)
    # Calculate distance from each point to each atom center
    for atom_id, (atomic_num, sphere_center) in enumerate(zip(atomicnum_list, geom)):
        if atom_id in excluded_atoms_ids:
            continue
        radii = table.GetRvdw(atomic_num)
        translated_points = points - sphere_center
        distances = np.linalg.norm(translated_points, axis=1)
        points_inside = points_inside | (distances <= radii)
    if count_per_axis == None:
        return points_inside
    count_num = count_per_axis[0] * count_per_axis[1] * count_per_axis[2]
    counts = np.zeros(count_num, dtype=np.int32)
    counts_num_each = np.zeros(count_num, dtype=np.int32)
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

# Model 
def normalize_axis(arr, axis=0, mean=[], std=[]):
    """
    Performs z-score normalization on a specific axis of the array
    
    Parameters:
    arr: ndarray, input array
    axis: int, axis to normalize
    
    Returns:
    normalized_arr: ndarray, normalized array
    """
    if len(mean) == 0 or len(std) == 0:
        mean = np.mean(arr, axis=axis, keepdims=True)  # Compute mean
        std = np.std(arr, axis=axis, keepdims=True)  # Compute standard deviation
    normalized_arr = (arr - mean) / std  # Normalize
    normalized_arr = np.nan_to_num(normalized_arr, 0)
    return normalized_arr, mean, std

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