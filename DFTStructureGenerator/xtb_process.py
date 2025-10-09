import os, shutil, glob
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from . import FormatConverter
from . import mol_manipulation 


def xtb_to_mol(mol, atoms, positions, conf_limit):
    """把原子坐标更新到mol分子里面，删除其中原有的构象，新增加一批构象

    Args:
        mol (Chem.Mol): 原始的mol分子
        atoms (list): 元素符号的列表
        positions (__iterable__): 多个原子坐标(m*n*3)
        conf_limit (int): 数量限制，只读取前conf_limit个坐标

    Returns:
        Chem.Mol: 带有更新坐标的分子
    """    
    num_conformers = mol.GetNumConformers()
    for i in range(num_conformers):
        mol.RemoveConformer(i)
    for idx in range(conf_limit):
        conf = Chem.rdchem.Conformer(mol.GetNumAtoms())
        conf.SetId(idx)
        for i, atom in enumerate(mol.GetAtoms()):
            assert atom.GetSymbol() == atoms[idx][i]
            conf.SetAtomPosition(i, positions[idx][i][:3])
        mol.AddConformer(conf)
    return mol

def xtb_update_mol(mol, xtbfile, conf_limit = 3, rmsd_limit=1.5):
    """从xtb输出文件(.xyz)中读取构象，根据设定的rmsd阈值和设定的构象数量限制，更新构象

    Args:
        mol (Chem.Mol): 
        xtbfile (str): xtb输出文件(.xyz)
        conf_limit: 数量限制，只读取前conf_limit个坐标

    Returns:
        mol: Chem.Mol with new positions
    """    


    atoms, positions = read_xyz(xtbfile) 
    assert len(atoms) == len(positions)
    if conf_limit > len(atoms):
        conf_limit = len(atoms)
    first_conf = 0 
    new_mol = copy.deepcopy(mol)
    new_mol = xtb_to_mol(new_mol, atoms, positions, conf_limit)
    if conf_limit == 1:
        return new_mol
    else:
        save_conf_id = [0]
        for i in range(1, conf_limit):
            if min([AllChem.GetConformerRMS(new_mol, i, conf_id) for conf_id in save_conf_id]) > rmsd_limit:
                save_conf_id.append(i)
        atoms = [atoms[i] for i in save_conf_id]
        positions = [positions[i] for i in save_conf_id]
        mol = xtb_to_mol(mol, atoms, positions, len(save_conf_id))
    return mol

def xtb_other_atoms(rest_atoms, all_num):
    """shift 1,2,3,4,6,7,8 into 1-4, 6-8
    Args:
        rest_atoms (_type_): _description_
        all_num (_type_): _description_
    """    
    else_atoms = [each for each in range(1,all_num + 1) if each not in rest_atoms]
    else_atoms.append(-1)
    return_str = ''
    temp_str = ''
    for atom_id, else_atom in enumerate(else_atoms[:-1]):
        if len(temp_str) == 0:
            temp_str += str(else_atom)
        if else_atoms[atom_id + 1] - else_atom != 1:
            if temp_str[0] != str(else_atom):
                temp_str += "-" 
                temp_str += str(else_atom)
            if atom_id != len(else_atoms) -2:
                temp_str += ','
            return_str += temp_str
            temp_str = ''
    return return_str

def xtb_write_xyz(mol, atom_list=None, position_list=None, xtb_dir='xtb_process/', smiles_name='test', dist_rest=None):
    """
    生成xtb优化文件：对mol的每一个构象，在id_%.8d_%d的文件夹中生成xtb优化文件
    parents of mol_to_xyz()

    Args:
        mol (str): mol
        xtb_dir (str, optional): parnets dir of charges files. Defaults to 'xtb_process/'.
        smiles_name (str, optional): name for charge files. Defaults to 'test'.
        dist_rest (list): Such as [[a,b], [c,d]], restrict the distance between atomIds

    Returns:
        file_dir: 
    """    
    if not os.path.isdir(xtb_dir):
        os.mkdir(xtb_dir)
    xyz_files = FormatConverter.mol_to_xyz(mol, atom_list, position_list, file_dir=xtb_dir + "%s.xyz" % smiles_name)
    file_dirs = []
    for eachfile in xyz_files:
        new_path = eachfile.split(".")[0] + "/"
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        shutil.move(eachfile, new_path + os.path.split(eachfile)[1])
        # if mol is not None:
        #     Chem.MolToMolFile(mol,new_path + "/%s.mol" % smiles_name, )
        if dist_rest is not None:
            if mol is None:
                atom_num = len(atom_list)
            else:
                atom_num = mol.GetNumAtoms()
            with open(new_path + '/.constrains', "wt", newline="\n") as f:
                f.write("$constrain\n")
                f.write("force constant=0.5\n")
                f.write("reference=" + os.path.split(eachfile)[1] + '\n')
                rest_atoms = []
                for (a,b) in dist_rest:
                    f.write("distance: %d, %d, auto\n" % (a,b))
                    rest_atoms += [a,b]
                rest_atoms = list(set(rest_atoms))
                else_str = xtb_other_atoms(rest_atoms, atom_num)
                f.write("$metadyn\natoms: ")
                f.write(else_str + '\n')
                f.write("$end\n")
        file_dirs.append(new_path + os.path.split(eachfile)[1])
    return file_dirs

def write_xtb_pbs(pbs_path, charge, que="gamma", root_dir='charg_0', uhf=0):
    """xtb运行的shell脚本生成工具，适用于PBS队列的超算

    Args:
        pbs_path (_type_): 脚本的文件地址
        charge (_type_): Charge of Mols.
        que (str, optional): _description_. Defaults to "gamma".
        root_dir (str, optional): 存放同时计算的Xtb任务的父目录. Defaults to 'charg_0'.
    """    
    with open(pbs_path, "wt", newline="\n") as f:
        f.write("#!/bin/bash\n#PBS -l nodes=1:ppn=28\n#PBS -l walltime=99:00:00\n#PBS -N crest-\n#PBS -q %s\n#PBS -j oe\n#PBS -o jobID.$PBS_JOBID\n" % que)
        f.write("export OMP_NUM_THREADS=28\nexport MKL_NUM_THREADS=28\nexport OMP_STACKSIZE=6000m\n\n")
        f.write("cd $PBS_O_WORKDIR\ntouch jobID.$PBS_JOBID\nrootdir=%s\n" % root_dir)
        f.write("folders=`ls $PBS_O_WORKDIR/$rootdir/`\nfor folder in $folders\ndo\n    cd $PBS_O_WORKDIR/$rootdir/$folder\n")
        f.write("    crest $folder.xyz -T 28 -gfn2 -chrg %d -uhf %d -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\n" % (charge, uhf))
        f.write("done")

def xtb_main(smiles_names, smileses, restrict = None, dir_path='xtb_process', que="gamma", core=1, uhf=0):
    """    !! 主流程
    对于smileses的所有分子，按照电荷进行归类，提交到多个节点。
    suball是总提交脚本，运行其即可

    Args:
        smiles_names: 每一个smiles的识别文件名
        smileses (_type_): smiles or mols
        dir_path (str, optional): root dir saved charge files. Defaults to 'xtb_process'.
        que (str, optional): 超算队列. Defaults to "gamma".
        core(int): 要提交到多少个不同节点计算。
        uhf(int): 单电子数
    """    

    if os.path.isdir(dir_path):
        # raise ValueError("Path has existed")
        pass
    else:
        os.mkdir(dir_path)
    charges = set()
    core_size = 0
    core_id = 0
    each_size = len(smileses) // core + 1
    for i, (smiles_name, smiles) in enumerate(zip(smiles_names, smileses)):
        if type(smiles) == str:
            mol = mol_manipulation.smiles2mol(smiles)
            if mol is None:
                print(smiles)
        elif type(smiles) == Chem.Mol:
            mol = smiles
        charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
        charges.add(charge)
        if charge == 0:
            core_size +=1
            now_core_id = core_id
        else:
            now_core_id = 0
        if core_size >= each_size:
            core_id += 1
            core_size = 0

        # AllChem.MMFFOptimizeMoleculeConfs(mol)
        if restrict == None:
            xtb_write_xyz(mol, xtb_dir=dir_path + "/" + "charg_%d_%d/" % (charge, now_core_id), smiles_name=smiles_name, )
        else:
            xtb_write_xyz(mol, xtb_dir=dir_path + "/" + "charg_%d_%d/" % (charge, now_core_id), smiles_name=smiles_name, dist_rest=restrict[i])
    with open(dir_path + "/suball", "wt", newline='\n') as f:
        for eachcharge in charges:
            if eachcharge == 0:
                for now_core_id in range(core_id + 1):
                    pbs_path = dir_path + "/" + "xtb_%d_%d.pbs" % (eachcharge, now_core_id)
                    write_xtb_pbs(pbs_path, eachcharge, que, root_dir="charg_%d_%d" % (eachcharge, now_core_id), uhf=uhf)
                    f.write("qsub %s\n" % "xtb_%d_%d.pbs" % (eachcharge, now_core_id))
            else:
                pbs_path = dir_path + "/" + "xtb_%d_%d.pbs" % (eachcharge, 0)
                write_xtb_pbs(pbs_path, eachcharge, que, root_dir="charg_%d_%d" % (eachcharge, 0), uhf=uhf)
                f.write("qsub %s\n" % "xtb_%d_%d.pbs" % (eachcharge, 0))

def read_xyz(file_dir):
    """把.xyz文件中所有的原子坐标全部都提取出来
    Args:
        file_dir (str): file_dir endwith .xyz

    Returns:
        atoms: list of atoms
        position: array of position
    """    
    with open(file_dir) as f:
        lines = f.readlines()
    line1 = lines[0]
    mol_num = len([line for line in lines if line == line1])
    atom_list = []
    positions = []
    start_id = 0
    for eachmol in range(mol_num):
        atom_num = int(lines[start_id].strip("\n").split("  ")[-1])
        atoms = []
        position = []
        start_id += 2
        for i in range(atom_num):
            pro_line = lines[start_id].split()
            atoms.append(pro_line[0])
            position.append([float(pro_line[1]), float(pro_line[2]), float(pro_line[3])])
            start_id += 1
        assert len(atoms) == atom_num
        atom_list.append(atoms)
        positions.append(position)
    return atom_list, positions

def after_xtb(mol, xtb_dir="xtb_process", save_dir="xtb_result", conf_limit=3, rmsd_limit=1.5, xtb_title=None, method="opt freq b3lyp/6-31g* em=gd3bj g09def", SpinMultiplicity=None):
    """给定指定的mol分子，在xtb_dir下寻找对应的任务，按照设定的阈值和数量限制产生Gaussian的初始文件

    Args:
        mol (Chem.Mol): mol
        xtb_dir (str, optional): xtb任务的根目录. Defaults to "xtb_process".
        save_dir (str, optional): Gaussian输入文件存储的目录. Defaults to "xtb_result".
        conf_limit (int, optional): 数量限制. Defaults to 3.
        rmsd_limit (float, optional): rmsd阈值. Defaults to 1.5.
        xtb_title (_type_, optional): Gaussian输入文件的title. Defaults to None.
        method (str, optional): Gaussian的方法. Defaults to "opt freq b3lyp/6-31g* em=gd3bj g09def".
        SpinMultiplicity (_type_, optional): 自旋多重度. Defaults to None.
    """
    mol_str = os.path.split(xtb_dir)[-1][:-2]
    charge = int(xtb_dir.split("\\")[-2].split("_")[-2])
    if not os.path.isfile(xtb_dir + "/crest_best.xyz"):
        print("mol_str:%s didn't find crest_best!" % (mol_str))
        # xtb_files = glob.glob(xtb_dir + '/id*.xyz')
        # xtb_update_mol(mol, xtb_files[0])
    else:
        try:
            xtb_update_mol(mol, xtb_dir + "/crest_conformers.xyz", conf_limit, rmsd_limit)
        except:
            print("mol_str:%s have something wrong!" % (mol_str))
    for conf_id in range(len(mol.GetConformers())):
        file_dir = save_dir + "/%s_%.4d.gjf" % (mol_str, conf_id)
        FormatConverter.mol_to_gjf(mol, file_dir, confid=conf_id, method=method, charge=charge, title=xtb_title, SpinMultiplicity=SpinMultiplicity)

def shift_to_sugan(target_file, quene_id = 1, uhf=0):
    """将提交脚本和总提交脚本改成适用于曙光超算合肥中心的格式。

    Args:
        target_file (_type_): Xtb文件的根目录
        quene_id (int, optional): 曙光合肥中心的队列编号. Defaults to 1.
        uhf (int, optional): 单电子数. Defaults to 0.
    """    
    pbs_files = glob.glob(target_file + '/*.pbs')
    for pbs_file in pbs_files:
        chrg = int(pbs_file.split(".pbs")[0].split("_")[-2])
        with open(pbs_file, "wt",  newline='\n') as f:
            number = int(pbs_file.split(".pbs")[0].split("_")[-1])
            f.write("#!/bin/bash\n#SBATCH -J g16\n#SBATCH -N 1\n#SBATCH --ntasks-per-node=32\n#SBATCH -p hfacnormal%.2d\n\nroot=`pwd`\nrootdir=charg_%d_%d\nfolders=`ls $root/$rootdir/`\n" % (quene_id, chrg, number))
            f.write("for folder in $folders\ndo\n    cd $root/$rootdir/$folder\n    crest $folder.xyz -T 28 -gfn2 -chrg %d -uhf %d -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\ndone\n" % (chrg, uhf))
    with open(target_file + '/suball', "wt",  newline='\n') as f:
        for pbs_file in pbs_files:
            name = os.path.split(pbs_file)[-1]
            f.write("sbatch %s\n" % name)
def shift_to_parra(target_file,chrg=0, uhf=0):
    """将提交脚本和总提交脚本改成适用于并行云超算的格式。

    Args:
        target_file (_type_): Xtb文件的根目录
        quene_id (int, optional): 曙光合肥中心的队列编号. Defaults to 1.
        chrg (int, optional): 电荷. Defaults to 0.
        uhf (int, optional): 单电子数. Defaults to 0.
    """    
    pbs_files = glob.glob(target_file + '/*.pbs')
    for pbs_file in pbs_files:
        with open(pbs_file, "wt",  newline='\n') as f:
            number = int(pbs_file.split(".pbs")[0].split("_")[-1])
            f.write("#!/bin/bash\n#SBATCH -p amd_512\n#SBATCH -N 1\n#SBATCH -n 1\n#SBATCH -c 28\n\nroot=`pwd`\nrootdir=charg_0_%d\nfolders=`ls $root/$rootdir/`\n" % (number))
            f.write("for folder in $folders\ndo\n    cd $root/$rootdir/$folder\n    crest $folder.xyz -T 28 -gfn2 -chrg %d -uhf %d -rthr 0.25 -shake 1 --mdlen 0.5 > crest.out\ndone\n" % (chrg, uhf))
    with open(target_file + '/suball', "wt",  newline='\n') as f:
        for pbs_file in pbs_files:
            name = os.path.split(pbs_file)[-1]
            f.write("sbatch %s\n" % name)
