import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rdkit import Chem
import pandas as pd
import os, shutil
from tqdm import tqdm
from DFTStructureGenerator import DFThandle, xtb_process, mol_manipulation, gendes, logfile_process, Tool

def Calc_AEIF_(mol, charges, excluded_atoms_ids=[], num_per_axis=20, radius=8, count_per_axis = [2,2,2]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    charge_energy = np.zeros(total_points, dtype=bool)
    # 计算每个点到每个原子中心的距离
    for atom_id, (sphere_center, atom_charge) in enumerate(zip(geom, charges)):
        if atom_id in excluded_atoms_ids:
            continue
        translated_points = (points - sphere_center)
        distances = np.linalg.norm(translated_points, axis=1)
        charge_energy = charge_energy + atom_charge / distances
    # 归一化电荷能
    normal_charge_energy = (charge_energy - np.min(charge_energy)) / (np.max(charge_energy) - np.min(charge_energy))
    # 绘制格点
    target_idx = [idx for idx, charge_energy_ in enumerate(charge_energy) if charge_energy_ > 0.1]
    xs, ys, zs = zip(*points[target_idx])
    # ax.scatter(xs, ys, zs, c=[(each, 1 - each, 0) for each in normal_charge_energy[target_idx]], s=10)

    # 设置坐标轴范围
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    line_a = np.linspace(- radius, radius, 2)
    one_ = np.ones_like(line_a)
    empty = np.zeros_like(line_a)
    # ax.plot(line_a, empty, empty, color='black', linestyle='--')
    # ax.plot(empty, line_a, empty, color='black', linestyle='--')
    # ax.plot(empty, empty, line_a, color='black', linestyle='--')
    a = [one_ * (- radius + 2 * each * radius / count_per_axis[0]) for each in range(count_per_axis[0] + 1)]
    b = [one_ * (- radius + 2 * each * radius / count_per_axis[1]) for each in range(count_per_axis[1] + 1)]
    c = [one_ * (- radius + 2 * each * radius / count_per_axis[2]) for each in range(count_per_axis[2] + 1)]
    for each_b in b:
        for each_c in c:
            ax.plot(line_a, each_b, each_c, color='gray', linestyle='--')
    for each_a in a:
        for each_c in c:
            ax.plot(each_a, line_a, each_c, color='gray', linestyle='--')
    for each_a in a:
        for each_b in b:
            ax.plot(each_a, each_b, line_a, color='gray', linestyle='--')
    # 绘制坐标轴平面
    # 显示图形
    ax.axis("off")
    # plt.savefig("test.svg", useSVG=True)
    plt.show()

new_mol_dir = "Data/newmols"
new_dft_dir = "Data/newGS"
smiles_csv = pd.read_csv("Data_clear_with_sites.csv")
# smiles_csv = smiles_csv.loc[(smiles_csv['Type'] == "Binol")]
aeif_dict = {}
for row_id, row in tqdm(smiles_csv.iterrows()):
    ligand_idx = row["Index"]
    conf_id = row["conf_id"]
    sites = [int(each) for each in row['Sites'].split()]
    mol = Chem.MolFromMolFile(os.path.join(new_mol_dir, f"{ligand_idx:05}.mol"), removeHs=False)
    log = logfile_process.Logfile(os.path.join(new_dft_dir, f"{ligand_idx:05}.log"))
    symbol_list, positions = log.symbol_list, log.running_positions[-1]
    charges = log.read_charge_spin_density()[0]
    cu_atom_id = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Cu"][0]
    out_position = mol_manipulation.trfm_rot(positions[sites[0]], positions[sites[-1]], positions[cu_atom_id], positions)
    if out_position[cu_atom_id][1] < 0:
        out_position = out_position @ mol_manipulation.rotation([1,0,0], 0, -1)
    mol = xtb_process.xtb_to_mol(mol, [symbol_list], [out_position], 1)
    areas = Calc_AEIF_(mol, charges, sites + [cu_atom_id], radius = 8,num_per_axis=20, count_per_axis = [2,2,2])
    aeif_dict[ligand_idx] = areas
    break