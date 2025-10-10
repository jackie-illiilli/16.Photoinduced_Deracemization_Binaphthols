"""
Module for reading and parsing Gaussian output files (.log files).
Handles extraction of molecular structures, energies, frequencies, and other computational chemistry data.
"""

from copy import deepcopy
import numpy as np
import os
import glob
import shutil
from rdkit import Chem
# from sterimol import sterimoltools

from . import Tool, FormatConverter


class bond_addition_function:
    """
    Utility class for defining and applying constraints on bond lengths, angles, or dihedrals
    during molecule validation, such as checking if a conformation satisfies specific geometric criteria.
    """

    def __init__(self):
        """
        Initialize the constraint dictionary and index counter.
        """
        self.addition = {"atoms": {}, "value": {}, "type": {}}
        self.addition_idx = 0

    def compare(self, value1, value2, type_):
        """
        Compare two values based on the specified type (more, less, or equal).

        Args:
            value1 (float): First value to compare.
            value2 (float): Second value to compare.
            type_ (str): Comparison type: "more" (>), "less" (<), or "equal" (approx. ==).

        Returns:
            bool: True if the comparison holds.
        """
        if type_ == "more":
            return value1 > value2
        elif type_ == "less":
            return value1 < value2
        elif type_ == "equal":
            return abs(value1 - value2) < 1e-6

    def add_function(self, atom_title_ids, value, type_):
        """
        Add a new geometric constraint (bond length, angle, or dihedral).

        Args:
            atom_title_ids (list): List of atom indices involved in the constraint.
            value (float): Target value for the constraint (e.g., bond length in Angstroms).
            type_ (str): Constraint type: "more", "less", or "equal".
        """
        self.addition["atoms"][self.addition_idx] = atom_title_ids
        self.addition["type"][self.addition_idx] = type_
        self.addition["value"][self.addition_idx] = value
        assert type_ in ["more", "less", "equal"]

        self.addition_idx += 1

    def apply_addition(self, atom_idx_list, position):
        """
        Apply all defined constraints to check if the current conformation satisfies them.

        Args:
            atom_idx_list (list): Mapping of title indices to atom indices.
            position (np.ndarray): 3D positions of atoms (n_atoms x 3).

        Returns:
            bool: True if all constraints are satisfied.
        """
        for addition_idx in range(self.addition_idx):
            atom_ids = [atom_idx_list[each] for each in self.addition["atoms"][addition_idx]]
            conf = Chem.rdchem.Conformer(len(atom_ids))
            for i, atom_id in enumerate(atom_ids):
                conf.SetAtomPosition(i, position[atom_id][:3])
            if len(atom_ids) == 2:
                bond_length = Chem.rdMolTransforms.GetBondLength(conf, 0, 1)
                if not self.compare(bond_length, self.addition["value"][addition_idx],
                                    self.addition["type"][addition_idx]):
                    return False
            elif len(atom_ids) == 3:
                bond_angle = Chem.rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
                if not self.compare(bond_angle, self.addition["value"][addition_idx],
                                    self.addition["type"][addition_idx]):
                    return False
            elif len(atom_ids) == 4:
                dihedral_angle = Chem.rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
                if not self.compare(dihedral_angle, self.addition["value"][addition_idx],
                                    self.addition["type"][addition_idx]):
                    return False
        return True


class Logfile:
    """
    Class for parsing Gaussian .log output files.
    Extracts structural, energetic, and vibrational information.
    Supports various job types: SPE (single point energy), OPT (optimization), TS (transition state),
    IRC (intrinsic reaction coordinate), OM (on-the-fly modified redundant).
    """

    def __init__(self, file_dir, mol_file_dir=None, read_title=True, freq_warning=False,
                 bond_attach_std='mol', bond_addition_function=None, bond_ignore_list=None):
        """
        Initialize the Logfile parser.

        Args:
            file_dir (str): Path to the .log output file.
            mol_file_dir (str, optional): Path to corresponding .mol file for bond validation. Defaults to None.
            read_title (bool, optional): Whether to read title information using the specified template. Defaults to True.
            freq_warning (bool, optional): Whether to warn about imaginary frequencies. Defaults to False.
            bond_attach_std (str, optional): Standard for bond attachment check ('mol' or other). Defaults to 'mol'.
            bond_addition_function (bond_addition_function, optional): Custom function for additional bond constraints.
            bond_ignore_list (list, optional): List of bonds to ignore in attachment checks.
        """
        self.file_dir = file_dir
        self.mol_file_dir = mol_file_dir
        with open(file_dir, "rt") as rf:
            filelines = rf.readlines()
            filelines = [line for line in filelines if line != ""]
        self.filelines = filelines
        self.normal_end = self.is_normal_end()
        if not self.normal_end:
            self.find_error_reason()
        # self.S_2 = self.read_S_2()
        self.title = self.read_title()
        self.charge, self.multiplicity = self.read_charge_multiplicity()
        # Special for Charge != 0
        # if self.charge != 0:
        #     self.normal_end = False
        #     self.charge = 0
        #     self.error_reason = "link 9999"

        if self.multiplicity == -1:
            print("It's not a logfile")
            return None

        self.method = self.read_method().lower()
        self.file_type = 'SPE'
        if "irc" in self.method:
            self.file_type = 'IRC'
        elif "modredundant" in self.method:
            self.file_type = "OM"
        elif "readfc" in self.method or "calcfc" in self.method:
            self.file_type = "TS"
        elif "opt" in self.method:
            self.file_type = "OPT"

        self.symbol_list, self.first_atom_position = self.read_first_position()
        if self.symbol_list == None:
            print("It's a wrong file with unknown wrong")
            return None

        if self.file_type != "SPE":
            self.running_positions = self.read_running_position()
            if self.running_positions is not None and len(self.running_positions) != 0:
                self.running_rmsd = self.react_RMSD()
            if self.normal_end and "freq" in self.method:
                self.unreal_freq, self.unreal_freq_matrix, self.first_unreal_freq = self.read_unreal_freq(freq_warning)
            else:
                self.unreal_freq = -1
                self.unreal_freq_matrix = []
        else:
            self.running_positions, self.unreal_freq, self.unreal_freq_matrix = [], 0, []

        if self.file_type == "OM":
            self.freeze, self.difreeze = self.read_freeze()
        else:
            self.freeze, self.difreeze = [], []

        self.all_engs, self.opt_engs = self.read_log_eng()
        if self.normal_end:
            self.running_time = self.read_log_time()
        else:
            self.all_engs = []
            self.opt_engs = []
            self.running_time = 0
        if self.mol_file_dir != None and self.file_type != "SPE" and len(self.running_positions) != 0:
            self.bond_attach = self.check_bond_attach(standard_file=bond_attach_std,
                                                      bond_addition_function=bond_addition_function,
                                                      bond_ignore_list=bond_ignore_list)
        else:
            self.bond_attach = True

        if self.file_type == "IRC":
            self.irc_result = self.irc_check()

        if self.file_type == "TS":
            if self.unreal_freq != 1:
                print("%s is not a right TS for unreal freq num of %d" % (self.file_dir, self.unreal_freq))
                self.is_right_ts = False
            else:
                self.is_right_ts = self.check_om(False)
                pass

    # def __repr__(self):
    #     print("Logfile Name: %s" % self.file_dir)
    #     print("Run Successful %s" % bool(self.normal_end))
    #     print("Logfile Type as %s" % self.file_type)
    #     print("Title: %s" % ' '.join([str(each) for each in self.title]))
    #     print("Charge: %d, Multiplicity: %d" % (self.charge, self.multiplicity))
    #     print("Symbol list:%s" % ' '.join(self.symbol_list))
    #     print("Number of Unreal Freq: %d" % self.unreal_freq )
    #     print("Method: %s" % self.method)
    #     print("Optimization Steps: %d" % (len(self.running_positions) - 1))
    #     print("Optimization Engs: %s" % ' '.join([str(each) for each in self.opt_engs]))
    #     print("Final Engs: %s" % ' '.join([str(each) for each in self.all_engs]))
    #     print("Elapse Time: %.3f min" % self.running_time)

    #     if self.file_type == 'IRC':
    #         print("IRC File Successful: %s" % bool(self.irc_result))
    #     if self.file_type == 'TS':
    #         print("TS File is Right: %s" % bool(self.is_right_ts))
    #     return "End"

    def is_normal_end(self):
        """
        Check if the Gaussian job terminated normally.

        Returns:
            bool: True if "Normal termination of Gaussian" is found in the last line.
        """
        lastline = self.filelines[-1]
        if lastline.find(" Normal termination of Gaussian") == -1:
            print("%s didn't run successful" % self.file_dir)
            return False
        else:
            return True

    def read_title(self):
        """
        Read title information from the log file using the template $$$$Title####charge????.

        Returns:
            list: List of title values (integers if applicable).
        """
        allfile = "".join(self.filelines)
        title = ""
        charge = 0
        # title = $$$$Title####charge????
        if allfile.find("$$$$") == -1 or allfile.find("####") == -1 or allfile.find("????") == -1:
            title = ""
            print("Can't find title")
        else:
            title = allfile[allfile.find("$$$$") + 4: allfile.find("####")]
            charge = allfile[allfile.find("####") + 4: allfile.find("????")]
            title = title.split()
            if len(title) > 1:
                title = [int(each) for each in title]
        return title

    def read_charge_multiplicity(self):
        """
        Extract charge and spin multiplicity from the log file.

        Returns:
            tuple: (charge: int, multiplicity: int). Returns (0, -1) if not found.
        """
        line_id, line = Tool.find_first_line(self.filelines, 'Charge = ', 'in')
        if line_id is None:
            print("Can't find charge and multiplicity")
            return 0, -1
        charge = int(line.split()[2])
        multiplicity = int(line.split()[-1])
        return charge, multiplicity

    def read_S_2(self):
        """
        Extract the <S**2> value (spin contamination) from the log file.

        Returns:
            float: <S**2> value, or -1 if not found.
        """
        lines = [[idx, each] for idx, each in enumerate(self.filelines) if "<S**2>" in each]
        if len(lines) == 0:
            return -1
        else:
            _, line_ = lines[-1]
            line_ = line_.strip("\n").split()
            idx = [idx for idx, each in enumerate(line_) if "S**2" in each][0]
            return float(line_[idx + 1])

    def read_first_position(self):
        """
        Read atomic symbols and initial coordinates from the "Symbolic Z-matrix" section.
        Falls back to running positions if not found.

        Returns:
            tuple: (symbol_list: list of str, positions: np.ndarray of shape (n_atoms, 3)).
                   Returns (None, None) if structure not found.
        """
        start_id = Tool.find_first_line(self.filelines, "Symbolic Z-matrix:", 'in')[0]
        if start_id != None:
            start_index = start_id + 2
            end_index = Tool.find_first_line(self.filelines[start_index:], ' \n', 'all')[0] + start_index
            if end_index is None:
                print("%s didn't have structure" % self.file_dir)
                return None, None
            # start_index = [i for i, line in enumerate(self.filelines) if line.find("Symbolic Z-matrix:") >= 0][0] + 2
            # end_index = [i + start_index for i, line in enumerate(self.filelines[start_index:]) if line == ' \n'][0]
            orientation = [line.split() for line in self.filelines[start_index: end_index]]
            symbol_list, position = [], []
            for each in orientation:
                if len(each) != 4:
                    return None, None
                symbol_list.append(each[0])
                position.append([float(each[1]), float(each[2]), float(each[3])])
            assert len(symbol_list) == len(position)
            return symbol_list, np.array(position)
        else:
            try:
                symbol, positions = self.read_running_position(read_first=1)
                return symbol, positions
            except:
                print("%s didn't have structure" % self.file_dir)
                return None, None

    def read_running_position(self, read_first=False):
        """
        Extract atomic positions from optimization steps or input orientation.
        The last position is the converged structure.

        Args:
            read_first (bool, optional): If True, return only the first position and symbols. Defaults to False.

        Returns:
            np.ndarray: Array of positions (n_steps x n_atoms x 3), or (symbols, first_positions) if read_first=True.
                        Returns None if not found.
        """
        orientation_sign = "Input orientation:"
        # if self.file_type == 'IRC' or read_first:
        #     orientation_sign = "Input orientation:"
        # else:
        #     orientation_sign = "Standard orientation:"
        start_indexs = [i for i, line in enumerate(self.filelines) if line.find(orientation_sign) >= 0]
        if len(start_indexs) == 0:
            orientation_sign = "Standard orientation:"
            start_indexs = [i for i, line in enumerate(self.filelines) if line.find(orientation_sign) >= 0]
        if len(start_indexs) == 0:
            print("%s Even not Input Structure, wrong file maybe" % self.file_dir)
            return None
        all_positions = []
        for each_start_index in start_indexs:
            start_index = each_start_index + 5
            end_index = Tool.find_first_line(self.filelines[start_index:], ' ---', 'in')[0] + start_index
            if end_index is None:
                break
            orientation = [line.strip("\n").split() for line in self.filelines[start_index: end_index]]
            position = []
            symbol = []
            for each in orientation:
                if len(each) != 6:
                    return all_positions
                position.append([float(each[3]), float(each[4]), float(each[5])])
                symbol.append(int(each[1]))
            all_positions.append(position)

        if read_first:
            return symbol, all_positions[0]
        else:
            if len(all_positions) == 1:
                return np.array(all_positions)
            return np.array(all_positions)[1:]

    def read_method(self):
        """
        Extract the computational method from the # route section.

        Returns:
            str: Method string (e.g., "opt freq b3lyp/6-31g(d)").
        """
        method_id, method = Tool.find_first_line(self.filelines, ' #p', 'start')
        if method is None:
            method_id, method = Tool.find_first_line(self.filelines, ' #', 'start')
            if method is None:
                print("%s with not method" % self.file_dir)
                return None
        method_final_line_id, _ = Tool.find_first_line(self.filelines[method_id:], " -------", 'start')
        method = "".join([each.strip("\n").lstrip(" ") for each in self.filelines[method_id: method_id + method_final_line_id]])
        method = method.split('#p ')[-1]
        return method

    def read_unreal_freq(self, freq_warning=True):
        """
        Detect imaginary (unreal) frequencies and extract the number, mode matrix, and first frequency value.

        Args:
            freq_warning (bool, optional): Whether to print a warning if imaginary frequencies are found. Defaults to True.

        Returns:
            tuple: (num_unreal_freq: int, unreal_freq_matrix: list of lists, first_unreal_freq: str).
                   Returns (-1, []) if frequencies not calculated.
        """
        start_index = [i for i, line in enumerate(self.filelines) if '(negative Signs)' in line]
        if len(start_index) != 0:
            start_index = start_index[-1]
        else:
            start_index = 0
        smallest_freq_index, smallest_freq_line = Tool.find_first_line(self.filelines[start_index:], ' Frequencies --', "start")
        smallest_freq_index += start_index
        if smallest_freq_index == None:
            print("%s didn't calc freq" % self.file_dir)
            return -1, []
        smallest_freq_list = smallest_freq_line.strip("\n").split()[2:]
        num_unreal_freq = sum([1 for each in smallest_freq_list if float(each) < 0])
        # freq_fileline = [[lid, Tool.remove_space(line.strip("\n"))[2]] for lid, line in enumerate(self.filelines) if line.startswith(" Frequencies --")][0]
        matrix = []
        start_id = smallest_freq_index + 5
        while True:
            line = self.filelines[start_id].strip("\n").split()
            if len(line) < 5:
                break
            line = [float(each) for each in line[2:5]]
            matrix.append(line)
            start_id += 1
        if freq_warning:
            print("%s have unreal freq" % self.file_dir)
        return num_unreal_freq, matrix, smallest_freq_list[0]

    def read_log_eng(self):
        """
        Extract energy information: SCF energies from optimization steps, and thermal corrections.

        Returns:
            tuple: (all_engs: list of floats [SCF, ZPE, thermal E, enthalpy, Gibbs],
                    opt_engs: list of SCF energies from opt steps).
        """
        all_engs = []
        opt_engs = []
        ee_line = [[idx, each] for idx, each in enumerate(self.filelines) if " SCF Done: " in each]
        if len(ee_line) == 0:
            print("%s, can't find any engs" % self.file_dir)
            return all_engs, opt_engs
        for line_id, each_ee_line in ee_line:
            # ee_line = [line for i, line in enumerate(
            #     self.filelines[start_index:]) if line.startswith(" SCF Done: ")][-1]
            if each_ee_line == None:
                ee = -1
            else:
                ee = float(each_ee_line.strip("\n").split()[4])
            opt_engs.append(ee)
        try:
            start_idx = ee_line[-1][0]
            zpc_line = Tool.find_first_line(self.filelines[start_idx:], " Zero-point correction=", "in")[-1]
            zpc = zpc_line.strip("\n").split(" ")[-2]
            cor_ee = Tool.find_first_line(self.filelines[start_idx:], " Thermal correction to Energy=", "in")[-1].strip("\n").split(" ")[-1]
            cor_Enthalpies = Tool.find_first_line(self.filelines[start_idx:], " Thermal correction to Enthalpy=", "in")[-1].strip("\n").split(" ")[-1]
            cor_Gibbs = Tool.find_first_line(self.filelines[start_idx:], " Thermal correction to Gibbs Free Energy=", "in")[-1].strip("\n").split(" ")[-1]
            all_engs = [ee, zpc, cor_ee, cor_Enthalpies, cor_Gibbs]
            all_engs = [float(each) for each in all_engs]
        except:
            return [ee], opt_engs
        return all_engs, opt_engs

    def read_log_time(self):
        """
        Calculate total elapsed time in minutes from all "Elapsed time:" lines.

        Returns:
            float: Total time in minutes.
        """
        alltime = 0
        for each in self.filelines:
            if each.startswith(" Elapsed time:"):
                times = list(each.strip("\n").split())
                alltime += float(times[2]) * 24 * 60 + float(times[4]) * 60 + float(times[6]) + float(times[8]) / 60
        return alltime

    def read_freeze(self):
        """
        Extract frozen bonds and dihedrals from ModRedundant section.

        Returns:
            tuple: (freeze: list of [atom1, atom2] for bonds, difreeze: list of [atom1,2,3,4] for dihedrals).
        """
        freeze_start_line_id, _ = Tool.find_first_line(self.filelines, "The following ModRedundant", "in")
        freeze, difreeze = [], []
        for line in self.filelines[freeze_start_line_id + 1:]:
            line = line.strip('\n').split()
            if len(line) < 4:
                break
            if line[0] == "B":
                freeze.append([int(line[1]), int(line[2])])
            elif line[0] == "D":
                difreeze.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
        return freeze, difreeze

    def find_error_reason(self):
        """
        Identify the reason for job failure from the error termination line.
        """
        errorline_id = [i for i, line in enumerate(self.filelines) if " Error termination" in line]
        # errorline_id = Tool.find_first_line(self.filelines,"start")[0]
        if len(errorline_id) == 0 and not self.normal_end:
            print(self.file_dir, "应该是没跑完")
            error_reason_line = "unfinished"
        else:
            errorline_id = errorline_id[-1]
            error_reason_line = self.filelines[errorline_id - 1]
            error_reason_line = error_reason_line.strip('.\n')
        self.error_reason = error_reason_line

    def modify_method(self, old_method, addition_part='opt', new_word="maxcycles=100"):
        """
        Modify the route section to add or update keywords (e.g., increase maxcycles).

        Args:
            old_method (str): Original method string.
            addition_part (str, optional): Keyword to target (e.g., 'opt'). Defaults to 'opt'.
            new_word (str, optional): New keyword to add (e.g., "maxcycles=100"). Defaults to "maxcycles=100".

        Returns:
            str: Updated method string.
        """
        method = old_method.split()
        opt_part = [[idx, each] for idx, each in enumerate(method) if addition_part in each]
        if len(opt_part) == 0:
            method = old_method
        else:
            opt_id = opt_part[0][0]
            opt_part = opt_part[0][1]
            if new_word.split("=")[0] in opt_part:
                method = old_method
            else:
                if "=" not in opt_part:
                    opt_part += f"=({new_word})"
                elif "(" in opt_part:
                    opt_part = opt_part.replace(")", f",{new_word})")
                else:
                    opt_part = opt_part.replace("=", "=(")
                    opt_part += f",{new_word})"
                method[opt_id] = opt_part
                method = " ".join(method)
        return method

    def solve_error_logfile(self, new_log_dir, Inv_dir, yqc_dir, savechk=None, readchk=None, maxcycles=0, method=None):
        """
        Handle failed jobs by generating a new input file with the lowest-energy structure
        and moving the log file. Supports specific error types like L103, convergence failure, etc.

        Args:
            new_log_dir (str): Directory for new log and gjf files.
            Inv_dir (str): Directory for inversion errors.
            yqc_dir (str): Directory for quadratic convergence jobs.
            savechk (str, optional): Path for %chk save.
            readchk (str, optional): Path for %chk read.
            maxcycles (int, optional): Max optimization cycles for restart.
            method (str, optional): Custom method for restart.
        """
        reason = self.error_reason
        if not os.path.isdir(new_log_dir):
            os.mkdir(new_log_dir)
        new_gjf_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1].split(".")[0] + '.gjf'
        if self.file_type in ["SPE", "IRC"]:
            return 0
        if "FormBX" in reason or "Linear angle" in reason or "Tors failed for dihedral" in reason:
            new_position = self.solve_l103_problem()
            title = " ".join(str(each) for each in self.title)
            FormatConverter.block_to_gjf(self.symbol_list, new_position, new_gjf_name, self.charge, self.multiplicity,
                                         title, self.method, freeze=self.freeze, difreeze=self.difreeze,
                                         savechk=savechk, readchk=readchk)
            new_log_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1]
            shutil.move(self.file_dir, new_log_name)
        elif "Convergence failure" in reason:
            title = " ".join(str(each) for each in self.title)
            new_log_name = yqc_dir + "/" + os.path.split(self.file_dir)[-1]
            new_gjf_name = yqc_dir + "/" + os.path.split(self.file_dir)[-1].split(".")[0] + '.gjf'
            if not os.path.isdir(yqc_dir):
                os.mkdir(yqc_dir)
            method = self.method + " scf=yqc"
            FormatConverter.block_to_gjf(self.symbol_list, self.running_positions[-1], new_gjf_name, self.charge,
                                         self.multiplicity, title, method, freeze=self.freeze, difreeze=self.difreeze,
                                         savechk=savechk, readchk=readchk)
            new_log_name = yqc_dir + "/" + os.path.split(self.file_dir)[-1]
            shutil.move(self.file_dir, new_log_name)
        elif "Inv3" in reason:
            title = " ".join(str(each) for each in self.title)
            new_gjf_name = Inv_dir + "/" + os.path.split(self.file_dir)[-1].split(".")[0] + '.gjf'
            new_log_name = Inv_dir + "/" + os.path.split(self.file_dir)[-1]
            if not os.path.isdir(Inv_dir):
                os.mkdir(Inv_dir)
            shutil.move(self.file_dir, new_log_name)

        else:
            print("%s. Error Reason is link 9999 or unfinished." % self.file_dir)
            title = " ".join(str(each) for each in self.title)
            if len(self.running_positions) != 0:
                opt_engs = deepcopy(self.opt_engs)
                idx = len(self.running_positions) - 1
                for _ in range(len(opt_engs)):
                    # try:
                    #     min_index =np.argmin(opt_engs)
                    #     if min_index < len(self.running_positions) - 1:
                    #         opt_engs[min_index] = 0
                    #         continue
                    #     if self.mol_file_dir != None:
                    #         bond_result = self.check_bond_attach('log', conf_id=min_index)
                    #         if bond_result:
                    #             target_position = self.running_positions[min_index]
                    #             break
                    #     else:
                    #         target_position = self.running_positions[min_index]
                    #         break
                    #     opt_engs[min_index] = 0
                    # except:
                    #     opt_engs[min_index] = 0
                    try:
                        target_position = self.running_positions[idx]
                        break
                    except:
                        idx -= 1
                        target_position = self.first_atom_position
            else:
                target_position = self.first_atom_position
            if method == None:
                method = self.method
                if maxcycles != 0:
                    method = self.modify_method(method, new_word=f'maxcycles={maxcycles}')
            new_log_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1]
            shutil.move(self.file_dir, new_log_name)
            if self.file_type == "TS":
                FormatConverter.block_to_gjf(self.symbol_list, target_position, new_gjf_name, self.charge,
                                             self.multiplicity, title, method, freeze=self.freeze,
                                             difreeze=self.difreeze, savechk=savechk, readchk=readchk)
            else:
                FormatConverter.block_to_gjf(self.symbol_list, target_position, new_gjf_name, self.charge,
                                             self.multiplicity, title, method, freeze=self.freeze,
                                             difreeze=self.difreeze, savechk=savechk, readchk=readchk)

    def l103_error_idx(self):
        """
        Identify atom indices involved in L103 errors (linear angles or failed torsions).

        Returns:
            tuple: (angle_idx: np.ndarray of shape (3,), dihedral_idx: np.ndarray of shape (n, 4)).
        """
        angle_idx = np.zeros(3)
        dihedral_idx = np.zeros((0, 4))
        for line in self.filelines[-15:-5]:
            line = line.strip('\n').strip(' ')
            if 'Bend failed for angle' in line:
                ww = line.split()
                angle_idx = np.array([ww[4], ww[6], ww[8]], dtype=int)

            elif 'Tors failed for dihedral' in line:
                ww = line.split()
                tmp_list = np.array([[ww[4], ww[6], ww[8], ww[10]]], dtype=int)
                dihedral_idx = np.append(dihedral_idx, tmp_list, axis=0)

            elif 'Linear angle in Tors.' in line:
                dihedral_idx = np.zeros((1, 4))
        if not dihedral_idx.shape[0]:
            dihedral_idx = np.zeros((1, 4))
        dihedral_idx = dihedral_idx.astype('int', copy=False)
        angle_idx = angle_idx.astype('int', copy=False)
        if not angle_idx.all() and dihedral_idx.all():
            tmp_idx = dihedral_idx[0]
            if dihedral_idx.shape[0] == 1:
                angle_idx = tmp_idx[1:]
            else:
                if (dihedral_idx[:, :3] == tmp_idx[:3]).all():
                    angle_idx = tmp_idx[:3]
                else:
                    angle_idx = tmp_idx[1:]
        return angle_idx, dihedral_idx

    def l103_adjust(self):
        """
        Adjust atomic positions to resolve L103 errors by rotating around an axis.

        Returns:
            np.ndarray: Adjusted positions (n_atoms x 3).
        """
        def get_Rotation_M(axial_v, theta):
            v = np.array(axial_v[:3])
            # Normalize
            u, v_, w = v / np.linalg.norm(v)
            a = theta
            R_M = np.array([[u**2 + (1 - u**2) * np.cos(a), u * v_ * (1 - np.cos(a)) - w * np.sin(a),
                             u * w * (1 - np.cos(a)) + v_ * np.sin(a), 0],
                            [u * v_ * (1 - np.cos(a)) + w * np.sin(a), v_**2 + (1 - v_**2) * np.cos(a),
                             v_ * w * (1 - np.cos(a)) - u * np.sin(a), 0],
                            [u * w * (1 - np.cos(a)) - v_ * np.sin(a), v_ * w * (1 - np.cos(a)) + u * np.sin(a),
                             w**2 + (1 - w**2) * np.cos(a), 0],
                            [0, 0, 0, 1]])
            return R_M

        if (self.dihedral_idx[:, :3] == self.angle_idx).all():
            atom_to_be_adjusted_idx = self.angle_idx[0] - 1
            o_idx = self.angle_idx[1] - 1
            v_idx = self.angle_idx[2] - 1
        elif not self.dihedral_idx.all() or (self.dihedral_idx[:, 1:] == self.angle_idx).all():
            atom_to_be_adjusted_idx = self.angle_idx[-1] - 1
            o_idx = self.angle_idx[-2] - 1
            v_idx = self.angle_idx[-3] - 1
        elif ((self.dihedral_idx[:, :3] == self.angle_idx) + (self.dihedral_idx[:, 1:] == self.angle_idx)).all():
            atom_to_be_adjusted_idx = self.angle_idx[-1] - 1
            o_idx = self.angle_idx[-2] - 1
            v_idx = self.angle_idx[-3] - 1
        else:
            atom_to_be_adjusted_idx = self.angle_idx[-1] - 1
            o_idx = self.angle_idx[-2] - 1
            v_idx = self.angle_idx[-3] - 1
        Coord = deepcopy(self.running_positions[-1])
        symbol_list = self.symbol_list
        angle_v = Coord[v_idx] - Coord[o_idx]
        changing_v = Coord[atom_to_be_adjusted_idx] - Coord[o_idx]
        axial_v = np.cross(angle_v, changing_v)
        axial_v = axial_v / np.linalg.norm(axial_v)
        if symbol_list[atom_to_be_adjusted_idx] == 'H':
            theta = np.pi / 6.
        else:
            theta = np.pi / 36.

        R_M = get_Rotation_M(axial_v, theta)
        o_coord = Coord[o_idx]
        tmp_v = np.append(changing_v, [1])
        tmp_v = np.dot(tmp_v, R_M)
        tmp_v = np.around(np.delete(tmp_v, 3), decimals=8)
        Coord[atom_to_be_adjusted_idx] = tmp_v + o_coord
        return Coord

    def solve_l103_problem(self):
        """
        Resolve L103 errors by adjusting positions or falling back to previous step.

        Returns:
            np.ndarray: Adjusted or fallback positions (n_atoms x 3).
        """
        self.angle_idx, self.dihedral_idx = self.l103_error_idx()
        if self.angle_idx.all():
            new_position = self.l103_adjust()
        else:
            print('!!! Warning file: %s; Unknown Error：%s' % (self.file_dir, self.error_reason))
            new_position = self.running_positions[-2]
        return new_position

    def check_bond_attach(self, standard_file='mol', print_num=False, conf_id=-1,
                          bond_addition_function=None, bond_ignore_list=None):
        """
        Validate bond lengths in the final structure against standard (.mol) using RDKit.
        Checks if distances are within 0.75-1.3 times the reference.

        Args:
            standard_file (str, optional): Source for standard positions ('mol'). Defaults to 'mol'.
            print_num (bool, optional): Print distance ratios. Defaults to False.
            conf_id (int, optional): Conformation ID for running positions. Defaults to -1 (last).
            bond_addition_function (bond_addition_function, optional): Additional constraints.
            bond_ignore_list (list, optional): Bonds to ignore (list of lists of atom indices).

        Returns:
            bool: True if all bonds are valid.
        """
        try:
            # assert standard_file == "mol"
            if self.running_positions is None:
                return False
            mol = Chem.MolFromMolFile(self.mol_file_dir, removeHs=False, sanitize=False)
            for atom_id, atom in enumerate(mol.GetAtoms()):
                if atom.GetSymbol() != self.symbol_list[atom_id] and atom.GetAtomicNum() != self.symbol_list[atom_id]:
                    print("wrong with", atom.GetIdx(), atom.GetAtomicNum(), atom_id, self.symbol_list[atom_id])
                    return False
            if standard_file == "mol":
                position = mol.GetConformer(0).GetPositions()
            else:
                position = self.first_atom_position
            new_position = self.running_positions[conf_id]
            except_idxs = []
            if self.file_type == 'TS' or self.file_type == "OM":
                title = [each - 1 for each in self.title]
                if bond_ignore_list == None:
                    except_idxs = []
                else:
                    except_idxs = [[title[idx] for idx in each] for each in bond_ignore_list]
            for bond in mol.GetBonds():
                ignore = False
                start_atom_id = bond.GetBeginAtomIdx()
                end_atom_id = bond.GetEndAtomIdx()
                distance_a = Tool.get_atoms_distance(position[start_atom_id], position[end_atom_id])
                distance_b = Tool.get_atoms_distance(new_position[start_atom_id], new_position[end_atom_id])
                num = distance_a / distance_b
                for except_idx in except_idxs:
                    if start_atom_id in except_idx and end_atom_id in except_idx:
                        ignore = True
                if print_num:
                    print("%d %d %.5f" % (start_atom_id, end_atom_id, num))
                if (num <= 0.75 or num >= 1.3) and not ignore:
                    print(os.path.split(self.file_dir)[-1], start_atom_id, end_atom_id, "with a wrong distance", num)
                    return False
            if bond_addition_function is not None:
                if not bond_addition_function.apply_addition(title, new_position):
                    return False
            return True
        except:
            return False

    def irc_check(self):
        """
        Check IRC results for bond formation/breaking based on title atoms.

        Returns:
            int: Index of problematic bond (0 or 1), or -1 if successful.
        """
        title = [each - 1 for each in self.title]
        atom1, atom2, atom3 = title[:3]
        new_position = self.running_positions[-1]
        except_idxs = [[atom1, atom2], [atom2, atom3]]
        for id_, except_idx in enumerate(except_idxs):
            start_atom_id = except_idx[0]
            end_atom_id = except_idx[1]
            distance = Tool.get_atoms_distance(new_position[start_atom_id], new_position[end_atom_id])
            if distance <= 2.2:
                print(os.path.split(self.file_dir)[-1], start_atom_id, end_atom_id, "atom may ircorrect", distance)
                return id_
        return -1

    def unreal_freq_improve(self, new_log_dir, savechk=None, readchk=None, method=None):
        """
        Improve imaginary frequency by displacing along the mode by 1.1x and generating new input.

        Args:
            new_log_dir (str): Directory for new files.
            savechk (str, optional): %chk save path.
            readchk (str, optional): %chk read path.
            method (str, optional): Custom method.
        """
        position = self.running_positions[-1]
        title = " ".join(str(each) for each in self.title)
        new_position = np.array(self.unreal_freq_matrix) * 1.1 + np.array(position)
        if not os.path.isdir(new_log_dir):
            os.mkdir(new_log_dir)
        new_log_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1]
        new_gjf_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1].split(".")[0] + ".gjf"
        shutil.move(self.file_dir, new_log_name)
        if method == None:
            method = self.method
        FormatConverter.block_to_gjf(self.symbol_list, new_position, new_gjf_name, self.charge, self.multiplicity,
                                     title, method, freeze=self.freeze, difreeze=self.difreeze,
                                     savechk=savechk, readchk=readchk)

    def react_RMSD(self):
        """
        Calculate RMSD between first and last optimization structures.

        Returns:
            float: RMSD value in Angstroms.
        """
        position_start = self.running_positions[0]
        position_end = self.running_positions[-1]
        sum_delta2 = 0
        for each_start, each_end in zip(position_start, position_end):
            delta = each_start - each_end
            sum_delta2 += delta @ delta
        sum_delta2 /= len(position_start)
        return np.sqrt(sum_delta2)

    def check_om(self, return_value=False, set_num=0.4):
        """
        Validate transition state imaginary frequency corresponds to reaction coordinate
        by checking distance changes in bonds defined by title.

        Args:
            return_value (bool, optional): If True, return distance changes. Defaults to False.
            set_num (float, optional): Threshold for significant change. Defaults to 0.4.

        Returns:
            bool or tuple: True if valid TS, or (delta1, delta2) if return_value=True.
        """
        title = [each - 1 for each in self.title]
        position = deepcopy(self.running_positions[-1])
        new_position = deepcopy(self.unreal_freq_matrix)
        assert self.unreal_freq != 0
        # return new_position, position
        new_position += position
        bond1_dist_change = Tool.get_atoms_distance(new_position[title[0]], new_position[title[1]]) - \
                            Tool.get_atoms_distance(position[title[0]], position[title[1]])
        bond2_dist_change = Tool.get_atoms_distance(new_position[title[1]], new_position[title[2]]) - \
                            Tool.get_atoms_distance(position[title[1]], position[title[2]])
        if (bond1_dist_change > set_num and bond2_dist_change < set_num) or \
           (bond1_dist_change < set_num and bond2_dist_change > set_num):
            if return_value:
                return (bond1_dist_change, bond2_dist_change)
            else:
                return 1
        else:
            print("%s may find wrong TS" % self.file_dir)
        return 0

    def read_charge_spin_density(self):
        """
        Extract Mulliken charges and spin densities.

        Returns:
            tuple: (charges: list of float, spin_densities: list of float).
        """
        lines = self.filelines
        start_ids = [id_ for id_, line in enumerate(lines) if line.startswith(" Mulliken charges:")]
        if len(start_ids) == 0:
            start_ids = [id_ for id_, line in enumerate(lines) if line.startswith(" Mulliken charges and spin densities:")]
        start_id = start_ids[-1] + 2
        end_id = [id_ for id_, line in enumerate(lines[start_id:]) if line.startswith(" Sum of Mulliken charges =")][-1] + start_id
        charges = []
        spin_density = []
        for line in lines[start_id:end_id]:
            line_split = line.strip("\n").split()
            if len(line_split) == 3:
                charges.append(line_split[-1])
            else:
                charges.append(line_split[-2])
                spin_density.append(line_split[-1])
        charges = [float(each) for each in charges]
        spin_density = [float(each) for each in spin_density]
        return charges, spin_density

    def orbit_str_split(self, str_):
        """
        Split orbital energy strings handling ranges like "a-b-c".

        Args:
            str_ (str): Input string from log.

        Returns:
            list: List of individual orbital indices or energies.
        """
        result_lists = []
        str_lists = str_.split()
        for each in str_lists:
            if each.count("-") >= 2:
                tmp_lists = [i for i in each.split("-") if i != ""]
                result_lists += ["-" + a for a in tmp_lists]
            else:
                result_lists.append(each)
        return result_lists

    def read_orbit_eng(self, HOMO_index=[-2, -1], LUMO_index=[0, 1]):
        """
        Extract HOMO and LUMO orbital energies in kcal/mol (converted from Hartree).

        Args:
            HOMO_index (list, optional): Indices relative to HOMO (negative for below). Defaults to [-2, -1].
            LUMO_index (list, optional): Indices relative to LUMO (positive). Defaults to [0, 1].

        Returns:
            list: [HOMO alphas, LUMO alphas, HOMO betas, LUMO betas] energies.
        """
        occ_eng = []
        virt_eng = []
        lines = self.filelines
        # start_id = [id for id, line in enumerate(lines) if line.startswith(" The electronic state is")][-1] + 1
        # end_id = [id for id, line in enumerate(lines[start_id:]) if line.startswith("          Condensed to atoms (all electrons):")][-1] + start_id
        occ_orbits = [line for line in lines if line.startswith(" Alpha  occ. eigenvalues")]
        virt_orbits = [line for line in lines if line.startswith(" Alpha virt. eigenvalues")]
        for occ_orbit in occ_orbits:
            occ_eng += [float(each) * 627.5 for each in self.orbit_str_split(occ_orbit.strip("\n").split("--")[-1])]
        for virt_orbit in virt_orbits:
            virt_eng += [float(each) * 627.5 for each in self.orbit_str_split(virt_orbit.strip("\n").split("--")[-1])]
        occ_eng = [occ_eng[each] for each in HOMO_index]
        virt_eng = [virt_eng[each] for each in LUMO_index]

        occ_eng_beta = []
        virt_eng_beta = []
        occ_orbits = [line for line in lines if line.startswith("  Beta  occ. eigenvalues")]
        virt_orbits = [line for line in lines if line.startswith("  Beta virt. eigenvalues")]
        if len(occ_orbits) == 0:
            return occ_eng + virt_eng
        for occ_orbit in occ_orbits:
            occ_eng_beta += [float(each) * 627.5 for each in self.orbit_str_split(occ_orbit.strip("\n").split("--")[-1])]
        for virt_orbit in virt_orbits:
            virt_eng_beta += [float(each) * 627.5 for each in self.orbit_str_split(virt_orbit.strip("\n").split("--")[-1])]
        occ_eng_beta = [occ_eng_beta[each] for each in HOMO_index]
        virt_eng_beta = [virt_eng_beta[each] for each in LUMO_index]
        return occ_eng + virt_eng + occ_eng_beta + virt_eng_beta

    def get_dipole(self):
        """
        Extract the dipole moment from the log file.

        Returns:
            float: Dipole moment in Debye, or -1 if not found.
        """
        lines = self.filelines
        dipole_line_id = [line_id for line_id, line in enumerate(lines) if line.startswith(" Dipole moment ")]
        if len(dipole_line_id) == 0:
            print(self.filelines, "did't have a dipole moment")
            return -1

        dipole_line = lines[dipole_line_id[-1] + 1]
        dipole_moment = float(dipole_line.strip("\n").split()[-1])
        return dipole_moment

    def get_sterimol_parameters(self, atoma, atomb):
        """
        Calculate Sterimol parameters (L, B1, B5) for a substituent atomb attached to atoma.

        Args:
            atoma (int): Central atom index (1-based).
            atomb (int): Substituent attachment atom index (1-based).

        Returns:
            tuple: (L: float, B1: float, B5: float).
        """
        file_Params = sterimoltools.calcSterimol(file=self.file_dir, radii="cpk", atomA=atoma + 1, atomB=atomb + 1, verbose=False)
        L = file_Params.lval
        B1 = file_Params.B1
        B5 = file_Params.newB5
        return L, B1, B5


if __name__ == "__main__":
    pwd = os.getcwd()
    log_files = glob.glob(pwd + "/*.log")
    # improve_dir = pwd + "/improve"
    for log_file in log_files:
        opt_log = Logfile(log_file)
        print("log_file", opt_log.read_orbit_eng([-1], [0]))
        # if not opt_log.normal_end:
        #     opt_log.solve_error_logfile(improve_dir)
        # elif opt_log.unreal_freq:
        #     opt_log.unreal_freq_improve(improve_dir)