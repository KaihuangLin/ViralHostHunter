import numpy as np
import pandas as pd
from Bio.SeqUtils import GC, CodonUsage
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class Encoder:
    """
    Reference: Predicting bacteriophage hosts based on sequences of annotated receptor-binding proteins
    Demo:
    encoder = Encoder()
    features = encoder.features(dna_sequences_list, protein_sequences_list)  # shape: (len(protein_sequences_list), 218)
    one_host = encoder.one_hot(protein_sequences_list, max_len)  # shape: (len(protein_sequences_list), max_len, 20, 1)
    """
    def __init__(self):
        self.amino_acid = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }

    def features(self, dna_list, protein_list):
        dna_features = self.dna_features(dna_list)
        protein_features = self.protein_features(protein_list)
        extra_features = self.extra_features(protein_list)
        features = pd.concat([dna_features, protein_features, extra_features], axis=1)
        return features.values

    def one_hot(self, protein_list, max_len):
        output = np.zeros(shape=(len(protein_list), max_len, 20))
        for i in range(len(protein_list)):
            matrix = np.eye(20)[[self.amino_acid[aa] for aa in protein_list[i]]]
            output[i][: len(matrix), :] = matrix
        output = output.reshape((-1, max_len, 20, 1))
        return output

    def dna_features(self, dna_list):
        A_freq, T_freq, C_freq, G_freq, GC_content = [], [], [], [], []
        codontable = {
            'ATA': [], 'ATC': [], 'ATT': [], 'ATG': [], 'ACA': [], 'ACC': [], 'ACG': [], 'ACT': [],
            'AAC': [], 'AAT': [], 'AAA': [], 'AAG': [], 'AGC': [], 'AGT': [], 'AGA': [], 'AGG': [],
            'CTA': [], 'CTC': [], 'CTG': [], 'CTT': [], 'CCA': [], 'CCC': [], 'CCG': [], 'CCT': [],
            'CAC': [], 'CAT': [], 'CAA': [], 'CAG': [], 'CGA': [], 'CGC': [], 'CGG': [], 'CGT': [],
            'GTA': [], 'GTC': [], 'GTG': [], 'GTT': [], 'GCA': [], 'GCC': [], 'GCG': [], 'GCT': [],
            'GAC': [], 'GAT': [], 'GAA': [], 'GAG': [], 'GGA': [], 'GGC': [], 'GGG': [], 'GGT': [],
            'TCA': [], 'TCC': [], 'TCG': [], 'TCT': [], 'TTC': [], 'TTT': [], 'TTA': [], 'TTG': [],
            'TAC': [], 'TAT': [], 'TAA': [], 'TAG': [], 'TGC': [], 'TGT': [], 'TGA': [], 'TGG': []
        }

        for item in dna_list:
            A_freq.append(item.count('A') / len(item))
            T_freq.append(item.count('T') / len(item))
            C_freq.append(item.count('C') / len(item))
            G_freq.append(item.count('G') / len(item))
            GC_content.append(GC(item))

            codons = [item[i: i+3] for i in range(0, len(item), 3)]
            l = []
            for key in codontable.keys():
                l.append(codons.count(key))
            l_norm = [float(i) / sum(l) for i in l]

            for j, key in enumerate(codontable.keys()):
                codontable[key].append(l_norm[j])

        synonym_codons = CodonUsage.SynonymousCodons
        codontable2 = {
            'ATA_b': [], 'ATC_b': [], 'ATT_b': [], 'ATG_b': [], 'ACA_b': [], 'ACC_b': [], 'ACG_b': [], 'ACT_b': [],
            'AAC_b': [], 'AAT_b': [], 'AAA_b': [], 'AAG_b': [], 'AGC_b': [], 'AGT_b': [], 'AGA_b': [], 'AGG_b': [],
            'CTA_b': [], 'CTC_b': [], 'CTG_b': [], 'CTT_b': [], 'CCA_b': [], 'CCC_b': [], 'CCG_b': [], 'CCT_b': [],
            'CAC_b': [], 'CAT_b': [], 'CAA_b': [], 'CAG_b': [], 'CGA_b': [], 'CGC_b': [], 'CGG_b': [], 'CGT_b': [],
            'GTA_b': [], 'GTC_b': [], 'GTG_b': [], 'GTT_b': [], 'GCA_b': [], 'GCC_b': [], 'GCG_b': [], 'GCT_b': [],
            'GAC_b': [], 'GAT_b': [], 'GAA_b': [], 'GAG_b': [], 'GGA_b': [], 'GGC_b': [], 'GGG_b': [], 'GGT_b': [],
            'TCA_b': [], 'TCC_b': [], 'TCG_b': [], 'TCT_b': [], 'TTC_b': [], 'TTT_b': [], 'TTA_b': [], 'TTG_b': [],
            'TAC_b': [], 'TAT_b': [], 'TAA_b': [], 'TAG_b': [], 'TGC_b': [], 'TGT_b': [], 'TGA_b': [], 'TGG_b': []
        }

        for item1 in dna_list:
            codons = [item1[l: l+3] for l in range(0, len(item1), 3)]
            codon_counts = []

            for key in codontable.keys():
                codon_counts.append(codons.count(key))

            for key_syn in synonym_codons.keys():
                total = 0
                for item2 in synonym_codons[key_syn]:
                    total += codons.count(item2)
                for j, key_table in enumerate(codontable.keys()):
                    if (key_table in synonym_codons[key_syn]) & (total != 0):
                        codon_counts[j] /= total

            for k, key_table in enumerate(codontable2.keys()):
                codontable2[key_table].append(codon_counts[k])

        features_codonbias = pd.DataFrame.from_dict(codontable2)
        features_dna = pd.DataFrame.from_dict(codontable)
        features_dna['A_freq'] = np.asarray(A_freq)
        features_dna['T_freq'] = np.asarray(T_freq)
        features_dna['C_freq'] = np.asarray(C_freq)
        features_dna['G_freq'] = np.asarray(G_freq)
        features_dna['GC'] = np.asarray(GC_content)

        features = pd.concat([features_dna, features_codonbias], axis=1)
        return features

    def protein_features(self, protein_list):
        mol_weight = []
        aromaticity = []
        instability = []
        flexibility = []
        prot_length = []
        pI = []
        helix_frac = []
        turn_frac = []
        sheet_frac = []
        frac_aliph = []
        frac_unch_polar = []
        frac_polar = []
        frac_hydrophob = []
        frac_pos = []
        frac_sulfur = []
        frac_neg = []
        frac_amide = []
        frac_alcohol = []
        AA_dict = {
            'G': [], 'A': [], 'V': [], 'L': [], 'I': [], 'F': [], 'P': [], 'S': [], 'T': [], 'Y': [],
            'Q': [], 'N': [], 'E': [], 'D': [], 'W': [], 'H': [], 'R': [], 'K': [], 'M': [], 'C': []
        }

        for item in protein_list:
            prot_length.append(len(item))
            frac_aliph.append((item.count('A') + item.count('G') + item.count('I') + item.count('L') + item.count('P')
                               + item.count('V')) / len(item))
            frac_unch_polar.append((item.count('S') + item.count('T') + item.count('N') + item.count('Q')) / len(item))
            frac_polar.append((item.count('Q') + item.count('N') + item.count('H') + item.count('S') + item.count(
                'T') + item.count('Y')
                               + item.count('C') + item.count('M') + item.count('W')) / len(item))
            frac_hydrophob.append(
                (item.count('A') + item.count('G') + item.count('I') + item.count('L') + item.count('P')
                 + item.count('V') + item.count('F')) / len(item))
            frac_pos.append((item.count('H') + item.count('K') + item.count('R')) / len(item))
            frac_sulfur.append((item.count('C') + item.count('M')) / len(item))
            frac_neg.append((item.count('D') + item.count('E')) / len(item))
            frac_amide.append((item.count('N') + item.count('Q')) / len(item))
            frac_alcohol.append((item.count('S') + item.count('T')) / len(item))
            protein_chars = ProteinAnalysis(item)
            mol_weight.append(protein_chars.molecular_weight())
            aromaticity.append(protein_chars.aromaticity())
            instability.append(protein_chars.instability_index())
            flexibility.append(np.mean(protein_chars.flexibility()))
            pI.append(protein_chars.isoelectric_point())
            H, T, S = protein_chars.secondary_structure_fraction()
            helix_frac.append(H)
            turn_frac.append(T)
            sheet_frac.append(S)

            for key in AA_dict.keys():
                AA_dict[key].append(item.count(key) / len(item))

        features_protein = pd.DataFrame.from_dict(AA_dict)
        features_protein['protein_length'] = np.asarray(prot_length)
        features_protein['mol_weight'] = np.asarray(mol_weight)
        features_protein['aromaticity'] = np.asarray(aromaticity)
        features_protein['instability'] = np.asarray(instability)
        features_protein['flexibility'] = np.asarray(flexibility)
        features_protein['pI'] = np.asarray(pI)
        features_protein['frac_aliphatic'] = np.asarray(frac_aliph)
        features_protein['frac_uncharged_polar'] = np.asarray(frac_unch_polar)
        features_protein['frac_polar'] = np.asarray(frac_polar)
        features_protein['frac_hydrophobic'] = np.asarray(frac_hydrophob)
        features_protein['frac_positive'] = np.asarray(frac_pos)
        features_protein['frac_sulfur'] = np.asarray(frac_sulfur)
        features_protein['frac_negative'] = np.asarray(frac_neg)
        features_protein['frac_amide'] = np.asarray(frac_amide)
        features_protein['frac_alcohol'] = np.asarray(frac_alcohol)
        features_protein['AA_frac_helix'] = np.asarray(helix_frac)
        features_protein['AA_frac_turn'] = np.asarray(turn_frac)
        features_protein['AA_frac_sheet'] = np.asarray(sheet_frac)
        return features_protein

    def extra_features(self, protein_list):
        extra_features = np.zeros((len(protein_list), 47))
        for i, item in enumerate(protein_list):
            feature_lst = []
            feature_lst += self.CTDC(item)
            feature_lst += self.CTDT(item)
            feature_lst += self.zscale(item)
            extra_features[i, :] = feature_lst
        extra_features_df = pd.DataFrame(extra_features, columns=[
            'CTDC1', 'CTDC2', 'CTDC3', 'CTDT1', 'CTDT2', 'CTDT3', 'CTDT4', 'CTDT5', 'CTDT6', 'CTDT7',
            'CTDT8', 'CTDT9', 'CTDT10', 'CTDT11', 'CTDT12', 'CTDT13', 'CTDT14', 'CTDT15', 'CTDT16', 'CTDT17',
            'CTDT18', 'CTDT19', 'CTDT20', 'CTDT21', 'CTDT22', 'CTDT23', 'CTDT24', 'CTDT25', 'CTDT26', 'CTDT27',
            'CTDT28', 'CTDT29', 'CTDT30', 'CTDT31', 'CTDT32', 'CTDT33', 'CTDT34', 'CTDT35', 'CTDT36', 'CTDT37',
            'CTDT38', 'CTDT39', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'
        ])
        return extra_features_df

    def Count1(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def CTDC(self, sequence):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        property = [
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101',
            'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess'
        ]
        for p in property:
            c1 = self.Count1(group1[p], sequence) / len(sequence)
            c2 = self.Count1(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            encoding = [c1, c2, c3]
        return encoding

    def CTDT(self, sequence):
        group1 = {'hydrophobicity_PRAM900101': 'RKEDQN', 'hydrophobicity_ARGP820101': 'QSTNGDE',
                  'hydrophobicity_ZIMJ680101': 'QNGSWTDERA', 'hydrophobicity_PONP930101': 'KPDESNQT',
                  'hydrophobicity_CASG920101': 'KDEQPSRNTG', 'hydrophobicity_ENGD860101': 'RDKENQHYP',
                  'hydrophobicity_FASG890101': 'KERSQD', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY',
                  'polarizability': 'GASDT', 'charge': 'KR', 'secondarystruct': 'EALMQKRH', 'solventaccess': 'ALFCGIVW'}

        group2 = {'hydrophobicity_PRAM900101': 'GASTPHY', 'hydrophobicity_ARGP820101': 'RAHCKMV',
                  'hydrophobicity_ZIMJ680101': 'HMCKV', 'hydrophobicity_PONP930101': 'GRHA',
                  'hydrophobicity_CASG920101': 'AHYMLV', 'hydrophobicity_ENGD860101': 'SGTAW',
                  'hydrophobicity_FASG890101': 'NTPG', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS',
                  'polarizability': 'CPNVEQIL', 'charge': 'ANCQGHILMFPSTWYV',
                  'secondarystruct': 'VIYCWFT', 'solventaccess': 'RKQEND'}

        group3 = {'hydrophobicity_PRAM900101': 'CLVIMFW', 'hydrophobicity_ARGP820101': 'LYPFIW',
                  'hydrophobicity_ZIMJ680101': 'LPFYI', 'hydrophobicity_PONP930101': 'YMFWLCVI',
                  'hydrophobicity_CASG920101': 'FIWC', 'hydrophobicity_ENGD860101': 'CVLIMF',
                  'hydrophobicity_FASG890101': 'AYHWVMFLIC', 'normwaalsvolume': 'MHKFRYW',
                  'polarity': 'HQRKNED', 'polarizability': 'KMHFRYW', 'charge': 'DE',
                  'secondarystruct': 'GNPSD', 'solventaccess': 'MSPTHY'}

        property = ['hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                    'hydrophobicity_PONP930101',
                    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101',
                    'normwaalsvolume',
                    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess']

        encoding = []
        aaPair = [sequence[j: j+2] for j in range(len(sequence)-1)]

        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 += 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 += 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 += 1
            encoding.append(c1221/len(aaPair))
            encoding.append(c1331/len(aaPair))
            encoding.append(c2332/len(aaPair))
        return encoding

    def zscale(self, sequence):
        zdict = {
            'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
            'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
            'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
            'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
            'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
            'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
            'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
            'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
            'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
            'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
            'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
            'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
            'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
            'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
            'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
            'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
            'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
            'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
            'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
            'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
            '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
        }
        z1, z2, z3, z4, z5 = 0, 0, 0, 0, 0
        for aa in sequence:
            z1 += zdict[aa][0]
            z2 += zdict[aa][1]
            z3 += zdict[aa][2]
            z4 += zdict[aa][3]
            z5 += zdict[aa][4]
        encoding = [z1/len(sequence), z2/len(sequence), z3/len(sequence), z4/len(sequence), z5/len(sequence)]
        return encoding
