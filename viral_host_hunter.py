import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from utils.encoder import Encoder
from model.lgbm import LightGBM
from model.dnn import DNN
from model.meta import Meta
from utils.lable2host import lable2host
from utils.taxnomic_info import species2genus, genus2family


class ViralHostHnter:
    def __init__(self, vhh_type, batch_size):
        self.vhh_type = vhh_type
        self.batch_size = batch_size
        self.lable2host = lable2host
        self.genus2family = genus2family
        self.species2genus = species2genus
        self.max_len = {'tail': 1500, 'lysin': 1000}
        self.cutoff = {
            'tail': {
                95: {'family': 0.651655536533789, 'genus': 0.609586374493945, 'species': 0.946488658862898},
                84: {'family': 0.453619077911532, 'genus': 0.478478651364142, 'species': 0.692632951360313},
                69: {'family': 0.30979088720081, 'genus': 0.298409778900575, 'species': 0.541229237074987},
                -1: {'family': -1, 'genus': -1, 'species': -1}
            },
            'lysin': {
                95: {'family': 0.8827, 'genus': 0.8729, 'species': 0.8200},
                84: {'family': 0.4811, 'genus': 0.4857, 'species': 0.6128},
                69: {'family': 0.3137, 'genus': 0.2821, 'species': 0.2385},
                -1: {'family': -1, 'genus': -1, 'species': -1}
            },
        }
        # load base module
        self.encoder = Encoder()
        self.lgbm = LightGBM(self.vhh_type)
        self.dnn = DNN(self.vhh_type)
        self.meta = Meta(self.vhh_type)

    def predict(self, dna_file, protein_file, precision, output_file):
        # reliable_pred <dict>, reliable_pred_proba <dict>
        reliable_pred, reliable_pred_proba = self.predict_(dna_file, protein_file, precision)
        # update predictions
        updated_pred = self.update(reliable_pred, precision)
        # output
        self.write(updated_pred, reliable_pred_proba, output_file)

    def predict_(self, dna_file, protein_file, precision):
        valid_dna_list, valid_protein_list, is_valid = self.check(dna_file, protein_file)
        pred = {'family': [], 'genus': [], 'species': []}
        pred_proba = {'family': [], 'genus': [], 'species': []}

        for i in tqdm(range(0, len(valid_dna_list), self.batch_size)):
            # encode
            features = self.encoder.features(
                valid_dna_list[i: i+self.batch_size],
                valid_protein_list[i: i+self.batch_size]
            )
            one_hot = self.encoder.one_hot(
                valid_protein_list[i: i+self.batch_size],
                self.max_len[self.vhh_type]
            )
            # predict
            for taxonomic_rank in ('family', 'genus', 'species'):
                lgbm_proba = self.lgbm.predict_proba(features, taxonomic_rank)
                dnn_proba = self.dnn.predict_proba(one_hot, taxonomic_rank)
                cat = np.concatenate([lgbm_proba, dnn_proba], axis=1)
                meta_pred = self.meta.predict(cat, taxonomic_rank).reshape(-1, )
                meta_proba = self.meta.predict_proba(cat, taxonomic_rank).reshape(-1, )
                pred[taxonomic_rank] += list(meta_pred)
                pred_proba[taxonomic_rank] += list(meta_proba)

        # filter out unreliable predictions
        reliable_pred = {'family': [], 'genus': [], 'species': []}
        reliable_pred_proba = {'family': [], 'genus': [], 'species': []}
        j = 0
        for i in range(len(is_valid)):
            if not is_valid[i]:
                for taxonomic_rank in ('family', 'genus', 'species'):
                    reliable_pred[taxonomic_rank].append('Unknown')
                    reliable_pred_proba[taxonomic_rank].append(-1)
            else:
                for taxonomic_rank in ('family', 'genus', 'species'):
                    if pred_proba[taxonomic_rank][j] < self.cutoff[self.vhh_type][precision][taxonomic_rank]:
                        reliable_pred[taxonomic_rank].append('Unknown')
                        reliable_pred_proba[taxonomic_rank].append(pred_proba[taxonomic_rank][j])
                    else:
                        reliable_pred[taxonomic_rank].append(
                            self.lable2host[self.vhh_type][taxonomic_rank][pred[taxonomic_rank][j]]
                        )
                        reliable_pred_proba[taxonomic_rank].append(pred_proba[taxonomic_rank][j])
                j += 1

        return reliable_pred, reliable_pred_proba

    def check(self, dna_file, protein_file):
        dna_list = [str(rec.seq) for rec in SeqIO.parse(dna_file, 'fasta')]
        protein_list = [str(rec.seq).strip("*") for rec in SeqIO.parse(protein_file, 'fasta')]
        # mark valid data
        is_valid = []
        for i in range(len(dna_list)):
            dna_seq, protein_seq = dna_list[i], protein_list[i]
            if len(protein_seq) < 50 or len(protein_seq) > self.max_len[self.vhh_type]: is_valid.append(False)
            elif not bool(re.match('^[ATCG]+$', dna_seq)): is_valid.append(False)
            elif not bool(re.match('^[ACDEFGHIKLMNPQRSTVWY]+$', protein_seq)): is_valid.append(False)
            else: is_valid.append(True)

        valid_dna_list, valid_protein_list = [], []
        for i in range(len(is_valid)):
            if is_valid[i]:
                valid_dna_list.append(dna_list[i])
                valid_protein_list.append(protein_list[i])

        return valid_dna_list, valid_protein_list, is_valid

    def update(self, reliable_pred, precision):
        # set unknown
        if self.vhh_type == 'tail':
            if precision == 95:
                for i in range(len(reliable_pred['genus'])):
                    if reliable_pred['genus'][i] in ('Faecalibacterium', 'Agrobacterium', 'Limosilactobacillus'):
                        reliable_pred['genus'][i] = 'Unknown'
                for i in range(len(reliable_pred['species'])):
                    if reliable_pred['species'][i] in (
                            'Faecalibacterium prausnitzii',
                            'Agrobacterium tumefaciens',
                            'Pseudomonas syringae'
                    ):
                        reliable_pred['species'][i] = 'Unknown'
            elif precision == 84:
                for i in range(len(reliable_pred['species'])):
                    if reliable_pred['species'][i] in (
                            'Faecalibacterium prausnitzii',
                            'Agrobacterium tumefaciens',
                            'Pseudomonas syringae'
                    ):
                        reliable_pred['species'][i] = 'Unknown'
            elif precision == 69:
                for i in range(len(reliable_pred['species'])):
                    if reliable_pred['species'][i] in ('Agrobacterium tumefaciens', 'Pseudomonas syringae'):
                        reliable_pred['species'][i] = 'Unknown'

        # update taxonomic predictions
        updated_pred = reliable_pred
        for i in range(len(updated_pred['family'])):
            if updated_pred['family'][i] == 'Unknown':
                # if family == unknown and genus == unknown, update family and genus according to species
                if updated_pred['genus'][i] == 'Unknown':
                    updated_pred['genus'][i] = self.species2genus[updated_pred['species'][i]]
                    updated_pred['family'][i] = self.genus2family[self.species2genus[updated_pred['species'][i]]]
                else:  # if family == unknown and genus != unknown, update family according to species or genus
                    if updated_pred['species'][i] != 'Unknown':
                        updated_pred['family'][i] = self.genus2family[self.species2genus[updated_pred['species'][i]]]
                    else:
                        updated_pred['family'][i] = self.genus2family[updated_pred['genus'][i]]
            else:  # if family != unknown and genus == unknown, update genus according to species
                if updated_pred['genus'][i] == 'Unknown':
                    updated_pred['genus'][i] = self.species2genus[updated_pred['species'][i]]

        return updated_pred

    def write(self, pred, pre_proba, output_file):
        df = pd.DataFrame()
        df['family'] = np.asarray(pred['family'])
        df['genus'] = np.asarray(pred['genus'])
        df['species'] = np.asarray(pred['species'])
        # df['family_proba'] = np.asarray(pre_proba['family'])
        # df['genus_proba'] = np.asarray(pre_proba['genus'])
        # df['species_proba'] = np.asarray(pre_proba['species'])

        df.to_csv(output_file, index=False)
