import numpy as np
from utils.save_load import load


class Meta:
    def __init__(self, vhh_type):
        self.meta = {
            'family': load('model/' + vhh_type + '/family/meta.pkl'),
            'genus': load('model/' + vhh_type + '/genus/meta.pkl'),
            'species': load('model/' + vhh_type + '/species/meta.pkl')
        }

    def predict(self, X, taxonomic_rank):
        pred = self.meta[taxonomic_rank].predict(X)

        return pred

    def predict_proba(self, X, taxonomic_rank):
        pred_proba = np.max(self.meta[taxonomic_rank].predict_proba(X), axis=1)

        return pred_proba
