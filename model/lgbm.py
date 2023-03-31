from utils.save_load import load


class LightGBM:
    def __init__(self, vhh_type):
        self.standard_scaler = {
            'family': load('model/' + vhh_type + '/family/std.pkl'),
            'genus': load('model/' + vhh_type + '/genus/std.pkl'),
            'species': load('model/' + vhh_type + '/species/std.pkl')
        }
        self.lgbm = {
            'family': load('model/' + vhh_type + '/family/lgbm.pkl'),
            'genus': load('model/' + vhh_type + '/genus/lgbm.pkl'),
            'species': load('model/' + vhh_type + '/species/lgbm.pkl')
        }

    def predict_proba(self, X, taxonomic_rank):
        X = self.standard_scaler[taxonomic_rank].transform(X)
        pred_proba = self.lgbm[taxonomic_rank].predict_proba(X)

        return pred_proba
