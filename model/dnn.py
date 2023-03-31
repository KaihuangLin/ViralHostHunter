from model.tail.dnn import tail_family_model, tail_genus_model, tail_species_model
from model.lysin.dnn import lysin_family_model, lysin_genus_model, lysin_species_model


class DNN:
    def __init__(self, vhh_type):
        self.vhh_type = vhh_type
        self.dnn = {
            'tail': {'family': tail_family_model(), 'genus': tail_genus_model(), 'species': tail_species_model()},
            'lysin': {'family': lysin_family_model(), 'genus': lysin_genus_model(), 'species': lysin_species_model()}
        }
        self.dnn[self.vhh_type]['family'].load_weights('model/' + vhh_type + '/family/dnn.hdf5')
        self.dnn[self.vhh_type]['genus'].load_weights('model/' + vhh_type + '/genus/dnn.hdf5')
        self.dnn[self.vhh_type]['species'].load_weights('model/' + vhh_type + '/species/dnn.hdf5')

    def predict_proba(self, X, taxonomic_rank):
        pred_proba = self.dnn[self.vhh_type][taxonomic_rank].predict(X)

        return pred_proba
