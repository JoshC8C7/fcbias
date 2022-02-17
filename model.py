import experiments

import feverbaseline.src.scripts as fvscr

class Model():

    name = None


    def __init__(self, name):
        self.name = name
        print()



class DA_Baseline_Model(Model):

    def __init__(self):
        super().__init__()

    def predict(self, predict_list):

