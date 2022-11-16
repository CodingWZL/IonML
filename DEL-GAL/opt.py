from xgboost.sklearn import XGBClassifier
import numpy as np
import os
from sklearn import preprocessing
from bayes_opt import BayesianOptimization
from pymatgen import Lattice, Structure, Composition
from ComponentFeature import Li_fraction
from matminer.featurizers.composition.element import BandCenter

enreg = XGBClassifier()

def ion(a,b,c,d):
    ##### get the new chemical formula
    formula = "Li"+str(a)+"Ca"+str(b)+"B"+str(c)+"O"+str(d)
    composition = Composition(formula)
    
	##### load the original HMFP vector
    feature = np.loadtxt("HMFP.txt").reshape(1,27)
	
	##### calculate the component feature
    component_feature = np.array(Li_fraction(Composition(composition)))
    BC = BandCenter()
    bandcenter = BC.featurize(composition)
    
	##### update the component feature
    feature[0][14:26] = component_feature[0:]
    feature[0][26] = bandcenter[0]
    
	##### load the well-trained model from five-fold cross-validation
    enreg.load_model('xgb-1.model')
    p_1 = enreg.predict_proba(feature)[0][1]
    enreg.load_model('xgb-2.model')
    p_2 = enreg.predict_proba(feature)[0][1]
    enreg.load_model('xgb-3.model')
    p_3 = enreg.predict_proba(feature)[0][1]
    enreg.load_model('xgb-4.model')
    p_4 = enreg.predict_proba(feature)[0][1]
    enreg.load_model('xgb-5.model')
    p_5 = enreg.predict_proba(feature)[0][1]
	
	##### print the average probability of being SIC
    return (p_1+p_2+p_3+p_4+p_5)/5

def main():
	##### define the range of optimizition object
    opt = BayesianOptimization(ion,\
    {'a':(0.8,1.8),\
    'b':(0.8,1.3),\
    'c':(0.8,1.3),\
    'd':(2.5,3.5)\
    }
    )
	##### exploration and exploitation
    opt.maximize(init_points=200, n_iter=20)

if __name__ == '__main__':
    main()
