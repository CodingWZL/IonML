import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import classification_report, accuracy_score, precision_score, cohen_kappa_score
from xgboost import plot_tree
import shap

##### five-fold cross-validation #####
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
######################################

##### HMFP and SIC values (Boolean) #####
X = np.loadtxt("HMFP.txt")
Y = np.loadtxt("SIC.txt")
#########################################

j=0
scores = []
for train, test in kfold.split(X, Y):
    j = j + 1
    enreg = XGBClassifier(learning_rate=0.057, n_estimators=218,\
            subsample=0.97, colsample_bytree=0.86, max_depth=5,\
            scale_pos_weight=0.93, nthread=3)
    #enreg = KNeighborsClassifier(n_neighbors=3, leaf_size=20)
    #enreg = RadiusNeighborsClassifier(radius=10.0, outlier_label =
    #enreg = RandomForestClassifier(n_estimators=200, max_depth=6)
    #enreg = SVR(kernel='rbf',degree=4, tol=1e-5,C=1.0, epsilon=0.1)
    #enreg = KernelRidge(kernel='rbf')
    #enreg = DecisionTreeClassifier(max_depth=6)
    #enreg = AdaBoostClassifier(base_estimator=a, learning_rate=0.1, #n_estimators=250)
    enreg.fit(X[train], Y[train])
    model = enreg.fit(X[train], Y[train])
	
	##### SHAP analysis #####
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[train])[:,12] # 13th feature
    a = np.abs(shap_values)
    a = a.mean(axis=0)
    np.savetxt("His"+str(j)+".txt",X[train][:,12])
    np.savetxt("shap_values"+str(j)+".txt",shap_values)
	#########################

    ##### model performance #####
    scores.append(precision_score(Y[test], enreg.predict(X[test]), average='weighted'))
    scores.append(accuracy_score(Y[test], enreg.predict(X[test])))
    print(classification_report(Y[test],enreg.predict(X[test])))
	#############################
    
	##### ROC curve on testing data #####
    fpr,tpr,thres = roc_curve(Y[test],enreg.predict(X[test]))
    results = np.vstack((fpr, tpr, thres))
    np.savetxt("results"+str(j)+".txt",results.T)
	#####################################

	##### save model #####
    enreg.save_model('xgb-'+ str(j) + '.model')
	######################
	
	##### plot the tree in XGBoost #####
    mpl.use('Agg')
    plot_tree(enreg,num_trees=83)
    plt.savefig('1.tif',dpi=300)
	####################################
    
	##### print the feature importance #####
    importances = enreg.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%2d) %f" % (f + 1, importances[f]))
	########################################

print(scores)