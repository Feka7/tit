#!/usr/bin/python

import pickle
from csv import reader
from pkl2dict import featureFormat, targetFeatureSplit
import numpy
import csv
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import ensemble
i=0
dict = pickle.load(open("myfile.pkl", "rb"))
dict1 = pickle.load(open("myfile2.pkl", "rb"))
#dict.pop("178")
#dict.pop("298")
#dict.pop("499")
features_list_base = ["Survived", "Sex", "Pclass","Onboard"]
data_base = featureFormat(dict, features_list_base )
labels_train_base, features_train_base = targetFeatureSplit( data_base )

features_list = ["Survived", "Sex", "Pclass","Onboard","Fare","Embarked","Age"]
data = featureFormat(dict, features_list )
labels_train, features_train = targetFeatureSplit( data )
'''
print("---Test prima fase---")

X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, test_size=0.15, random_state=42)

clf = DecisionTreeClassifier(random_state=0)
clf1 = GaussianNB()
clf2 = SVC()

p = {
    'max_leaf_nodes': [2, 4, 6, 10, 20, 40],
    'min_samples_split': [1.0, 5, 10 ,100]
    }

p1 = {}

p2 = {'C': [1, 10, 100, 1000],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
    }

cross_validation = KFold(n_splits=5,random_state=0,shuffle=True)
grid_search_decision = GridSearchCV(clf,
                            param_grid= p,
                           cv=cross_validation,
                            n_jobs=-1,
                            verbose=0)

grid_search_decision.fit(X_train, y_train)
print("DecisionTreeClassifier")
print('Best score: {}'.format(grid_search_decision.best_score_))
print('Best parameters: {}'.format(grid_search_decision.best_params_))

grid_search_gaus = GridSearchCV(clf1,
                            param_grid= p1,
                           cv=cross_validation,
                            n_jobs=-1,
                            verbose=0)

grid_search_gaus.fit(X_train, y_train)
print("GaussianNB")
print('Best score: {}'.format(grid_search_gaus.best_score_))
print('Best parameters: {}'.format(grid_search_gaus.best_params_))

grid_search_svc = GridSearchCV(clf2,
                            param_grid= p2,
                           cv=cross_validation,
                            n_jobs=-1,
                            verbose=0)

grid_search_svc.fit(X_train, y_train)
print("SVC")
print('Best score: {}'.format(grid_search_svc.best_score_))
print('Best parameters: {}'.format(grid_search_svc.best_params_))

print("---Valuto le prestazioni con i migliori parametri---")

clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=10, min_samples_split=5)
clf.fit(X_train, y_train)
res = clf.predict(X_test)
print("DecisionTreeClassifier score: ",clf.score(X_test, y_test),", precison: ",precision_score(y_test, res),", recall: ",recall_score(y_test, res))

clf1 = GaussianNB()
clf1.fit(X_train, y_train)
res = clf1.predict(X_test)
print("GaussianNB score: ",clf1.score(X_test, y_test),", precison: ",precision_score(y_test, res),", recall: ",recall_score(y_test, res))

clf2 = SVC(C=1, kernel="rbf", gamma="scale")
clf2.fit(X_train, y_train)
res = clf2.predict(X_test)
print("SVC score: ",clf2.score(X_test, y_test),", precison: ",precision_score(y_test, res),", recall: ",recall_score(y_test, res))

clf = GradientBoostingClassifier(random_state=0)
p = {
    #'loss': ['deviance', 'exponential'],
    'n_estimators': [100],
    'max_leaf_nodes':[4, 8, 16],
    'max_depth': [8,12,16],
    #'learning_rate': [0.5,1.,2.],
    'max_features':[3,4,5,6,None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'subsample': [0.4,0.5,0.625,0.75]
}
'''
print('---PRIMA FASE---')

clf_d = DecisionTreeClassifier()
clf_v = SVC()
clf_g = GaussianNB()

p = {}

print("---DecisionTreeClassifier 1° fase---")
for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_d = GridSearchCV(clf_d,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_d.fit(features_train_base, labels_train_base)
    print('Best score: {}'.format(grid_search_d.best_score_))
    print('Best parameters: {}'.format(grid_search_d.best_params_))

print("---GaussianNB 1° fase---")
for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_g = GridSearchCV(clf_g,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_g.fit(features_train_base, labels_train_base)
    print('Best score: {}'.format(grid_search_g.best_score_))
    print('Best parameters: {}'.format(grid_search_g.best_params_))

print("---SVC 1° fase---")
for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_v = GridSearchCV(clf_v,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_v.fit(features_train_base, labels_train_base)
    print('Best score: {}'.format(grid_search_v.best_score_))
    print('Best parameters: {}'.format(grid_search_v.best_params_))

print('---SECONDA FASE---')

clf_d2 = DecisionTreeClassifier()
clf_v2 = SVC()
clf_g2 = GaussianNB()

p = {}

print("---DecisionTreeClassifier 2° fase---")
for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_d2 = GridSearchCV(clf_d2,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_d2.fit(features_train, labels_train)
    print('Best score: {}'.format(grid_search_d2.best_score_))
    print('Best parameters: {}'.format(grid_search_d2.best_params_))

print("---GaussianNB 2° fase---")
for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_g2 = GridSearchCV(clf_g2,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_g2.fit(features_train, labels_train)
    print('Best score: {}'.format(grid_search_g2.best_score_))
    print('Best parameters: {}'.format(grid_search_g2.best_params_))

print("---SVC 2° fase---")
for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_v2 = GridSearchCV(clf_v2,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_v2.fit(features_train, labels_train)
    print('Best score: {}'.format(grid_search_v2.best_score_))
    print('Best parameters: {}'.format(grid_search_v2.best_params_))


print("---Fase finale---")


clf_final = DecisionTreeClassifier()

p={
 'random_state': [0],
 'criterion': ["gini", "entropy"],
 'max_depth': [8,10,16,32, None],
 'max_features':[1,2,3,4,5,6,None],
 'max_leaf_nodes':[4, 8, 64, 128, None],
 'splitter':['best', 'random']
}

for i in range(0,5):
    cross_validation = StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
    grid_search_f = GridSearchCV(clf_final,
                                param_grid= p,
                               cv=cross_validation,
                                n_jobs=-1,
                                verbose=0)

    grid_search_f.fit(features_train, labels_train)
    print('Best score: {}'.format(grid_search_f.best_score_))
    print('Best parameters: {}'.format(grid_search_f.best_params_))


print("---Creazione csv piattaforma kaggle---")
clf = DecisionTreeClassifier(random_state=0,max_depth=10,max_features=6,max_leaf_nodes=64,criterion='gini',splitter='best' ).fit(features_train, labels_train)
clf.fit(features_train, labels_train)

#prendo Cabin anche se poi non viene utilizzata, in modo
#da poter eseguire correttamente featureFormat
features_list1 = ["Cabin", "Sex", "Pclass","Onboard","Fare","Embarked","Age"]
data1 = featureFormat(dict1, features_list1 )
nada, test_features = targetFeatureSplit( data1 )
result=clf.predict(test_features)

with open('submission1.csv', 'w', newline='') as f:
    wr = csv.writer(f)

    wr.writerow(['PassengerId', 'Survived'])
    j = 892
    for x in result:
        wr.writerow([j, int(x)])
        j+=1
