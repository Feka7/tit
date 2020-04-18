#!/usr/bin/python

import pickle
from csv import reader


def csv2dict(csv_file):
    data = {}
    with open(csv_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        if header != None:
            l = int(len(header))
            for row in csv_reader:
                tmp = {}
                for x in range(1, l):
                    tmp[header[x]] = row[x]
                data[row[0]] = tmp
        return data

#Inserire train.csv dopo aver eseguito il file con test.csv
dict = csv2dict("test.csv")


for x in dict.keys():

    ###Class###
    if dict[x]['Pclass'] == "1":
        dict[x]['Pclass'] = 3.
    elif dict[x]['Pclass'] == "2":
        dict[x]['Pclass'] = 2.
    else:
        dict[x]['Pclass'] = 1.

    ###Sex###
    if dict[x]['Sex']=="male":
        dict[x]['Sex'] = 1.
    else:
        dict[x]['Sex'] = 0.

    ###Familiari a bordo, se presenti 1 altrimenti 0

    if float(dict[x]['Parch']+dict[x]['SibSp']) == 0:
        dict[x]['Onboard']=0.
    else:
        dict[x]['Onboard']=1.

    ###Imbarco###
    if dict[x]['Embarked']=="S":
            dict[x]['Embarked']=1.
    elif dict[x]['Embarked']=="Q":
            dict[x]['Embarked']=2.
    else: dict[x]['Embarked']=3.

    ###Tariffa, se non presente = 0###
    if dict[x]['Fare'] == "":
        dict[x]['Fare']=0.
    else:
        dict[x]['Fare']=float(dict[x]['Fare'])

    ###Età divisa in 3 classi: giovani < 18\
    ###                        mezza Età < 40\
    ###                        adulti >= 40
    if dict[x]['Age'] == "":
        dict[x]['Age']=2.
    elif float(dict[x]['Age']) < 18:
        dict[x]['Age']=1.
    elif float(dict[x]['Age']) < 40:
        dict[x]['Age']=2.
    else:
        dict[x]['Age']=3.

    ###Cabina: se presente = 1, altrimenti 0
    if dict[x]['Cabin'] != "":
        dict[x]['Cabin']=0.
    elif dict[x]['Cabin'] == "" and dict[x]['Pclass'] == "1":
             dict[x]['Cabin']=0.
    else:
        dict[x]['Cabin']=1.


output = open('myfile2.pkl', 'wb')
pickle.dump(dict, output)
output.close()
