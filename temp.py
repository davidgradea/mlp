from sklearn import neural_network
import pandas as pd
import numpy as np

# INCARCARE DATE
dataset = pd.read_csv("glass.csv",delimiter=',')

dataset = np.asarray(dataset)

etichete = dataset[:, -1] #extragem ultima coloana, deoarece contine etichete
dataset=dataset[:, 1:-1] #extragem datele din coloanele 1 pana la penultima


date_train = dataset[0:113990]
etichete_train = etichete[0:113990]

date_test = dataset[113990:151987]
etichete_test = etichete[113990:151987]
# print('Date de antrenare:\n', date_train)
#print('Etichete de antrenare:\n', etichete_train)
# print('\nDate de testare:\n', date_test)
# print('Etichete de testare:\n', etichete_test)


# CREARE SI ANTRENARE MLP
clf = neural_network.MLPClassifier(hidden_layer_sizes=(9,5),learning_rate_init=0.1)
clf.fit(date_train, etichete_train)

# TESTARE MLP
predictii = clf.predict(date_test)

acc = 0
# ACURATETE
for i in range(len(etichete_test)):
    if etichete_test[i] == predictii[i]:
        acc = acc + 1
print('Acuratetea=' + str((acc / len(etichete_test)) * 100) + '%')