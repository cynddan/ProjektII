import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np


MODEL_LABELS_FILENAME = "model_labels.dat"

Temperatura = []
PoraRoku = []
Decyzja = []
Data = []

maxTemp = -1000
minTemp = 1000
maxSeason = -1000
minSeason = 1000

def Max(a, b):
    if a > b:
        return a
    else:
        return b

def Min(a, b):
    if a < b:
        return a
    else:
        return b

#Odczyt Danych 
file = open("trainingData.txt", "r")

for line in file.readlines():

    tempStr = ""
    seasonStr = ""
    decisionStr = ""

    readDataStep = 0

    for char in line.rstrip():
        if char != ' ':
            if readDataStep == 0:
                tempStr += char

            elif readDataStep == 1:
                seasonStr += char

            elif readDataStep == 2:
                decisionStr += char
        else:
            readDataStep += 1
    
    Temperatura.append(float(tempStr))
    Decyzja.append(float(decisionStr))

    if seasonStr == "winter":
        PoraRoku.append(int(1))

    elif seasonStr == "autumn":
        PoraRoku.append(int(2))

    elif seasonStr == "spring":
        PoraRoku.append(int(3))

    elif seasonStr == "summer":
        PoraRoku.append(int(4))


    file.close()

#Normalizacja Danych
i = 0
while i < len(Temperatura):
    maxTemp = max(Temperatura[i], maxTemp)
    minTemp = min(Temperatura[i], minTemp)

    maxSeason = max(PoraRoku[i], maxSeason)
    minSeason = min(PoraRoku[i], minSeason)

    i += 1

i = 0
while i < len(Temperatura):
    Temperatura[i] = (Temperatura[i] - minTemp) / (maxTemp - minTemp)
    PoraRoku[i] = (PoraRoku[i] - minSeason) / (maxSeason - minSeason)

    i += 1

#Przygotowanie danych do przetworzenia przez sieć
for dana in range(len(Temperatura)):
    Data.append([Temperatura[dana],PoraRoku[dana]])

Data = np.array(Data)
Label = np.array(Decyzja)


print(Label.shape)
print(Data.shape)

(X_train, X_test, Y_train, Y_test) = train_test_split(Data, Label, random_state = 42)

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)


with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

#Tworzenie sieci
model = Sequential()
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=3, batch_size=1)


# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
model.save("model.h5")


model.save_weights("model_weight.h5")
print("Saved model to disk")