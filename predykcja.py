import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

Dane = []

file = open('daneDoPredykcji.txt', 'r')

file.readlines()
for line in file.readlines():
    readDataStep = 0
    tempStr = ""
    seasonStr = ""
    for char in line.rstrip():
        if char != ' ':
            if readDataStep == 0:
                tempStr += char

            elif readDataStep == 1:
                seasonStr += char
        else:
            readDataStep += 1

    Dane.append(float(tempStr))

    if seasonStr == "winter":
        Dane.append(int(1))

    elif seasonStr == "autumn":
        Dane.append(int(2))

    elif seasonStr == "spring":
        Dane.append(int(3))

    elif seasonStr == "summer":
        Dane.append(int(4))

file.close()

Dane = np.array(Dane)

prediction = loaded_model.predict(Dane)
print(prediction)