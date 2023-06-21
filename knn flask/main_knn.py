import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

def calculate_thresholds(dataset_path):
    global scaling, knn
    data = pd.read_csv(dataset_path)
    scaling = StandardScaler()

    for col in ["PERUT TERASA MEMBESAR", "PERUT KEMBUNG", "NYERI PERUT", "MUAL/ MUNTAH", 
    "NAFSU MAKAN MENURUN", "CEPAT KENYANG", "GANGGUAN BAK", "GANGGUAN BAB", "GANGGUAN MENSTRUASI", 
    "PENURUNAN BB"]:
        data[col] = scaling.fit_transform(data[col].values.reshape(-1,1))

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

    k_range = range(1,len(X_test)+1)
    scores = {}
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        scores[k] = accuracy_score(y_test, y_pred)
        scores_list.append(accuracy_score(y_test,y_pred))

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train,y_train)

def predict(data):# = [1,1,0,0,1,0,0,0,0,1,0]):
    df = pd.DataFrame(data)
    data2 = scaling.transform(df.values.reshape(-1,1))

    # Create your NumPy array
    arr = np.array(data2)

    # Convert the NumPy array into a list without the outer brackets
    lst = [val for sublist in arr for val in sublist]

    df2 = pd.DataFrame(lst)

    hasil = knn.predict(df2.T)

    # Predicting
    if hasil.item() == 2:
      return int(2)
    else:
      return int(3)