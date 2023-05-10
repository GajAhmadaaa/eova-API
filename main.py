import pandas as pd

def calculate_thresholds(dataset_path):
    #set weights to global
    global weights, treshold1_max

    # Read dataset
    df = pd.read_csv(dataset_path)

    # Extract train and label data
    train = df.iloc[:,0:-1] # needed for mean
    label = df.iloc[:,-1] # assuming the last column are label

    # Filter the rows where the label is 2 or 3
    df_label_2 = df[label == 2].iloc[:,0:-1]
    df_label_3 = df[label == 3].iloc[:,0:-1]

    # Calculate total ganas tot from df_label_3
    ganas_dari_tot_pilihan = [df_label_3.sum()[i]/df.shape[0] for i in range(train.shape[1])]
    total_ganas_tot = sum(ganas_dari_tot_pilihan)

    # Normalize from ganas_dari_tot_pilihan
    weights = [ganas_dari_tot_pilihan[i]/total_ganas_tot for i in range(train.shape[1])]

    # Calculate treshold 1
    treshold1 = pd.DataFrame()
    for i, j in zip(range(train.shape[1]), weights):
        column_name = df_label_2.columns[i]
        column_values = df_label_2.iloc[:,i].replace(1, j)
        treshold1[column_name] = column_values
    treshold1_max = treshold1.iloc[:, 1:].sum(axis=1).max()

    # Calculate treshold 2
    treshold2 = 2/df_label_3.iloc[:, 1:].sum().max()

    return treshold1_max, treshold2, weights

    # Predicting
def predict(data):
    #treshold1_max, treshold2, weights = calculate_thresholds("dataset.csv")
    data_weights = [w * d for w, d in zip(weights, data)]

    # Predicting
    if sum(data_weights) - treshold1_max > 0:
      if data[0] == 1:
        return int(3), float(f'{sum(data_weights[1:] * 100):.2f}'), float(f'{data_weights[0] * 100:.2f}')
      else:
        return int(3), float(f'{sum(data_weights):.2f}') * 100, int(0)
    else:
        return int(2), float(f'{sum(data_weights):.2f}') * 100, int(0)
