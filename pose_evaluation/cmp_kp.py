import pandas as pd
import numpy as np
from argparse import ArgumentParser

def AKD(gt, test):
    df1 = pd.read_pickle(gt)
    df2 = pd.read_pickle(test)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    #df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
    df2 = df2.sort_values(by=['file_name', 'frame_number'])

    scores = []

    for i in range(df1.shape[0]):
        file_name1 = df1['file_name'].iloc[i].split('.')[0]
        file_name2 = df2['file_name'].iloc[i].split('.')[0]
        file_name1 == file_name2
        df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
        if df2['value'].iloc[i] is not None: 
            scores.append(np.mean(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float)))

    print ("AKD Average difference: %.7s" % np.mean(scores))
