from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import matplotlib.pyplot as plt

\
path="stock_list_m_2\\"
path2="output_m\\list_"
path3="modelsaa\\list_"
df_code = pd.read_csv(path+'codelist_csv.csv', encoding='shift-jis')
listcode_stock = df_code['code'].astype(str).tolist()

# 連続するマイナス値のカウントを作成する関数
def count_negative_streak(column):
    streaks = []  # 各セルごとの連続マイナスのカウント
    count = 0     # 連続マイナスのカウント
    
    for val in column:
        if val < 0:
            count += 1
        else:
            count = 0
        streaks.append(count)
    
    return streaks
# 連続するマイナス値のカウントを作成する関数
def count_plus_streak(column):
    streaks = []  # 各セルごとの連続マイナスのカウント
    count = 0     # 連続マイナスのカウント
    
    for val in column:
        if val > 0:
            count += 1
        else:
            count = 0
        streaks.append(count)
    
    return streaks

ind = 0
for codem in listcode_stock:

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    
    df = df.drop(columns=['high',"low"])
    # 空文字列を含む列を削除する
    df = df.dropna(axis=0)
    #print(df)
    #df.loc[df.index[-1], "Close"] = 0
    df['div'] = df["Close"]-df["Close"].shift(1)
    # 各列に対して、連続するマイナス値のカウント列を追加

    df['div_negative_streak'] = count_negative_streak(df['div'])
    df['div_plus_streak'] = count_plus_streak(df['div'])
    df['div_negative_streak_shift'] = df['div_negative_streak'].shift(1)
    df['div_plus_streak_shift'] = df['div_plus_streak'].shift(1)




    if ind == 0:
        merged_df = df
    else:
        # データフレームを縦方向に結合
        merged_df = pd.concat([merged_df, df], axis=0)
    ind += 1


    

merged_df.to_csv(path2+'predicted_output2_merged_df.csv', index=False)
print("Predicted output saved to 'predicted_output.csv'")
