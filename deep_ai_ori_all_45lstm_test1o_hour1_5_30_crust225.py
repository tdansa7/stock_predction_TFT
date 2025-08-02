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
sum_value = 0
for codem in listcode_stock:

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    
    df = df.drop(columns=['high',"low"])
    # 日付ごとにグループ化する
    #grouped = df.groupby('date')
    # 空文字列を含む列を削除する
    #df = df.dropna(axis=0).reset_index(drop=True)
    #print(df.at[77, "time"])
    df = df.replace("", np.nan).dropna().reset_index(drop=True)
    #print(df.at[77, "time"])
    # 日付ごとにグループ化する
    grouped = df.groupby('date')
    #print(df)

    # 各日付ごとに処理する
    for df, group in grouped:
        #print(group)
        #df.loc[df.index[-1], "Close"] = 0
        group['div'] = group["Close"]-group["Close"].shift(1)
        sumopen = group["Close"]
        for i in range(1, 10):
            sumopen = sumopen + group["Close"].shift(i)
        sumopen = sumopen*0.1
        group["Close_ave"] = sumopen
        group["value"] =  group["Close"].shift(-1) - group["Close"]
        flag_a = 1
        plus_or = 0
        time_ind = 0
        group["flag"] = 0  # フラグ列を初期化
        group["flag_v"] = 0 
        # 各行をforループで進めながら処理
        for index, row in group.iterrows():
            # 任意の計算や処理をここに追加
            if time_ind <10 :
                time_ind += 1
                continue

            if flag_a == 1 :
                value_a = (group.at[index, "Close"] - group.at[index, "Close_ave"]) * (group.at[index-1, "Close"] - group.at[index-1, "Close_ave"])
                if value_a < 0:
                    flag_a = 0
                    plus_or = group.at[index, "Close"] - group.at[index, "Close_ave"]
            else:
                value_a = (group.at[index, "Close"] - group.at[index, "Close_ave"]) * (group.at[index-1, "Close"] - group.at[index-1, "Close_ave"])
                plus_or = group.at[index, "Close"] - group.at[index, "Close_ave"]
                if value_a < 0:
                    flag_a = 1
                    group.at[index, "flag"] = 1
                else:
                    flag_a = 1
            if group.at[index, "flag"] == 1:
                if pd.notna(group.at[index, "value"]):
                    sum_value = sum_value - (plus_or * group.at[index, "value"] / abs(plus_or))
                    group.at[index, "flag_v"] = - (plus_or * group.at[index, "value"] / abs(plus_or))





        # 各列に対して、連続するマイナス値のカウント列を追加


        group['div_negative_streak'] = count_negative_streak(group['div'])
        group['div_plus_streak'] = count_plus_streak(group['div'])
        group['div_negative_streak_shift'] = group['div_negative_streak'].shift(1)
        group['div_plus_streak_shift'] = group['div_plus_streak'].shift(1)




        if ind == 0:
            merged_df = group
        else:
            # データフレームを縦方向に結合
            merged_df = pd.concat([merged_df, group], axis=0)
        ind += 1
        #print(merged_df)


    

merged_df.to_csv(path2+'predicted_output2_merged_df.csv', index=False)
print("Predicted output saved to 'predicted_output.csv'")
print(sum_value)
