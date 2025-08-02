from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

path="stock_list_m_2\\"
path_day="stock_list2_k\\"
path2="output_m\\list_"
path3="modelsaa\\list_"
df_code = pd.read_csv(path+'codelist_csv.csv', encoding='shift-jis')
listcode_stock = df_code['code'].astype(str).tolist()


# ニューラルネットワークの定義
class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(DynamicNN, self).__init__()
        layers = []
        # 入力層
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        # 隠れ層
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # 出力層
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        #layers.append(nn.Softmax(dim=1))

        # モデルの層を設定
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

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
    print(codem)

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    df_day = pd.read_csv(path_day+codem+'_input.csv')#, encoding='shift-jis'
    df_day = df_day.replace("", np.nan).dropna().reset_index(drop=True)
    # 日付列をdatetime型に変換（必要に応じて）
    df['date'] = pd.to_datetime(df['date'])
    df_day['date'] = pd.to_datetime(df_day['date'])

    
    sumopen = df_day["Close"].shift(1) 
    for i in range(2, 26):
        sumopen = sumopen + df_day["Close"].shift(i)
    sumopen = sumopen*0.04
    df_day["ave_25"] = sumopen

    sumopen = df_day["Close"].shift(1) 
    for i in range(2, 6):
        sumopen = sumopen + df_day["Close"].shift(i)
    sumopen = sumopen*0.2
    df_day["ave_5"] = sumopen

    df = pd.merge(df, df_day[['date',"ave_5", "ave_25"]], on='date', how='left')

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
        group['div_5d'] = (group["Close"]-group["ave_5"])/group["ave_5"]
        group['div_25d'] = (group["Close"]-group["ave_25"])/group["ave_25"]
        sumopen = group["Close"]
        for i in range(1, 5):
            sumopen = sumopen + group["Close"].shift(i)
        sumopen = sumopen*0.2
        group["Close_ave_5"] = sumopen

        sumopen = group["Close"]
        for i in range(1, 25):
            sumopen = sumopen + group["Close"].shift(i)
        sumopen = sumopen*0.04
        group["Close_ave_25"] = sumopen

        group['div_5m'] = (group["Close"]-group["Close_ave_5"])/group["Close_ave_5"]
        group['div_5m_1'] = group['div_5m'].shift(1)
        group['div_25m'] = (group["Close"]-group["Close_ave_25"])/group["Close_ave_25"]
        group['div_25m_1'] = group['div_25m'].shift(1)

        group["value"] =  group["Close"].shift(-1) - group["Close"]
        group["value_10"] =  group["Close"].shift(-10) - group["Close"]
        group["value_10_u"] =  group["value_10"].apply(lambda x: 1 if x > 0 else 0)
        group["value_10_d"] =  group["value_10"].apply(lambda x: 1 if x < 0 else 0)
        group["value_h"] = group["value"]/group["Close_ave_5"]
        flag_a = 1
        plus_or = 0
        time_ind = 0
        group["flag"] = 0  # フラグ列を初期化
        # 各行をforループで進めながら処理
        for index, row in group.iterrows():
            # 任意の計算や処理をここに追加
            if time_ind <10 :
                time_ind += 1
                continue

            if flag_a == 1 :
                value_a = (group.at[index, "Close"] - group.at[index, "Close_ave_5"]) * (group.at[index-1, "Close"] - group.at[index-1, "Close_ave_5"])
                if value_a < 0:
                    flag_a = 0
                    plus_or = group.at[index, "Close"] - group.at[index, "Close_ave_5"]
            else:
                value_a = plus_or*(group.at[index, "Close"] - group.at[index-1, "Close"])
                if value_a < 0:
                    #value_b = plus_or*(((group.at[index, "Close"] - group.at[index, "Close_ave_5"])/group.at[index, "Close"])-0.001*(group.at[index, "Close"] - group.at[index, "Close_ave_5"])/abs(group.at[index, "Close"] - group.at[index, "Close_ave_5"]))
                    value_b = plus_or*(group.at[index, "Close"] - group.at[index,"Close_ave_5"])
                    if value_b>0:
                        flag_a = 1
                        group.at[index, "flag"] = sign(plus_or)
                    else:
                        flag_a = 1
            if group.at[index, "flag"] == 1:
                if pd.notna(group.at[index, "value"]):
                    sum_value = sum_value + (plus_or * group.at[index, "value"] / abs(plus_or))
                else:
                    group.at[index, "flag"] = 0
                




        # 各列に対して、連続するマイナス値のカウント列を追加
        group['close_div0'] = group["Close"]/group["Close_ave_5"]
        for i in range(1, 10):
            group[f'close_div{i}'] = group['close_div0'].shift(i)
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


    

merged_df.to_csv(path2+'predicted_output2_merged_df_a.csv', index=False)
print("Predicted output saved to 'predicted_output.csv'")
print(sum_value)

merged_df = merged_df.dropna().reset_index(drop=True)
# 列Aの値が1の行を抽出
filtered_df_p = merged_df[merged_df['div_25m'] > 0]
filtered_df_m = merged_df[merged_df['div_25m'] < 0]
#filtered_df_p = merged_df[merged_df["flag"] == 1]
#filtered_df_m = merged_df[merged_df["flag"] == -1]

list_u_d = ["value_10_u","value_10_d"]

for out in list_u_d:
    days_pred = 1000
    horizon = 1
    # 学習データとテストデータに分割


    # date列で最後に出現した値を取得
    #last_date = filtered_df_p['date'].iloc[-1]
    #print("最後に出現したdate:", last_date)

    # 最後に出現したdate値を含む行でデータフレームを作成
    #test_df = filtered_df_p[filtered_df_p['date'] == last_date]
    # 最後のdate以外の行を取得
    #train_df = filtered_df_p[filtered_df_p['date'] != last_date]

    # 最後から3日の日付を取得
    unique_dates = filtered_df_m['date'].drop_duplicates()

    # 最後の3つの一意な日付を取得
    last_3_dates = unique_dates.nlargest(2)

    # 最後から3日間の行を含むデータフレームを作成
    test_df = filtered_df_m[filtered_df_m['date'].isin(last_3_dates)]

    # 最後の3日以外の行を含むデータフレームを作成
    train_df = filtered_df_m[~filtered_df_m['date'].isin(last_3_dates)]
    print(train_df)

    list_sel = ['div_5m','div_5m_1','div_25m','div_25m_1','div_5d','div_25d','div_negative_streak','div_plus_streak','div_negative_streak_shift','div_plus_streak_shift']
    for i in range(0, 6):
        list_sel.append(f'close_div{i}')
    print(list_sel)
    list_out = [out]



    for ROWNAME in list_sel:
        AVEROW = test_df[ROWNAME].mean()
        STDROW = test_df[ROWNAME].std()
        test_df[ROWNAME+"ave"] = AVEROW
        test_df[ROWNAME+"std"] = STDROW
        train_df[ROWNAME+"ave"] = AVEROW
        train_df[ROWNAME+"std"] = STDROW
        filtered_df_p[ROWNAME+"std"] = STDROW
        filtered_df_p[ROWNAME+"ave"] = AVEROW
        test_df[ROWNAME] = (test_df[ROWNAME]-test_df[ROWNAME+"ave"])/test_df[ROWNAME+"std"]
        train_df[ROWNAME] = (train_df[ROWNAME]-train_df[ROWNAME+"ave"])/train_df[ROWNAME+"std"]
    #for ROWNAME in list_out:
    #    AVEROW = test_df[ROWNAME].mean()
    #    STDROW = test_df[ROWNAME].std()
    #    test_df[ROWNAME+"ave"] = AVEROW
    #    test_df[ROWNAME+"std"] = STDROW
    #    train_df[ROWNAME+"ave"] = AVEROW
    #    train_df[ROWNAME+"std"] = STDROW
    #    filtered_df_p[ROWNAME+"std"] = STDROW
    #    filtered_df_p[ROWNAME+"ave"] = AVEROW
    #    test_df[ROWNAME] = test_df[ROWNAME]/test_df[ROWNAME+"std"]
    #    train_df[ROWNAME] = train_df[ROWNAME]/train_df[ROWNAME+"std"]


    #train_df = filtered_df_p.iloc[:-days_pred]  # 最後の28行を除いたすべての行
    #test_df = filtered_df_p.iloc.iloc[-days_pred:]   # 最後の28行

    # 平均と標準偏差を計算
    #mean = train_df[columns_to_standardize].mean()
    #std = train_df[columns_to_standardize].std()

    # データの標準化
    #train_df[columns_to_standardize] = (train_df[columns_to_standardize] - mean) / std

    # 平均と標準偏差を計算
    #mean = test_df[columns_to_standardize].mean()
    #std = test_df[columns_to_standardize].std()

    # データの標準化
    #test_df[columns_to_standardize] = (test_df[columns_to_standardize] - mean) / std

    X = train_df[list_sel].values
    y = train_df[list_out].values
    xdf = pd.DataFrame(X, columns=list_sel)
    xdf.to_csv(path2+'predicted_output_xdf.csv', index=False)

    # ロジスティック回帰モデルの作成と訓練
    model = LogisticRegression()
    model.fit(X, y)

    # テストデータでの予測
    X_test = test_df[list_sel].values
    y_test = test_df[list_out].values
    y_pred = model.predict(X_test)

    print(y_pred)

    # 精度を評価
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 結果の表示
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)

