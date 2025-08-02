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
        group["code"] = codem
        # 最初の行の日付を取得して、テキスト形式に変換
        first_date = group.iloc[0]['date']
        first_date_text = first_date.strftime('%Y-%m-%d')  # '%Y-%m-%d' は年-月-日形式

        group["ID"] = codem + first_date_text
        group['div'] = group["Close"]-group["Close"].shift(1)
        group['div_5d'] = (group["Close"]-group["ave_5"])/group["ave_5"]
        group['div_25d'] = (group["Close"]-group["ave_25"])/group["ave_25"]

        sumopen = group["Close"].rolling(window=10).mean()

        group["Close_ave_5"] = sumopen

        sumopen = group["Close"].rolling(window=30).mean()

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

        ####macd####

        # 短期EMA（Exponential Moving Average）の計算
        group["MACD1"] = group['Close'].ewm(span=10, adjust=False).mean()

        # 長期EMAの計算
        group["MACD2"] = group['Close'].ewm(span=30, adjust=False).mean()

        # MACD（短期EMA - 長期EMA）の計算
        group["MACD3"] = group["MACD1"] - group["MACD2"]

        # シグナルラインの計算
        group["MACD4"] = group["MACD3"].ewm(span=6, adjust=False).mean()

        # MACDヒストグラムの計算
        group["MACD5"] = group["MACD3"] - group["MACD4"]

        sumopen = group["Close"].rolling(window=30).mean()

        group["MACD1"] = group["MACD1"]/group['Close']
        group["MACD2"] = group["MACD2"]/group['Close']
        group["MACD3"] = group["MACD3"]/group['Close']
        group["MACD4"] = group["MACD4"]/group['Close']
        group["MACD5"] = group["MACD5"]/group['Close']
        group["MACD5_shift"] = group["MACD5"].shift(1)
        ############################################

        # ボリンジャーバンドのパラメータ
        window = 20  # 移動平均の期間
        num_std = 2  # 標準偏差の倍率

        # 中央線 (20日単純移動平均)
        group['SMA'] = group['Close'].rolling(window=window).mean()

        # 標準偏差
        group['STD'] = group['Close'].rolling(window=window).std()

        # 上部バンド
        group['Upper Band2'] = group['SMA'] + (num_std * group['STD'])

        # 下部バンド
        group['Lower Band2'] = group['SMA'] - (num_std * group['STD'])

        # 上部バンド
        group['Upper Band1'] = group['SMA'] + ( group['STD'])

        # 下部バンド
        group['Lower Band1'] = group['SMA'] - ( group['STD'])

        ################################################
        #RSI#
        # 前日との価格の変化を計算
        price_delta = group['Close'].diff()

        # 上昇幅と下落幅を計算
        gain = price_delta.where(price_delta > 0, 0)
        loss = -price_delta.where(price_delta < 0, 0)

        # 移動平均を計算
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        # RS（相対的な強さ）を計算
        rs = avg_gain / avg_loss

        # RSIを計算
        rsi = 100 - (100 / (1 + rs))
        group["RSI"] = rsi
        ###############################################



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
filtered_df_p = merged_df[(merged_df['div_25m'] > 0)&(merged_df['div_5m'] > 0)&(merged_df['div_5d'] > 0)&(merged_df['div_25d'] > 0)]
filtered_df_m = merged_df[merged_df['div_25m'] < 0]
#filtered_df_p = merged_df[merged_df["flag"] == 1]
#filtered_df_m = merged_df[merged_df["flag"] == -1]

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
unique_dates = filtered_df_p['date'].drop_duplicates()

# 最後の3つの一意な日付を取得
last_3_dates = unique_dates.nlargest(2)

# 最後から3日間の行を含むデータフレームを作成
test_df = filtered_df_p[merged_df['date'].isin(last_3_dates)]

# 最後の3日以外の行を含むデータフレームを作成
train_df = filtered_df_p[~merged_df['date'].isin(last_3_dates)]
print(train_df)

list_sel = ['div_5m','div_5m_1','div_25m','div_25m_1','div_5d','div_25d','div_negative_streak','div_plus_streak','div_negative_streak_shift','div_plus_streak_shift']
for i in range(0, 6):
    list_sel.append(f'close_div{i}')
print(list_sel)
list_out = ["value_10_u"]



#for ROWNAME in list_sel:
#    AVEROW = test_df[ROWNAME].mean()
#    STDROW = test_df[ROWNAME].std()
#    test_df[ROWNAME+"ave"] = AVEROW
#    test_df[ROWNAME+"std"] = STDROW
#    train_df[ROWNAME+"ave"] = AVEROW
#    train_df[ROWNAME+"std"] = STDROW
#    filtered_df_p[ROWNAME+"std"] = STDROW
#    filtered_df_p[ROWNAME+"ave"] = AVEROW
#    test_df[ROWNAME] = (test_df[ROWNAME]-test_df[ROWNAME+"ave"])/test_df[ROWNAME+"std"]
#    train_df[ROWNAME] = (train_df[ROWNAME]-train_df[ROWNAME+"ave"])/train_df[ROWNAME+"std"]
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


# テンソルに変換
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# データローダーの設定
batch_size = 8
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルの初期化
input_size = len(list_sel)  # 入力パラメータの数
#hidden_size = 1024  # 隠れ層のサイズ
output_size = len(list_out)  # 出力パラメータの数
# 隠れ層のサイズをリストで指定
#hidden_layers = [2048, 1024,512,256,128,64, 32]  # 隠れ層の数と各層のニューロン数を指定
print(input_size)
print(len(list_out))
hidden_layers = [10*input_size, input_size, output_size]

model = DynamicNN(input_size, hidden_layers, output_size)
# ロス関数とオプティマイザ
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# トレーニングループ
num_epochs = 500
best_loss = float('inf')  # 最小損失を追跡するための初期値
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved with loss: {best_loss}")

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

# 最小損失モデルのロード
#best_model = SimpleNN(input_size, hidden_size, num_classes, temperature)
model.load_state_dict(torch.load(best_model_path))
# モデルの保存
torch.save(model.state_dict(), path3)
print("Model saved to 'model.pth'")
# 新しいデータの予測
X = test_df[list_sel].values
new_data = torch.tensor(X, dtype=torch.float32)  # ダミー新規データ
model.eval()
with torch.no_grad():
    output = model(new_data)
    print(f"Predicted output: {output}")
# 出力結果をデータフレームに変換
#output = output.T
test_df_out = filtered_df_p[~filtered_df_p['date'].isin(last_3_dates)]
#test_df_out = test_df_out.drop(columns=target_columns_up)
#test_df_out = test_df_out.drop(columns=target_columns_down)
output_df = pd.DataFrame(output.numpy(), columns=list_out)
test_df_out = test_df_out.reset_index(drop=True)  # 入力データのインデックスをリセット
output_df = output_df.reset_index(drop=True)  # 予測結果のインデックスをリセット
#result_df = pd.concat([test_df_out.reset_index(drop=True), output_df], axis=1)
result_df = pd.concat([test_df_out, output_df], axis=1)

# 出力結果をCSVに保存
result_df.to_csv(path2+'predicted_output.csv', index=False)
test_df_out.to_csv(path2+'predicted_output2_testdf.csv', index=False)
print("Predicted output saved to 'predicted_output.csv'")
