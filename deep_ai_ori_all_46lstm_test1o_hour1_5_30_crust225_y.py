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



#merged_df.to_csv(path2+'predicted_output2_merged_df_a.csv', index=False)

merged_df = pd.read_csv(path2+'predicted_output2_merged_df_a.csv')
merged_df['date'] = pd.to_datetime(merged_df['date'])
print(merged_df)

merged_df = merged_df.dropna().reset_index(drop=True)
# 列Aの値が1の行を抽出
filtered_df_p = merged_df[(merged_df['div_25m'] > 0)&(merged_df['div_5m'] > 0)&(merged_df['div_5d'] > 0)&(merged_df['div_25d'] > 0)&(merged_df['div_plus_streak'] < 5)]
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
