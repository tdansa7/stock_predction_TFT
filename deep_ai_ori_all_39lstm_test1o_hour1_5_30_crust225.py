from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np


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
def calculate_difference_up(row):
    col_prefix = row['max_column_up'][:4]
    return row[f'{col_prefix}Close'] - row[f'{col_prefix}Open']
def calculate_difference_down(row):
    col_prefix = row['max_column_down'][:4]
    return -row[f'{col_prefix}Close'] + row[f'{col_prefix}Open']

path="stock_list2\\"
path2="output\\list_"
path3="modelsaa\\list_"
df_code = pd.read_csv(path+'codelist_csv.csv', encoding='shift-jis')
listcode_stock = df_code['code'].astype(str).tolist()
listcode = listcode_stock

listcode_ind  = ["DAU","J_D","N225"]

listcode = listcode_stock +listcode_ind 


print(listcode_stock)
print(listcode_ind)

print(listcode)
listcode_rem = []


days_pred =500








listdf = []

for codem in listcode_stock:

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    if len(df)<1000:
        listcode_rem.append(codem)
        print(codem)
        continue
    if df['Open'].iloc[-1]>4000:
        listcode_rem.append(codem)
        print(codem)
        continue
    if df['Open'].iloc[-1]<2000:
        listcode_rem.append(codem)
        print(codem)
        continue
    df = df.drop(columns=['high',"low"])
    #df.loc[df.index[-1], "Close"] = 0
    df['open_div'] = df['Open']/df["Close"].shift(1)
    df['close_div'] = df['Close'].shift(1)/df["Close"].shift(2)
    df['open_by_close'] = df['Close']/df['Open']
    df['open_by_close_up'] = df['open_by_close'].apply(lambda x: 1 if x > 1 else 0)
    df['open_by_close_down'] = df['open_by_close'].apply(lambda x: 1 if x < 1 else 0)

    sumopen = df['Open']
    for i in range(1, 25):
        sumopen = sumopen + df['Open'].shift(i)
    sumopen = sumopen*0.04

    df['open_div_25'] = (df['Open']-sumopen)/sumopen
    
    df['date'] = pd.to_datetime(df['date'])
    #print(df)
    #for i in range(1, 11):
    #    df[f'close_div{i}'] = df['Close'].shift(1+i)/df["Close"].shift(2+i)


    df.columns = ['date' if col == 'date' else codem + col for col in df.columns]
    listdf.append(df)
for codem in listcode_ind:

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    df = df.drop(columns=['high',"low"])
    #df.loc[df.index[-1], "Close"] = 0
    df['open_div'] = df['Open']/df["Close"].shift(1)
    df['close_div'] = df['Close'].shift(1)/df["Close"].shift(2)
    df['open_by_close'] = df['Close']/df['Open']
    df['open_by_close_up'] = df['open_by_close'].apply(lambda x: 1 if x > 1 else 0)
    df['open_by_close_down'] = df['open_by_close'].apply(lambda x: 1 if x < 1 else 0)

    sumopen = df['Open']
    for i in range(1, 25):
        sumopen = sumopen + df['Open'].shift(i)
    sumopen = sumopen*0.04

    df['open_div_25'] = (df['Open']-sumopen)/sumopen
    
    df['date'] = pd.to_datetime(df['date'])
    #print(df)
    #for i in range(1, 11):
    #    df[f'close_div{i}'] = df['Close'].shift(1+i)/df["Close"].shift(2+i)


    df.columns = ['date' if col == 'date' else codem + col for col in df.columns]
    listdf.append(df)
merged_df = listdf[0]

# リスト内包表記を使用して複数の値を削除
listcode = [x for x in listcode if x not in listcode_rem]
# リスト内包表記を使用して複数の値を削除
listcode_stock = [x for x in listcode_stock if x not in listcode_rem]

list_out = []
list_sel = []
list_clas = []
target_columns_up = []
target_columns_down = []
for name in listcode_stock:
    list_out.append(name+'open_by_close')
    list_clas.append(name+'open_by_close_up')
    
for name in listcode_stock:
    target_columns_up.append(name+'open_by_close_up')
    target_columns_down.append(name+'open_by_close_down')

for name in listcode_stock:
    list_sel.append(name+'open_div')
    list_sel.append(name+'close_div')
    list_sel.append(name+'open_div_25')
    
    #for i in range(1, 5):
    #    list_sel.append(name+f'close_div{i}')
for name in listcode_ind:
    list_sel.append(name+'close_div')
    #for i in range(1, 5):
    #    list_sel.append(name+f'close_div{i}')
name = "N225"
list_sel.append(name+'open_div')
list_sel.append('sum_open')

    
# 標準化を適用しない列を指定
exclude_columns = list_out




for df in listdf[1:]:
    merged_df = merged_df.merge(df, on='date', how='left')  # 'date'は共通の列名に置き換えてください
#merged_df['date'] = pd.to_datetime(merged_df['date'])

# 空のリストを作成
date = []

# データをリストに追加
j = 0

for row in merged_df.itertuples():
    date.append({'ds': merged_df.at[0, 'date']+pd.Timedelta(days=j)})

    j = j+1

# 新しい列をデータフレームに追加
merged_df['ds'] = [item['ds'] for item in date]
sum_column = []
sum_column2 = []
for index, row in merged_df.iterrows():
    sum_value = 0
    for codea in listcode_stock:
        sum_value = sum_value + row[codea + 'Close'] - row[codea + 'Open']
    sum_column.append(sum_value)


merged_df['y'] = sum_column
merged_df['sum_open'] = 0
for codea in listcode_stock:
    merged_df['sum_open'] = merged_df['sum_open'] + merged_df[codea + 'Open'] - merged_df[codea + 'Close'].shift(1) 
print(merged_df)


merged_df.to_csv(path2+'output_pred'+codem+'.csv')

##############クラスタリング
classa_df = merged_df.iloc[:-days_pred]
print(classa_df)
classa_df =classa_df.dropna()
classa_df = classa_df[list_clas]
df_transposed = classa_df.transpose()

df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed[1:]
print(df_transposed)

merged_df = merged_df.drop(columns=['date','ds'])
merged_df = merged_df.dropna()
print(merged_df)
# 標準化を適用する列を選択
columns_to_standardize = [col for col in merged_df.columns if col not in exclude_columns]

#merged_df['unique_id'] = "all"

horizon = 1
# 学習データとテストデータに分割
train_df = merged_df.iloc[:-days_pred]  # 最後の28行を除いたすべての行
test_df = merged_df.iloc[-days_pred:]   # 最後の28行

# 平均と標準偏差を計算
mean = train_df[columns_to_standardize].mean()
std = train_df[columns_to_standardize].std()

# データの標準化
train_df[columns_to_standardize] = (train_df[columns_to_standardize] - mean) / std

# 平均と標準偏差を計算
mean = test_df[columns_to_standardize].mean()
std = test_df[columns_to_standardize].std()

# データの標準化
test_df[columns_to_standardize] = (test_df[columns_to_standardize] - mean) / std

print(list_sel)
X = train_df[list_sel].values
y = train_df[list_out].values
xdf = pd.DataFrame(X, columns=list_sel)
xdf.to_csv(path2+'predicted_output_xdf.csv', index=False)

# テンソルに変換
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# データローダーの設定
batch_size = 32
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
hidden_layers = [10*input_size, 5*input_size, output_size]

model = DynamicNN(input_size, hidden_layers, output_size)
# ロス関数とオプティマイザ
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# トレーニングループ
num_epochs = 2000
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
test_df_out = merged_df.iloc[-days_pred:]
test_df_out = test_df_out.drop(columns=list_out)
#test_df_out = test_df_out.drop(columns=target_columns_down)
output_df = pd.DataFrame(output.numpy(), columns=list_out)
test_df_out = test_df_out.reset_index(drop=True)  # 入力データのインデックスをリセット
output_df = output_df.reset_index(drop=True)  # 予測結果のインデックスをリセット
#result_df = pd.concat([test_df_out.reset_index(drop=True), output_df], axis=1)
result_df = pd.concat([test_df_out, output_df], axis=1)
result_df['max_column_up'] = result_df[list_out].idxmax(axis=1)
result_df['max_column_down'] = result_df[list_out].idxmin(axis=1)

#result_df['up_buy'] = result_df[result_df['max_column_up'].str[:4]+'Close']-result_df[result_df['max_column_up'].str[:4]+'Open']
#result_df['down_sell'] = -result_df[result_df['max_column_down']+'Close']+result_df[result_df['max_column_down']+'Open']
result_df['up_buy'] = result_df.apply(calculate_difference_up, axis=1)
result_df['down_sell'] = result_df.apply(calculate_difference_down, axis=1)

#result_df['up_buy'] = result_df.apply(apply_operation_up, axis=1)
#result_df['down_sell'] = result_df.apply(apply_operation_down, axis=1)

sum_a = result_df['up_buy'].sum()+result_df['down_sell'].sum()
print(sum_a)

# 出力結果をCSVに保存
result_df.to_csv(path2+'predicted_output.csv', index=False)
test_df_out.to_csv(path2+'predicted_output2_testdf.csv', index=False)
merged_df.to_csv(path2+'predicted_output2_merged_df.csv', index=False)
print("Predicted output saved to 'predicted_output.csv'")
