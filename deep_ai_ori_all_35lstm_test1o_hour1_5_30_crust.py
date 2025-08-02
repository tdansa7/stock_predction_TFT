from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ニューラルネットワークの定義
class SingleOutputMultiElementNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleOutputMultiElementNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # ソフトマックス関数を追加

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        output = self.softmax(x)  # 出力にソフトマックス関数を適用
        return output


listcode = ["7201","7202","7203","7205","7211","7222","7261","7267","7269","7270","DAU","J_D","N225"]
listcode_stock = ["7201","7202","7203","7205","7211","7222","7261","7267","7269","7270"]
listcode_ind  = ["DAU","J_D","N225"]

path="stock_list\\"
path2="output\\list_"
path3="modelsaa\\list_"
days_pred =500

list_out = []
list_sel = []
for name in listcode_stock:
    list_out.append(name+'open_by_close')
for name in listcode_stock:
    list_sel.append(name+'open_div')
    list_sel.append(name+'close_div')
    for i in range(1, 11):
        list_sel.append(name+f'close_div{i}')
for name in listcode_ind:
    list_sel.append(name+'close_div')
    for i in range(1, 11):
        list_sel.append(name+f'close_div{i}')
name = "N225"
list_sel.append(name+'open_div')
list_sel.append('sum_open')

    




listdf = []

for codem in listcode:

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    df = df.drop(columns=['high',"low"])
    df['open_div'] = df['Open']/df["Close"].shift(1)
    df['close_div'] = df['Close'].shift(1)/df["Close"].shift(2)
    df['open_by_close'] = df['Close']/df['Open']
    df['open_by_close'] = df['open_by_close'].apply(lambda x: 1 if x > 1.01 else (0 if x < 0.99 else 0.5))
    for i in range(1, 11):
        df[f'close_div{i}'] = df['Close'].shift(1+i)/df["Close"].shift(2+i)


    df.columns = ['data' if col == 'data' else codem + col for col in df.columns]
    listdf.append(df)
merged_df = listdf[0]


for df in listdf[1:]:
    merged_df = merged_df.merge(df, on='data', how='inner')  # 'date'は共通の列名に置き換えてください
merged_df['data'] = pd.to_datetime(merged_df['data'])

# 空のリストを作成
data = []

# データをリストに追加
j = 0

for row in merged_df.itertuples():
    data.append({'ds': merged_df.at[0, 'data']+pd.Timedelta(days=j)})

    j = j+1

# 新しい列をデータフレームに追加
merged_df['ds'] = [item['ds'] for item in data]
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

merged_df = merged_df.drop(columns=['data'])
merged_df = merged_df.dropna()
print(merged_df)
merged_df['unique_id'] = "all"

horizon = 1
# 学習データとテストデータに分割
train_df = merged_df.iloc[:-days_pred]  # 最後の28行を除いたすべての行
test_df = merged_df.iloc[-days_pred:]   # 最後の28行

X = train_df[list_sel].values
y = train_df[list_out].values

# テンソルに変換
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# データローダーの設定
batch_size = 256
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルの初期化
input_size = len(list_sel)  # 入力パラメータの数
hidden_size = 1024  # 隠れ層のサイズ
output_size = len(list_out)  # 出力パラメータの数

model = SingleOutputMultiElementNN(input_size, hidden_size, output_size)

# ロス関数とオプティマイザ
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニングループ
num_epochs = 5000
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
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
output_df = pd.DataFrame(output.numpy().T, list_out)

# 出力結果をCSVに保存
output_df.to_csv(path2+'predicted_output.csv', index=False)
print("Predicted output saved to 'predicted_output.csv'")
