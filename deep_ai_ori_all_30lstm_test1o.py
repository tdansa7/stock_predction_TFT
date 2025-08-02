from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    RNN, 
    LSTM, 
    GRU, 
    NBEATS, 
    NHITS, 
    PatchTST,
)
from neuralforecast.utils import AirPassengersDF
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import matplotlib.pyplot as plt
plt.style.use('ggplot') # グラフのスタイル
listcode = ["2212"]
codem="2212"
path="stock_data\\list_"
df = pd.read_csv(path+codem+'_day_com_hou_2.csv')


# 売上などの時系列データ（sales_data.csv）
#df = pd.read_csv('sales_data.csv')
df['ds'] = pd.to_datetime(df['ds'])
print(df)

# 店舗特性データ（static_data.csv）の読み込み
#static_df = pd.read_csv('static_data.csv')
#print(static_df)

# unique_idごとにデータをグループ化
grouped = df.groupby('unique_id')
plt.figure(figsize=(12, 9))
# グループごとにグラフを描画
for name, group in grouped:
    plt.plot(group['ds'], group['y'], label=name)
plt.legend()
plt.title('Sales Over Time by Store')
plt.ylim(bottom=0)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

fig, axes = plt.subplots(nrows=5, figsize=(12, 50))
# グループと軸（作成したサブプロット）を順に反復処理します
for (name, group), ax in zip(grouped, axes):
    ax.plot(group['ds'], group['y'], label=name)
    ax.set_ylim(bottom=0)
    ax.set_title(name)  # タイトルをグループの名前に設定します
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
plt.tight_layout()
plt.show()

# 学習データとテストデータに分割
train_df = df[df['ds'] < '2024-01-01']
test_df = df[df['ds'] >= '2024-01-01']
# 予測期間
horizon = 30

# モデルのパラメータ
rnn_params = {
    'input_size': 4 * horizon, 
    'h': horizon, 
    #'futr_exog_list':['holiday','promo'],
    #'hist_exog_list':['promo'],
    #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
    'scaler_type':'robust',
    'max_steps': 50, 
}
# モデルのインスタンス
rnn_model = PatchTST(**rnn_params)
nf = NeuralForecast(models=[rnn_model], freq='B')
# モデルの学習
nf.fit(df=train_df)

# 予測の実施
#Y_hat_df = nf.predict(futr_df=test_df)
#Y_hat_df = nf.predict().reset_index()
Y_hat_df = nf.predict()
#Y_hat_df["ds"] = test_df["ds"]
print(Y_hat_df)


# 計算結果を格納
results = []
# 'unique_id'ごとに計算
for unique_id in test_df['unique_id'].unique():
    y_true = test_df[test_df['unique_id'] == unique_id]['y']
    y_pred = Y_hat_df[Y_hat_df.index == unique_id]['PatchTST']
    
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    results.append([unique_id, mae, mape, r2])
results_df = pd.DataFrame(
    results, 
    columns=['unique_id', 'MAE', 'MAPE', 'R2'])
print(results_df)

# 各unique_idごとにサブプロットを生成
fig, axes = plt.subplots(nrows=len(Y_hat_df.index.unique()), figsize=(12, 50))
#for ax, unique_id in zip(axes, Y_hat_df.index.unique()):
ax=axes
unique_id=Y_hat_df.index.unique()
# 現在のunique_idのデータを選択
#actual = test_df[test_df["unique_id"] == unique_id]
actual = test_df[test_df["unique_id"] == 2212]
predicted = Y_hat_df.loc[unique_id]
# 実際のデータと予測データを同じプロットに描画
ax.plot(actual["ds"], actual["y"], label='Actual')
ax.plot(predicted["ds"], predicted["PatchTST"], label='Predicted')
# タイトルとラベルの設定
ax.set_title(f"ID: {unique_id}")
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_ylim(bottom=0) 
ax.legend()
plt.tight_layout()
plt.show()
print("happy")