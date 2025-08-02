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
path2="output\\list_"
days_pred =20

df1 = pd.read_csv(path+codem+'_day_com_hou.csv')


# 売上などの時系列データ（sales_data.csv）
#df = pd.read_csv('sales_data.csv')
df1['datetime'] = pd.to_datetime(df1['datetime'])


# 空のリストを作成
data = []

# データをリストに追加
j = 0

for row in df1.itertuples():
    data.append({'ds': df1.at[0, 'datetime']+pd.Timedelta(hours=7*j), 'y': df1.at[row.Index, 'Open'], 'unique_id': codem})
    for i in range(1, 6):
        # データをリストに追加
        data.append({'ds': df1.at[0, 'datetime']+ pd.Timedelta(hours=7*j+i), 'y': df1.at[row.Index, f'hour{i}'], 'unique_id': codem})
    data.append({'ds': df1.at[0, 'datetime'] + pd.Timedelta(hours=7*j+6), 'y': df1.at[row.Index, 'Close'], 'unique_id': codem})
    j = j+1
        

df = pd.DataFrame(data)
print(df)
# 店舗特性データ（static_data.csv）の読み込み
#static_df = pd.read_csv('static_data.csv')
#print(static_df)

# unique_idごとにデータをグループ化
grouped = df.groupby('unique_id')
#plt.figure(figsize=(12, 9))
# グループごとにグラフを描画
#for name, group in grouped:
#    plt.plot(group['ds'], group['y'], label=name)
#plt.legend()
#plt.title('Sales Over Time by Store')
#plt.ylim(bottom=0)
#plt.xlabel('Date')
#plt.ylabel('Sales')
#plt.show()

#fig, axes = plt.subplots(nrows=5, figsize=(12, 50))
# グループと軸（作成したサブプロット）を順に反復処理します
#for (name, group), ax in zip(grouped, axes):
#    ax.plot(group['ds'], group['y'], label=name)
#    ax.set_ylim(bottom=0)
#    ax.set_title(name)  # タイトルをグループの名前に設定します
#    ax.set_xlabel('Date')
#    ax.set_ylabel('Sales')
#    ax.legend()
#plt.tight_layout()
#plt.show()
horizon = 7
# 学習データとテストデータに分割
train_df = df.iloc[:-horizon*days_pred]  # 最後の28行を除いたすべての行
test_df = df.iloc[-horizon*days_pred:]   # 最後の28行
# 予測期間
#horizon = 28
test_df = test_df.iloc[:horizon]

# モデルのパラメータ
rnn_params = {
    'input_size': 20 * horizon, 
    'h': horizon, 
    #'futr_exog_list':['holiday','promo'],
    #'hist_exog_list':['promo'],
    #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
    'scaler_type':'robust',
    'max_steps': 50, 
}
# モデルのインスタンス
rnn_model = PatchTST(**rnn_params)
nf = NeuralForecast(models=[rnn_model], freq='h')
# モデルの学習
nf.fit(df=train_df)
nf.save(path='./checkpoints/test_run_PatchTST/',
        model_index=None, 
        overwrite=True,
        save_dataset=True)

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
actual = test_df[test_df["unique_id"] == "2212"]
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
df.to_csv(path2+'output_act'+codem+'.csv')
predicted.to_csv(path2+'output_pred'+codem+'.csv')
listdata = []
sumdata = 0
for j in range(0, days_pred):
    #listdata.append({'Close': df.loc[len(df)-horizon*20+j*7+6,"y"]})
    for i in range(0, 6):
        train_df = df.iloc[:-horizon*20+j*7+i+1]
        Y_hat_df_sub = nf.predict(df=train_df)
        Y_hat_df = pd.DataFrame(Y_hat_df_sub)
        #Y_hat_df.to_csv(path2+'output_pred2'+codem+str(j)+str(i)+'.csv')
        listdata.append({'Close': train_df.loc[len(train_df)-1,"y"]})
        last_price = train_df.loc[len(train_df)-1,"y"]
        close_price = df.loc[len(df)-horizon*20+j*7+6,"y"]
    
        #price_data = last_price["y"].values
        close_pricepre =  Y_hat_df.head(6-i)
        close_pricepre = close_pricepre.tail(1)["PatchTST"].values

        if(last_price < close_pricepre):
            sumdata = sumdata + close_price-last_price
        else:
            sumdata = sumdata - close_price+last_price
df_close = pd.DataFrame(listdata)
df_close.to_csv(path2+'output_pred2'+codem+"close"+'.csv')

print(sumdata)
print("happy")