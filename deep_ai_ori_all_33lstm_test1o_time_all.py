from sklearn.preprocessing import StandardScaler
import datetime

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
    Autoformer,
    TFT,
    TimesNet,
)
from neuralforecast.utils import AirPassengersDF
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import matplotlib.pyplot as plt
plt.style.use('ggplot') # グラフのスタイル
listcode = ["2229","2809","3231","4151"]
#codem="2229"
path="stock_data\\list_"
path2="output\\list_"
days_pred =100
data = []
nowaa = datetime.datetime.now()
for codem in listcode:
        

    df1 = pd.read_csv(path+codem+'_day_com_hou.csv')


    # 売上などの時系列データ（sales_data.csv）
    #df = pd.read_csv('sales_data.csv')
    df1['datetime'] = pd.to_datetime(df1['datetime'])


    # 空のリストを作成
    #data = []

    # データをリストに追加
    j = 0
    deltaday = nowaa - pd.Timedelta(hours=7*len(df1))

    for row in df1.itertuples():
        data.append({'ds': deltaday + pd.Timedelta(hours=7*j), 'y': df1.at[row.Index, 'Open'], 'unique_id': codem})
        for i in range(1, 6):
            # データをリストに追加
            data.append({'ds': deltaday+ pd.Timedelta(hours=7*j+i), 'y': df1.at[row.Index, f'hour{i}'], 'unique_id': codem})
        data.append({'ds': deltaday + pd.Timedelta(hours=7*j+6), 'y': df1.at[row.Index, 'Close'], 'unique_id': codem})
        j = j+1
            

df = pd.DataFrame(data)
df["mean"] = df['y'].ewm(span=days_pred*7, adjust=False).mean()
df["befor"] = df['y'].shift(-1)
df = df.dropna()
print(df)
df.to_csv(path2+'output_df'+codem+"close"+'.csv')
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
horizon = 1
# 学習データとテストデータに分割
#train_df = df.iloc[:-7*days_pred]  # 最後の28行を除いたすべての行
#test_df = df.iloc[-7*days_pred:]   # 最後の28行
last_day = df.tail(1)
last_day = last_day.iloc[0]['ds']
dayaaa = last_day - pd.Timedelta(hours=7*days_pred)

train_df = df[df.ds <= dayaaa]
test_df = df[df.ds > dayaaa]
# 予測期間
#horizon = 28
#test_df = test_df.iloc[:horizon]

# モデルのパラメータ
rnn_params = {
    'input_size': 120 * horizon, 
    'h': horizon, 
    #'futr_exog_list':["befor"],
    #'hist_exog_list':['mean'],
    #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
    'scaler_type':'robust',
    'max_steps': 1000, 
}
# モデルのインスタンス
rnn_model = TimesNet(**rnn_params)
nf = NeuralForecast(models=[rnn_model], freq='h')
# モデルの学習
nf.fit(df=train_df)
nf.save(path='./checkpoints/test_TimesNet'+codem+'/',
        model_index=None, 
        overwrite=True,
        save_dataset=True)

# 予測の実施
#Y_hat_df = nf.predict(futr_df=test_df)
#Y_hat_df = nf.predict().reset_index()
Y_hat_df = nf.predict()
#Y_hat_df["ds"] = test_df["ds"]
print(Y_hat_df)


#test_df[test_df["unique_id"] == codem]
listdata = []
sumdata = 0
for j in range(1, days_pred-1):
    print(j)
    dayaaa = last_day - pd.Timedelta(hours=7*j) +pd.Timedelta(hours=1) 

    train_df = df[df.ds <= dayaaa]
    test_df = df[df.ds > dayaaa]

    #listdata.append({'Close': df.loc[len(df)-horizon*20+j*7+6,"y"]})
    #for i in range(0, 6):
    i=0
    #train_df = df.iloc[:-7*days_pred+j*7+i+2]
    #print(train_df)
    #train_df2 = df.iloc[:-horizon*20+(j+1)*7+i+1]
    #train_df2 = train_df2.tail(7)
    Y_hat_df_sub = nf.predict(df=train_df)
    for codem in listcode:
        actual = Y_hat_df_sub[Y_hat_df_sub.index == codem]["TimesNet"]
        Y_hat_df = pd.DataFrame(actual)
        actual = test_df[test_df["unique_id"] == codem]
        #Y_hat_df.to_csv(path2+'output_pred2'+codem+str(j)+str(i)+'.csv')
        #listdata.append({'Close': train_df.loc[len(train_df)-1,"y"]})
        last_price = train_df[train_df["unique_id"] == codem].tail(1)
        last_price = last_price.iloc[0]['y']
        close_price = actual.head(1)
        #print(close_price)
        #close_price = close_price.head(1)
        close_price = close_price.iloc[0]['y']

        #close_price = df.loc[len(df)-7*days_pred+j*7+6,"y"]

        #price_data = last_price["y"].values
        #close_pricepre =  Y_hat_df.head(6-i)
        #close_pricepre = close_pricepre.tail(1)["TFT"].values
        close_pricepre =  Y_hat_df.head(1)["TimesNet"].values
        close_pricepre = close_pricepre[0]
        if(last_price < close_pricepre):
            sumdata = sumdata + close_price-last_price
            listdata.append({'No':j,
                'code':codem,
                'Open': last_price ,
                'Close': close_price, 
                'preClose': close_pricepre,
                'preprice': close_price-last_price
            })
        else:
            sumdata = sumdata - close_price+last_price
            listdata.append({'No':j,
                'code':codem,
                'Open': last_price ,
                'Close': close_price, 
                'preClose': close_pricepre,
                'preprice': -close_price+last_price
            })
df_close = pd.DataFrame(listdata)
df_close.to_csv(path2+'output_pred2'+codem+"close"+'.csv')

print(sumdata)
print("happy")