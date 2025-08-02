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
    Autoformer,
    TFT,
    TimesNet,
    TSMixerx,
)
from neuralforecast.utils import AirPassengersDF
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from neuralforecast.losses.pytorch import RMSE


class TSMixerx(TSMixerx):
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        self.optimizer=optimizer
        self.scheduler=scheduler
        return [optimizer], [scheduler]
    def training_step(self, batch, batch_idx):
        # トレーニングステップの処理
        windows = self._create_windows(batch, step="train")
        y_idx = batch["y_idx"]
        original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
        windows = self._normalization(windows=windows, y_idx=y_idx)

        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        output = self(windows_batch)
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(output=output, loc=y_loc, scale=y_scale)
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        if torch.isnan(loss):
            raise Exception("Loss is NaN, training stopped.")

        # 現在の学習率をログに記録
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path)
        model = cls(**checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

plt.style.use('ggplot') # グラフのスタイル
#listcode = ["2212","2229","2809","3231","4151","7203","8002","8591","9502","9531"]
listcode = ["7201","7202","7203","7205","7211","7222","7261","7267","7269","7270","DAU","J_D","N225"]
listcode_stock = ["7201","7202","7203","7205","7211","7222","7261","7267","7269","7270"]
listcode_ind  = ["DAU","J_D","N225"]
#codem="2229"
path="stock_list\\"
path2="output\\list_"
days_pred =500
#max_steps = 100

list_sel = []
for name in listcode_stock:
    list_sel.append(name+'open_div')
    list_sel.append(name+'close_div')
for name in listcode_ind:
    list_sel.append(name+'close_div')
name = "N225"
list_sel.append(name+'open_div')
list_sel.append('sum_open')

    




listdf = []

for codem in listcode:

    df = pd.read_csv(path+codem+'_input.csv')#, encoding='shift-jis'
    df = df.drop(columns=['high',"low"])
    df['open_div'] = df['Open']/df["Close"].shift(1)
    df['close_div'] = df['Close'].shift(1)/df["Close"].shift(2)

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

#df = pd.DataFrame(data)
#df["mean"] = df['y'].ewm(span=days_pred*7, adjust=False).mean()
#df["Open"] = df['Open'].shift(-1)
#df["RSI"] = df['RSI'].shift(-1)
#df["MACD"] = df['MACD'].shift(-1)
#df["MA"] = df['MA'].shift(-1)
#df['hour1'] = df['hour1'].shift(-1)
#df['hour2'] = df['hour2'].shift(-1)
#df['hour3'] = df['hour3'].shift(-1)
#df['hour4'] = df['hour4'].shift(-1)
#df['hour5'] = df['hour5'].shift(-1)
#df['SAR'] = df['SAR'].shift(-1)

merged_df = merged_df.drop(columns=['data'])
merged_df = merged_df.dropna()
print(merged_df)
merged_df['unique_id'] = "all"
# 店舗特性データ（static_data.csv）の読み込み
#static_df = pd.read_csv('static_data.csv')
#print(static_df)

# unique_idごとにデータをグループ化
grouped = merged_df.groupby('unique_id')
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
train_df = merged_df.iloc[:-days_pred]  # 最後の28行を除いたすべての行
test_df = merged_df.iloc[-days_pred:]   # 最後の28行
# 予測期間
#horizon = 28
test_df = test_df.iloc[:horizon]

# モデルのパラメータ
rnn_params = {
    'input_size': 10 * horizon, 
    'h': horizon, 

    'futr_exog_list':list_sel,
    #'hist_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3','hour4','hour5'],
    #'hist_exog_list':['hour5'],
    #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
    'scaler_type':'standard',
    'max_steps': 4000, 
    'learning_rate':0.0001,
    'batch_size' : 512,
    'n_series' : 1,
    'n_block' : 10,
    'ff_dim' : 1024,
    #'loss' : RMSE(),
    #'trainer_kwargs': trainer_kwargs
}
# モデルのインスタンス
rnn_model = TSMixerx(**rnn_params)
nf5 = NeuralForecast(models=[rnn_model], freq='D')
#print(nf5.lerning_rate)
# モデルの学習
#nf5.fit(df=train_df)
#nf5.save(path='./checkpoints/test_TSMixerx_day_car_on'+codem+'/',
#        model_index=None, 
#        overwrite=True,
#        save_dataset=True)
nf5 = NeuralForecast.load(path='./checkpoints/test_TSMixerx_day_car_on'+codem+'/')

# チェックポイントのパスを取得
#best_model_path = checkpoint_callback.best_model_path
#best_model_path='./checkpoints/TSMixerx_0 (2).ckpt'

# モデルの読み込み
#nf5 = TSMixerx.load_from_checkpoint(best_model_path)


listdata = []
sumdata = 0

for j in range(0, days_pred):
    print(j)
    #listdata.append({'Close': df.loc[len(df)-horizon*20+j*7+6,"y"]})

    train_df1 = merged_df.iloc[:-j-1]
    test_df = merged_df.iloc[-j-1:]   # 最後の28行
    test_df = test_df.iloc[:horizon]
    #train_df1.to_csv(path2+'output_pred_traindf'+codem+"close"+'.csv')
    #train_df2 = df.iloc[:-horizon*20+(j+1)*7+i+1]
    #train_df2 = train_df2.tail(7)
    Y_hat_df_sub = nf5.predict(df=train_df1,futr_df=test_df)
    Y_hat_df_5 = pd.DataFrame(Y_hat_df_sub)
    #Y_hat_df.to_csv(path2+'output_pred2'+codem+str(j)+str(i)+'.csv')
    #listdata.append({'Close': train_df1.loc[len(train_df1)-1,"y"]})

    last_price_close = merged_df.tail(j+1)
    last_price_close1 = last_price_close.head(1)
    last_price_close = last_price_close1.iloc[0]['y']
    last_price_7201 = last_price_close1.iloc[0]['7201Open']

    
    close_pricepre =  Y_hat_df_5.head(1)["TSMixerx"].values
    close_pricepre_5 = close_pricepre[0]
    #print(close_pricepre)

    if(close_pricepre_5 > 0):
        sumdata = sumdata + last_price_close
        listdata.append({'No':j,
            'presum': close_pricepre_5 ,
            'sum': last_price_close ,
            '7201Open': last_price_7201 ,
        })
    else:
        sumdata = sumdata - last_price_close
        listdata.append({'No':j,
            'presum': close_pricepre_5 ,
            'sum': last_price_close ,
            '7201Open': last_price_7201 ,
        })

        
df_close = pd.DataFrame(listdata)
df_close.to_csv(path2+'output_pred2'+codem+"close"+'.csv')
#train_df1.to_csv(path2+'output_pred_traindf'+codem+"close"+'.csv')
merged_df.to_csv(path2+'output_df1'+codem+"close"+'.csv')



print(sumdata)
print("happy")