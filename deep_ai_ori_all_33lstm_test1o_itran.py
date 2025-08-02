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
    iTransformer,
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



class iTransformer(iTransformer):
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
listcode = ["2809"]
codem="2809"
path="stock_data\\list_"
path2="output\\list_"
days_pred =100

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
df["mean"] = df['y'].ewm(span=days_pred*7, adjust=False).mean()
df["befor"] = df['y'].shift(1)
df["y"] = df["y"]/df["befor"]
df = df.dropna()
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
horizon = 1
# 学習データとテストデータに分割
train_df = df.iloc[:-7*days_pred]  # 最後の28行を除いたすべての行
test_df = df.iloc[-7*days_pred:]   # 最後の28行
# 予測期間
#horizon = 28
test_df = test_df.iloc[:horizon]

# モデルのパラメータ
rnn_params = {
    'input_size': 105 * horizon, 
    'h': horizon, 
    'n_series':105,
    #'futr_exog_list':["befor"],
    #'hist_exog_list':['mean'],
    #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
    #'scaler_type':'robust',
    'max_steps': 2500,
    'scaler_type':'standard',
    #'loss' : RMSE(), 
    'learning_rate':0.0001,
    'hidden_size':  1024,
    'n_heads':  32,
    'e_layers':  8,
    'd_layers':  4,
    'd_ff':  2048,
    'batch_size' : 2048,
}
# モデルのインスタンス
rnn_model = iTransformer(**rnn_params)
nf = NeuralForecast(models=[rnn_model], freq='h')
# モデルの学習
nf.fit(df=train_df)
nf.save(path='./checkpoints/test_iTransformer'+codem+'/',
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
    y_pred = Y_hat_df[Y_hat_df.index == unique_id]["iTransformer"]
    
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
actual = test_df[test_df["unique_id"] == codem]
predicted = Y_hat_df.loc[unique_id]
# 実際のデータと予測データを同じプロットに描画
ax.plot(actual["ds"], actual["y"], label='Actual')
ax.plot(predicted["ds"], predicted["iTransformer"], label='Predicted')
# タイトルとラベルの設定
ax.set_title(f"ID: {unique_id}")
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_ylim(bottom=0) 
#ax.legend()
#plt.tight_layout()
#plt.show()
df.to_csv(path2+'output_act'+codem+'.csv')
predicted.to_csv(path2+'output_pred'+codem+'.csv')
listdata = []
sumdata = 0
for j in range(0, days_pred-1):
    print(j)
    #listdata.append({'Close': df.loc[len(df)-horizon*20+j*7+6,"y"]})
    #for i in range(0, 6):
    i=-1
    train_df = df.iloc[:-7*days_pred+j*7+i+2]
    #print(train_df)
    #train_df2 = df.iloc[:-horizon*20+(j+1)*7+i+1]
    #train_df2 = train_df2.tail(7)
    Y_hat_df_sub = nf.predict(df=train_df)
    Y_hat_df = pd.DataFrame(Y_hat_df_sub)
    #Y_hat_df.to_csv(path2+'output_pred2'+codem+str(j)+str(i)+'.csv')
    #listdata.append({'Close': train_df.loc[len(train_df)-1,"y"]})
    last_price = train_df.tail(1)
    last_beforprice = last_price.iloc[0]['befor']
    last_price = last_price.iloc[0]['y'] *last_price.iloc[0]['befor']
    close_price = df.tail(7*days_pred-j*7-2-i)
    #print(close_price)
    close_price = close_price.head(1)
    close_beforprice = close_price.iloc[0]['befor']
    close_price = close_price.iloc[0]['y']*close_price.iloc[0]['befor']

    #close_price = df.loc[len(df)-7*days_pred+j*7+6,"y"]

    #price_data = last_price["y"].values
    #close_pricepre =  Y_hat_df.head(6-i)
    #close_pricepre = close_pricepre.tail(1)["TFT"].values
    close_pricepre =  Y_hat_df.head(1)["iTransformer"].values
    close_pricepre = close_pricepre[0]*close_beforprice
    if(last_price < close_pricepre):
        sumdata = sumdata + close_price-last_price
        listdata.append({'No':j,
            'Open': last_price ,
            'Close': close_price, 
            'preClose': close_pricepre,
            'preprice': close_price-last_price,
            'beforprice':close_beforprice
        })
    else:
        sumdata = sumdata - close_price+last_price
        listdata.append({'No':j,
            'Open': last_price ,
            'Close': close_price, 
            'preClose': close_pricepre,
            'preprice': -close_price+last_price,
            'beforprice':close_beforprice
        })
df_close = pd.DataFrame(listdata)
df_close.to_csv(path2+'output_pred2_iTransformer'+codem+"close"+'.csv')

print(sumdata)
print("happy")