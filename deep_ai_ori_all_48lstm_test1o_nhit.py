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
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
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
listcode = [7201,7202,7203,7205,7211,7261,7267,7269,7270,7272,6902,6981,6103,6273,6305,6473,6471,3289,8801,8802]
#listcode = [7201,7202,7203,7205,7211,7261,7267,7269,7270,7272]
codem="2809"
path="output_m\\list_"
path2="output\\list_"
days_pred =100

df = pd.read_csv(path+'predicted_output2_merged_df_a.csv')
df = df[df['code'].isin(listcode)]
# 'code' 列の一意の値を取得
#codes = df['code'].unique()

# 上から半分の値を選択
#half_index = len(codes) // 30
#selected_codes = codes[:half_index]
# 選択された 'code' の値に基づいてデータをフィルタリング
#df = df[df['code'].isin(selected_codes)]

# 売上などの時系列データ（sales_data.csv）
#df = pd.read_csv('sales_data.csv')
# 日付と時間を結合してタイムスタンプに変換
df['ds'] = pd.to_datetime(df['date'] + ' ' + df['time'])
# time列をdatetime形式に変換（時間のみ）
df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
df['date'] = pd.to_datetime(df['date'])

# 最後から3日の日付を取得
unique_dates = df['date'].drop_duplicates()

# 最後の3つの一意な日付を取得
last_3_dates = unique_dates.nlargest(2)

# 最後から3日間の行を含むデータフレームを作成
test_df = df[df['date'].isin(last_3_dates)]

# 最後の3日以外の行を含むデータフレームを作成
train_df = df[~df['date'].isin(last_3_dates)]
print(train_df)


#df1['date'] = pd.to_datetime(df1['date'])


# 空のリストを作成
data = []

# データをリストに追加
j = 0

for row in train_df.itertuples():
    data.append({'ds': train_df.at[row.Index, 'ds'], 
        'y': train_df.at[row.Index, 'div_5m'], 
        'unique_id': train_df.at[row.Index, 'ID'],
    })
    j = j+1
        

train_df_a = pd.DataFrame(data)
#df["mean"] = df['y'].ewm(span=days_pred*7, adjust=False).mean()
#df["befor"] = df['y'].shift(1)
#df["y"] = df["y"]/df["befor"]
train_df_a = train_df_a.dropna()
print(train_df_a)
train_df_a.to_csv(path+'predicted_output2_train_df.csv', index=False)

# 空のリストを作成
data = []

# データをリストに追加
j = 0

for row in test_df.itertuples():
    data.append({'ds': test_df.at[row.Index, 'ds'], 
        'y': test_df.at[row.Index, 'div_5m'], 
        'unique_id': test_df.at[row.Index, 'ID'],
    })
    j = j+1

test_df_a = pd.DataFrame(data)
#df["mean"] = df['y'].ewm(span=days_pred*7, adjust=False).mean()
#df["befor"] = df['y'].shift(1)
#df["y"] = df["y"]/df["befor"]
test_df_a = test_df_a.dropna()
print(train_df_a)

# 店舗特性データ（static_data.csv）の読み込み
#static_df = pd.read_csv('static_data.csv')
#print(static_df)

# unique_idごとにデータをグループ化
grouped = train_df_a.groupby('unique_id')
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
horizon = 30
#horizon_weight = [1] * horizon  # 均等な重みを持つリスト
# モデルのパラメータ
rnn_params = {
    'input_size':  1* horizon, 
    'h': horizon, 
    #'futr_exog_list':["befor"],
    #'hist_exog_list':['mean'],
    #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
    #'scaler_type':"robust",
    'max_steps': 500,
    'scaler_type':'standard',
    'loss' : RMSE(), 
    'learning_rate':0.000001,
    'stack_types':  ["identity", "identity","identity"],#["identity", "identity", "identity"]
    'n_blocks':  [ 1, 1,1],#[1, 1, 1]
    'mlp_units': 3 * [[512, 512]],#3 * [[512, 512]]
    'n_pool_kernel_size': [2, 2, 1],#[2, 2, 1]
    'n_freq_downsample': [4, 2, 1], #[4, 2, 1]
    'pooling_mode': 'AvgPool1d', #['MaxPool1d', 'AvgPool1d']
    'interpolation_mode':  'cubic',#['linear', 'nearest', 'cubic'].
    'batch_size' : 512,
    #'lr_scheduler':torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
}
# モデルのインスタンス
rnn_model = NHITS(**rnn_params)
nf = NeuralForecast(models=[rnn_model], freq='min')
# モデルの学習
nf.fit(df=train_df_a)
#nf = NeuralForecast.load(path='./checkpoints/test_iTransformer_60'+codem+'/')

nf.save(path='./checkpoints/test_NHITS_60'+codem+'/',
        model_index=None, 
        overwrite=True,
        save_dataset=True)

#loaded_rnn_model = nf.models[0]
# パラメータの確認
#print(loaded_rnn_model.hparams)  # モデルのハイパーパラメータを表示

# 予測の実施
#Y_hat_df = nf.predict(futr_df=test_df)
#Y_hat_df = nf.predict().reset_index()
#Y_hat_df = nf.predict()
#Y_hat_df["ds"] = test_df["ds"]
#print(Y_hat_df)
# 14時以前のデータに絞り込む
test_df_a_b = test_df_a[test_df_a['ds'].dt.time < pd.to_datetime('14:00', format='%H:%M').time()]
test_df_a_c = test_df_a[test_df_a['ds'].dt.time > pd.to_datetime('13:59', format='%H:%M').time()]
test_F =  nf.predict(df=test_df_a_b)
# 計算結果を格納
results = []
# 'unique_id'ごとに計算
for unique_id in test_df_a['unique_id'].unique():
    # 実測値を取得
    y_true = test_df_a_c[test_df_a_c['unique_id'] == unique_id]['y']
    y_true_time = test_df_a_c[test_df_a_c['unique_id'] == unique_id]['ds']
    
    # 予測値を取得
    y_pred = test_F[test_F.index == unique_id]["NHITS"].values
    y_pred_time = test_F[test_F.index == unique_id]['ds'].values

    # グラフの作成
    plt.figure(figsize=(10, 6))
    
    # 実測値をプロット
    plt.plot(y_true_time, y_true, label='Actual', marker='o')
    
    # 予測値をプロット
    plt.plot(y_pred_time, y_pred, label='Predicted', marker='x')
    
    # グラフのタイトルやラベルの設定
    plt.title(f'Actual vs Predicted for unique_id: {unique_id}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # グラフを画像として保存
    plt.savefig(path+"\\list_"+f'actual_vs_predicted_{unique_id}.png', dpi=300)  # 画像名と解像度を指定
    #plt.show()

    # 必要なら結果を格納
    results.append((unique_id, y_true, y_pred))
results_df = pd.DataFrame(
    results, 
    columns=['unique_id', 'true', 'pred'])
print(results_df)

print("happy")