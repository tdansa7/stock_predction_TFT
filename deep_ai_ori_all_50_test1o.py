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



class TFT(TFT):
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
#listcode = [7201,7202,7203,7205,7211,7261,7267,7269,7270,7272,6902,6981,6103,6273,6305,6473,6471,3289,8801,8802]
listcode = [7201,7203,7211,7261,7267,7269,7270,7272]
codem="2809"
path="output_m\\list_"
path2="output\\list_"
days_pred =100

df = pd.read_csv(path+'predicted_output2_merged_df_a.csv')
#df = df[df['code'].isin(listcode)]
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

# データフレームをコード列ごとに分けて列を作成
df_pivot = train_df.pivot(index='ds', columns='code', values='div_25m')

# 結果を表示
print(df_pivot)

# 列同士の相関係数を計算
correlation_matrix = df_pivot.corr()

# 2. 上三角行列の絶対値の中から最大相関を見つける
# 対角線を除いた上三角部分のマスクを作成
mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

# マスクを適用し、相関行列の絶対値の中から最大値を取得
corr_matrix_abs = correlation_matrix.abs()
max_corr = corr_matrix_abs.where(mask).stack().idxmax()

# 3. 結果を表示
print(f"最も相関係数が高い2つの列: {max_corr}")
print(f"相関係数: {correlation_matrix.loc[max_corr]}")

# 例えば、'A'列との相関が高い上位10個の列を取得
target_column = 8306
top_10_corr = correlation_matrix[target_column].abs().sort_values(ascending=False).head(11)  # 自分自身（相関1.0）も含まれるため11個

# 自分自身を除いて上位10列を表示
top_10_corr = top_10_corr[top_10_corr.index != target_column]
top_10_columns = top_10_corr.index.tolist()
# 'A'列もリストに追加
top_10_columns.append(target_column)

# 結果を表示
print("Top 10 correlated columns:", top_10_columns)

print(top_10_corr)

# 結果を表示
print(correlation_matrix)

df_a = df[df['code'].isin(top_10_columns)]

# 最後から3日の日付を取得
unique_dates = df['date'].drop_duplicates()

# 最後の3つの一意な日付を取得
last_3_dates = unique_dates.nlargest(2)

# 最後から3日間の行を含むデータフレームを作成
test_df = df_a[df_a['date'].isin(last_3_dates)]

# 最後の3日以外の行を含むデータフレームを作成
train_df = df_a[~df_a['date'].isin(last_3_dates)]

# データフレームをコード列ごとに分けて列を作成
df_pivot = train_df.pivot(index='ds', columns='code', values='div_25m')
for date, group in train_df.groupby('date'):
    plt.figure(figsize=(10, 6))
    df_pivot = group.pivot(index='ds', columns='code', values='value')
    
    for column in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[column], label=column, marker='o')
    
    plt.title(f'Intraday Movement on {date}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Code')
    plt.grid(True)
    plt.show()

test_df = test_df.dropna()

# データフレームをコード列ごとに分けて列を作成
df_pivot = test_df.pivot(index='ds', columns='code', values='div_25m')
# データフレームをコード列ごとに分けて列を作成
df_pivot_C = test_df.pivot(index='ds', columns='code', values='Close')


# 各行の平均を計算し、新しい列 'row_mean' として追加
df_pivot['row_mean'] = df_pivot.mean(axis=1)

# インデックスで結合
combined_pivot = df_pivot.join(df_pivot_C, how='inner', lsuffix='_1', rsuffix='_2')

# 結果を表示
print(combined_pivot)


combined_pivot.to_csv(path+'predicted_output2_df_pivot.csv')

for date, group in train_df.groupby('date'):
    plt.figure(figsize=(10, 6))
    df_pivot = group.pivot(index='ds', columns='code', values='value')
    # 各行の平均を計算し、新しい列 'row_mean' として追加
    df_pivot['row_mean'] = df_pivot.mean(axis=1)
    
    for column in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[column], label=column, marker='o')
    
    plt.title(f'Intraday Movement on {date}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Code')
    plt.grid(True)
    plt.show()


#df1['date'] = pd.to_datetime(df1['date'])

df = df[df['code'].isin(listcode)]


# 空のリストを作成
data = []

# データをリストに追加
j = 0

for row in train_df.itertuples():
    data.append({'ds': train_df.at[row.Index, 'ds'], 
        'y': train_df.at[row.Index, 'div_25m'], 
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
        'y': test_df.at[row.Index, 'div_25m'], 
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
    'scaler_type':'standard',
    'max_steps': 500, 
    'learning_rate':0.001,
    'batch_size' : 256,
    'hidden_size': 256,
    'loss' : RMSE(),
}
# モデルのインスタンス
rnn_model = TFT(**rnn_params)
nf = NeuralForecast(models=[rnn_model], freq='min')
# モデルの学習
nf.fit(df=train_df_a)
#nf = NeuralForecast.load(path='./checkpoints/test_iTransformer_60'+codem+'/')

nf.save(path='./checkpoints/test_TFT_60'+codem+'/',
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
    y_pred = test_F[test_F.index == unique_id]["TFT"].values
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