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
    FEDformer,
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

class CustomFEDformer(FEDformer):
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.01)
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

plt.style.use('ggplot') # グラフのスタイル
#listcode = ["2212","2229","2809","3231","4151","7203","8002","8591","9502","9531"]
listcode = ["2809"]
#codem="2229"
path="stock_data\\list_"
path2="output\\list_"
days_pred =100

for codem in listcode:

    df = pd.read_csv(path+codem+'_day_com_hou.csv')


    # 売上などの時系列データ（sales_data.csv）
    #df = pd.read_csv('sales_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    ##指標
    #RSI#
    # 前日との価格の変化を計算
    price_delta = df['Close'].diff()

    # 上昇幅と下落幅を計算
    gain = price_delta.where(price_delta > 0, 0)
    loss = -price_delta.where(price_delta < 0, 0)

    # 移動平均を計算
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    # RS（相対的な強さ）を計算
    rs = avg_gain / avg_loss

    # RSIを計算
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi.shift(1)


    ##ストキャ##

    df["STK1"] = 100 * ((df['Open'] - df['low'].shift(1).rolling(window=14).min()) / (df['high'].shift(1).rolling(window=14).max() - df['low'].shift(1).rolling(window=14).min()))
    df["STK2"] = df["STK1"].rolling(window=3).mean()
    df["STK3"] = df["STK1"] - df["STK2"]

    #df['Lowest_Low'] = df['low'].shift(1).rolling(window=14).min()
    #df['Highest_High'] = df['high'].shift(1).rolling(window=14).max()

    #df["STK1"] = 100 * ((df['Open'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']))
    #df["STK2"] = (df['Open'].rolling(window=3).sum() - df['Lowest_Low'].rolling(window=3).sum()) / (df['Highest_High'].rolling(window=3).sum()-df['Lowest_Low'].rolling(window=3).sum())
    #df["STK3"] = df["STK2"].rolling(window=3).mean()
    #df["STK4"] = df["STK1"] - df["STK2"]
    #df["STK5"] = df["STK2"] - df["STK3"]

    #df.drop(['Lowest_Low', 'Highest_High'], axis=1, inplace=True)

    ##サイコロジカルライン
    # 前日との比較で上昇日と下落日を判定
    up_days = (df['Close'] > df['Close'].shift(1)).astype(int)
    down_days = (df['Close'] < df['Close'].shift(1)).astype(int)

    # 上昇日と下落日の割合を計算
    up_ratio = up_days.rolling(window=14).mean()
    down_ratio = down_days.rolling(window=14).mean()

    # サイコロジカル・ラインの計算
    cycolg = up_ratio / (up_ratio + down_ratio) * 100
    df["cycolg"] = cycolg.shift(1)
    #df.drop(["cycolg"], axis=1, inplace=True)


    ##RCI

    rci_results = pd.DataFrame()
    #df["RCI"] = None

    for i in range(0, len(df)-15):
        subset = df['Close'].iloc[i:i+14]
        
        # 各変数について順位を計算
        #ranks = subset.apply(rankdata)
        ranked_df = subset.rank(ascending=False, method='max')
        
        # 各変数間のランキングの相関を計算
        rci = np.corrcoef(ranked_df.index, ranked_df.values)[0, 1]
        
        # 結果を結合
        #rci_results = pd.concat([rci_results, rci])
        #print(ranked_df.index)
        #print(ranked_df.values)
        df.loc[i+15, "RCI"] =float(rci)



    ####macd####

    # 短期EMA（Exponential Moving Average）の計算
    df["MACD1"] = df['Open'].ewm(span=10, adjust=False).mean()

    # 長期EMAの計算
    df["MACD2"] = df['Open'].ewm(span=20, adjust=False).mean()

    # MACD（短期EMA - 長期EMA）の計算
    df["MACD3"] = df["MACD1"] - df["MACD2"]

    # シグナルラインの計算
    df["MACD4"] = df["MACD3"].ewm(span=6, adjust=False).mean()

    # MACDヒストグラムの計算
    df["MACD5"] = df["MACD3"] - df["MACD4"]
    sumopen = df['Open']
    for i in range(1, 25):
        sumopen = sumopen + df['Open'].shift(i)
    sumopen = sumopen*0.04
    df["MACD1"] = (df["MACD1"]-sumopen)/sumopen
    df["MACD2"] = (df["MACD2"]-sumopen)/sumopen
    df["MACD3"] = (df["MACD3"]-sumopen)/sumopen
    df["MACD4"] = (df["MACD4"]-sumopen)/sumopen
    df["MACD5"] = (df["MACD5"]-sumopen)/sumopen





    ########パラボリック

    # 初期値の設定 
    max_acceleration=0.2
    acceleration=0.02
    df['SAR'] = df['high']
    df['AF'] = acceleration
    df['EP'] = df['high']

    # トレンドの方向を示す列
    trend_direction = 1  # 1: 上昇トレンド, -1: 下降トレンド

    # パラボリックSARの計算
    for i in range(2, len(df)-1):
        if trend_direction == 1:
            if df['low'].iloc[i] < df['SAR'].iloc[i - 1]:
                df['SAR'].iloc[i] = df['EP'].iloc[i - 1]
                df['AF'].iloc[i] = acceleration
                trend_direction = -1
                df['EP'].iloc[i] = df['high'].iloc[i]
            else:
                if df['high'].iloc[i] > df['EP'].iloc[i - 1]:
                    df['EP'].iloc[i] = df['high'].iloc[i]
                    df['AF'].iloc[i] = min(df['AF'].iloc[i - 1] + acceleration, max_acceleration)
                else:
                    df['EP'].iloc[i] = df['EP'].iloc[i - 1]
                df['SAR'].iloc[i] = df['SAR'].iloc[i - 1] + df['AF'].iloc[i] * (df['EP'].iloc[i] - df['SAR'].iloc[i - 1])

        else:  # trend_direction == -1
            if df['high'].iloc[i] > df['SAR'].iloc[i - 1]:
                df['SAR'].iloc[i] = df['EP'].iloc[i - 1]
                df['AF'].iloc[i] = acceleration
                trend_direction = 1
                df['EP'].iloc[i] = df['low'].iloc[i]
            else:
                if df['low'].iloc[i] < df['EP'].iloc[i - 1]:
                    df['EP'].iloc[i] = df['low'].iloc[i]
                    df['AF'].iloc[i] = min(df['AF'].iloc[i - 1] + acceleration, max_acceleration)
                else:
                    df['EP'].iloc[i] = df['EP'].iloc[i - 1]
                df['SAR'].iloc[i] = df['SAR'].iloc[i - 1] + df['AF'].iloc[i] * (df['EP'].iloc[i] - df['SAR'].iloc[i - 1])
        df.to_csv('output2'+codem+'.csv')
    sumopen = df['Open']
    df['SAR'] = df['SAR'].shift(1)
    df['AF'] = df['AF'].shift(1)
    df['EP'] = df['EP'].shift(1)
    df.drop(['AF'], axis=1, inplace=True)

    for i in range(1, 25):
        sumopen = sumopen + df['Open'].shift(i)
    sumopen = sumopen*0.04
    df['EP'] = (df['EP']-sumopen)/sumopen
    df['SAR'] = (df['SAR']-sumopen)/sumopen

    ##ADX
    df['HL'] = df['high'] - df['low']
    df['HC'] = abs(df['high'] - df['Close'].shift(1))
    df['LC'] = abs(df['low'] - df['Close'].shift(1))
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df.drop(['HL', 'HC', 'LC'], axis=1, inplace=True)

    #df['TR'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['Close'].shift(1)), abs(x['low'] - x['Close'].shift(1))), axis=1)

    # Directional Movement (DM)の計算
    df['HC'] = df['high'] - df['high'].shift(1)
    df['LC'] = df['low'].shift(1) - df['low']
    df['DM_plus'] = df.apply(lambda x: max(x['HC'], 0), axis=1)
    df['DM_minus'] = df.apply(lambda x: max(x['LC'], 0), axis=1)
    df.drop(['HC', 'LC'], axis=1, inplace=True)

    # Smoothed True Range (ATR)の計算
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()

    # Smoothed Directional Movement (ADM)の計算
    df['ADM_plus'] = df['DM_plus'].ewm(span=14, adjust=False).mean()
    df['ADM_minus'] = df['DM_minus'].ewm(span=14, adjust=False).mean()

    # Directional Index (DI)の計算
    df['DI_plus'] = (df['ADM_plus'] / df['ATR']) * 100
    df['DI_minus'] = (df['ADM_minus'] / df['ATR']) * 100

    # Relative Strength (RSI)の計算
    df['RSI1'] = (df['DI_plus']-df['DI_minus']) / (df['DI_plus']+df['DI_minus'])
    df['DX'] = abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus']) * 100

    # ADXの計算
    df['ADX1'] = df['DX'].ewm(span=14, adjust=False).mean() * 100
    df['ADX2'] = df['DX'].ewm(span=28, adjust=False).mean() * 100
    df['ADX3'] = df['ADX1'] - df['ADX2']

    # 不要な列の削除
    df.drop(['TR', 'DM_plus', 'DM_minus', 'ADM_plus', 'ADM_minus','DX'], axis=1, inplace=True)

    df['ATR'] = (df['ATR'].shift(1)-sumopen)/sumopen
    df['DI_plus'] = df['DI_plus'].shift(1)
    df['DI_minus'] = df['DI_minus'].shift(1)
    df['RSI1'] = df['RSI1'].shift(1)
    df['ADX1'] = df['ADX1'].shift(1)
    df['ADX2'] = df['ADX2'].shift(1)
    df['ADX3'] = df['ADX3'].shift(1)

    ##強弱レシオ

    df['HC'] = df['high'] - df['Open']
    df['LC'] = df['Open'] - df['low']
    df['DM_plus'] = df.apply(lambda x: max(x['HC'], 0), axis=1)
    df['DM_minus'] = df.apply(lambda x: max(x['LC'], 0), axis=1)

    df['a_resio'] = 100*df['DM_plus'].rolling(window=20).sum()/df['DM_minus'].rolling(window=20).sum()

    df['HC'] = df['high'] - df['Close'].shift(1)
    df['LC'] = df['Close'].shift(1) - df['low']
    df['DM_plus'] = df.apply(lambda x: max(x['HC'], 0), axis=1)
    df['DM_minus'] = df.apply(lambda x: max(x['LC'], 0), axis=1)

    df['b_resio'] = 100*df['DM_plus'].rolling(window=20).sum()/df['DM_minus'].rolling(window=20).sum()

    df['HC'] = df['high'] - 0.5*(df['high'].shift(1)+df['low'].shift(1))
    df['LC'] = 0.5*(df['high'].shift(1)+df['low'].shift(1)) - df['low']
    df['DM_plus'] = df.apply(lambda x: max(x['HC'], 0), axis=1)
    df['DM_minus'] = df.apply(lambda x: max(x['LC'], 0), axis=1)


    df['c_resio'] = 100*df['DM_plus'].rolling(window=20).sum()/df['DM_minus'].rolling(window=20).sum()
    df.drop(['HC', 'LC','DM_plus','DM_minus'], axis=1, inplace=True)
    df['a_resio'] = df['a_resio'].shift(1)
    df['b_resio'] = df['b_resio'].shift(1)
    df['c_resio'] = df['c_resio'].shift(1)




    #####


    # 空のリストを作成
    data = []

    # データをリストに追加
    j = 0

    for row in df.itertuples():
        data.append({'ds': df.at[0, 'datetime']+pd.Timedelta(days=j), 
            'y': df.at[row.Index, 'Close'], 
            'unique_id': codem,
            'Open':df.at[row.Index, 'Open'] ,
            "RSI": df.at[row.Index, 'RSI'],
            'MACD': df.at[row.Index, "MACD5"],
            'MA': df.at[row.Index, "MACD2"],
            'hour1': df.at[row.Index, "hour1"],
            'hour2': df.at[row.Index, "hour2"],
            'hour3': df.at[row.Index, "hour3"],
            'hour4': df.at[row.Index, "hour4"],
            'hour5': df.at[row.Index, "hour5"],
            'SAR': df.at[row.Index, 'SAR'],
        })
        j = j+1
            

    df = pd.DataFrame(data)
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
    train_df = df.iloc[:-days_pred]  # 最後の28行を除いたすべての行
    test_df = df.iloc[-days_pred:]   # 最後の28行
    # 予測期間
    #horizon = 28
    test_df = test_df.iloc[:horizon]

    # モデルのパラメータ
    rnn_params = {
        'input_size': 40 * horizon, 
        'h': horizon, 
        'futr_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'hour5'],
        #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
        'scaler_type':'robust',
        'max_steps': 1000, 
        'learning_rate':0.001,
        'batch_size' : 8,
        'windows_batch_size':512,
        'inference_windows_batch_size':512,
        'hidden_size' : 256,
    }
    # モデルのインスタンス
    rnn_model = CustomFEDformer(**rnn_params)
    nf5 = NeuralForecast(models=[rnn_model], freq='D')
    #print(nf5.lerning_rate)
    # モデルの学習
    nf5.fit(df=train_df)
    nf5.save(path='./checkpoints/test_FEDformer_day_hour5_on'+codem+'/',
            model_index=None, 
            overwrite=True,
            save_dataset=True)
    #nf5 = NeuralForecast.load(path='./checkpoints/test_TimesNet_day_hour5_on'+codem+'/')

    Y_hat_df = nf5.predict(futr_df=test_df)
    #Y_hat_df["ds"] = test_df["ds"]
    print(Y_hat_df)


    # 計算結果を格納
    results = []
    # 'unique_id'ごとに計算
    for unique_id in test_df['unique_id'].unique():
        y_true = test_df[test_df['unique_id'] == unique_id]['y']
        y_pred = Y_hat_df[Y_hat_df.index == unique_id]["CustomFEDformer"]
        
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
    ax.plot(predicted["ds"], predicted["CustomFEDformer"], label='Predicted')
    # タイトルとラベルの設定
    ax.set_title(f"ID: {unique_id}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_ylim(bottom=0) 
    #ax.legend()
    #plt.tight_layout()
    #plt.show()

    # モデルのパラメータ
    rnn_params = {
        'input_size': 40 * horizon, 
        'h': horizon, 
        'futr_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3','hour4'],
        #'hist_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'hour4'],
        #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
        'scaler_type':'robust',
        'max_steps': 1000, 
        'learning_rate':0.001,
        'batch_size' : 8,
        'windows_batch_size':512,
        'inference_windows_batch_size':512,
        'hidden_size' : 256,
    }
    
    # モデルのインスタンス
    rnn_model = CustomFEDformer(**rnn_params)
    nf4 = NeuralForecast(models=[rnn_model], freq='D')
    # モデルの学習
    nf4.fit(df=train_df)
    nf4.save(path='./checkpoints/test_FEDformer_day_hour4_on'+codem+'/',
            model_index=None, 
            overwrite=True,
            save_dataset=True)

    # モデルのパラメータ
    rnn_params = {
        'input_size': 40 * horizon, 
        'h': horizon, 
        'futr_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3'],
        #'hist_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'hour3'],
        #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
        'scaler_type':'robust',
        'max_steps': 1000, 
        'learning_rate':0.001,
        'batch_size' : 8,
        'windows_batch_size':512,
        'inference_windows_batch_size':512,
        'hidden_size' : 256,
    }
    # モデルのインスタンス
    rnn_model = CustomFEDformer(**rnn_params)
    nf3 = NeuralForecast(models=[rnn_model], freq='D')
    # モデルの学習
    nf3.fit(df=train_df)
    nf3.save(path='./checkpoints/test_FEDformer_day_hour3_on'+codem+'/',
            model_index=None, 
            overwrite=True,
            save_dataset=True)

    # モデルのパラメータ
    rnn_params = {
        'input_size': 40 * horizon, 
        'h': horizon, 
        'futr_exog_list':["RSI",'MACD',"MA",'Open','hour1','hour2'],
        #'hist_exog_list':['Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'hour2'],
        #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
        'scaler_type':'robust',
        'max_steps': 1000, 
        'learning_rate':0.001,
        'batch_size' : 8,
        'windows_batch_size':512,
        'inference_windows_batch_size':512,
        'hidden_size' : 256,
    }
    # モデルのインスタンス
    rnn_model = CustomFEDformer(**rnn_params)
    nf2 = NeuralForecast(models=[rnn_model], freq='D')
    # モデルの学習
    nf2.fit(df=train_df)
    nf2.save(path='./checkpoints/test_FEDformer_day_hour2_on'+codem+'/',
            model_index=None, 
            overwrite=True,
            save_dataset=True)

    # モデルのパラメータ
    rnn_params = {
        'input_size': 40 * horizon, 
        'h': horizon, 
        'futr_exog_list':["RSI",'MACD',"MA",'Open','hour1'],
        #'hist_exog_list':['Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'hour1'],
        #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
        'scaler_type':'robust',
        'max_steps': 1000, 
        'learning_rate':0.001,
        'batch_size' : 8,
        'windows_batch_size':512,
        'inference_windows_batch_size':512,
        'hidden_size' : 256,
    }
    # モデルのインスタンス
    rnn_model = CustomFEDformer(**rnn_params)
    nf1 = NeuralForecast(models=[rnn_model], freq='D')
    # モデルの学習
    nf1.fit(df=train_df)
    nf1.save(path='./checkpoints/test_FEDformer_day_hour1_on'+codem+'/',
            model_index=None, 
            overwrite=True,
            save_dataset=True)

    # モデルのパラメータ
    rnn_params = {
        'input_size': 40 * horizon, 
        'h': horizon, 
        'futr_exog_list':["RSI",'MACD',"MA",'Open'],
        #'hist_exog_list':['Open','hour1','hour2','hour3','hour4','hour5'],
        #'hist_exog_list':["RSI",'MACD',"MA",'Open'],
        #'stat_exog_list':['Type_1', 'Type_2', 'Type_3'],
        'scaler_type':'robust',
        'max_steps': 1000, 
        'learning_rate':0.001,
        'batch_size' : 8,
        'windows_batch_size':512,
        'inference_windows_batch_size':512,
        'hidden_size' : 256,
    }
    # モデルのインスタンス
    rnn_model = CustomFEDformer(**rnn_params)
    nf_open = NeuralForecast(models=[rnn_model], freq='D')
    # モデルの学習
    nf_open.fit(df=train_df)
    nf_open.save(path='./checkpoints/test_FEDformer_day_open_1_on'+codem+'/',
            model_index=None, 
            overwrite=True,
            save_dataset=True)

    #モデルロード#
    #nf_1 = NeuralForecast.load(path='./checkpoints/test_runTFT_day/')
    # 予測の実施
    #Y_hat_df = nf.predict(futr_df=test_df)
    #Y_hat_df = nf.predict().reset_index()
    Y_hat_df = nf_open.predict(futr_df=test_df)
    #Y_hat_df["ds"] = test_df["ds"]
    print(Y_hat_df)


    # 計算結果を格納
    results = []
    # 'unique_id'ごとに計算
    for unique_id in test_df['unique_id'].unique():
        y_true = test_df[test_df['unique_id'] == unique_id]['y']
        y_pred = Y_hat_df[Y_hat_df.index == unique_id]["CustomFEDformer"]
        
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
    ax.plot(predicted["ds"], predicted["CustomFEDformer"], label='Predicted')
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
    for j in range(1, days_pred):
        print(j)
        #listdata.append({'Close': df.loc[len(df)-horizon*20+j*7+6,"y"]})

        train_df1 = df.iloc[:-j]
        test_df = df.iloc[-j:]   # 最後の28行
        test_df = test_df.iloc[:horizon]
        #train_df1.to_csv(path2+'output_pred_traindf'+codem+"close"+'.csv')
        #train_df2 = df.iloc[:-horizon*20+(j+1)*7+i+1]
        #train_df2 = train_df2.tail(7)
        Y_hat_df_sub = nf_open.predict(df=train_df1,futr_df=test_df)
        Y_hat_df_open = pd.DataFrame(Y_hat_df_sub)
        Y_hat_df_sub = nf1.predict(df=train_df1,futr_df=test_df)
        Y_hat_df_1 = pd.DataFrame(Y_hat_df_sub)
        Y_hat_df_sub = nf2.predict(df=train_df1,futr_df=test_df)
        Y_hat_df_2 = pd.DataFrame(Y_hat_df_sub)
        Y_hat_df_sub = nf3.predict(df=train_df1,futr_df=test_df)
        Y_hat_df_3 = pd.DataFrame(Y_hat_df_sub)
        Y_hat_df_sub = nf4.predict(df=train_df1,futr_df=test_df)
        Y_hat_df_4 = pd.DataFrame(Y_hat_df_sub)
        Y_hat_df_sub = nf5.predict(df=train_df1,futr_df=test_df)
        Y_hat_df_5 = pd.DataFrame(Y_hat_df_sub)
        #Y_hat_df.to_csv(path2+'output_pred2'+codem+str(j)+str(i)+'.csv')
        #listdata.append({'Close': train_df1.loc[len(train_df1)-1,"y"]})
        last_price_open = test_df.head(1)
        last_price_open = last_price_open.iloc[0]['Open']
        last_price_1 = test_df.head(1)
        last_price_1 = last_price_1.iloc[0]['hour1']
        last_price_2 = test_df.head(1)
        last_price_2 = last_price_2.iloc[0]['hour2']
        last_price_3 = test_df.head(1)
        last_price_3 = last_price_3.iloc[0]['hour3']
        last_price_4 = test_df.head(1)
        last_price_4 = last_price_4.iloc[0]['hour4']
        last_price_5 = test_df.head(1)
        last_price_5 = last_price_5.iloc[0]['hour5']

        #print(last_price)
        close_price = df.tail(j)
        close_price = close_price.head(1)
        close_price = close_price.iloc[0]['y']
        #print(close_price)
        
        

        #price_data = last_price["y"].values
        close_pricepre =  Y_hat_df_open.head(1)["CustomFEDformer"].values
        close_pricepre_open = close_pricepre[0]
        close_pricepre =  Y_hat_df_1.head(1)["CustomFEDformer"].values
        close_pricepre_1 = close_pricepre[0]
        close_pricepre =  Y_hat_df_2.head(1)["CustomFEDformer"].values
        close_pricepre_2 = close_pricepre[0]
        close_pricepre =  Y_hat_df_3.head(1)["CustomFEDformer"].values
        close_pricepre_3 = close_pricepre[0]
        close_pricepre =  Y_hat_df_4.head(1)["CustomFEDformer"].values
        close_pricepre_4 = close_pricepre[0]
        close_pricepre =  Y_hat_df_5.head(1)["CustomFEDformer"].values
        close_pricepre_5 = close_pricepre[0]
        #print(close_pricepre)

        if(last_price_1 < close_pricepre_1):
            if(last_price_2 < close_pricepre_2):
                if(last_price_3 < close_pricepre_3):
                    if(last_price_4 < close_pricepre_4):
                        if(last_price_5 < close_pricepre_5):
                            sumdata = sumdata + close_price-last_price_1
                            listdata.append({'No':j,
                                'Open': last_price_open ,
                                'hour1': last_price_1 ,
                                'hour2': last_price_2 ,
                                'hour3': last_price_3 ,
                                'hour4': last_price_4 ,
                                'hour5': last_price_5 ,
                                'Close': close_price, 
                                'preCloseopen': close_pricepre_open,
                                'preClose1': close_pricepre_1,
                                'preClose2': close_pricepre_2,
                                'preClose3': close_pricepre_3,
                                'preClose4': close_pricepre_4,
                                'preClose5': close_pricepre_5,
                                'preprice': close_price-last_price_1
                            })
                        else:
                            sumdata = sumdata + last_price_5-last_price_1
                            listdata.append({'No':j,
                                'Open': last_price_open ,
                                'hour1': last_price_1 ,
                                'hour2': last_price_2 ,
                                'hour3': last_price_3 ,
                                'hour4': last_price_4 ,
                                'hour5': last_price_5 ,
                                'Close': close_price, 
                                'preCloseopen': close_pricepre_open,
                                'preClose1': close_pricepre_1,
                                'preClose2': close_pricepre_2,
                                'preClose3': close_pricepre_3,
                                'preClose4': close_pricepre_4,
                                'preClose5': close_pricepre_5,
                                'preprice': last_price_5-last_price_1
                            })
                    else:
                        sumdata = sumdata + last_price_4-last_price_1
                        listdata.append({'No':j,
                            'Open': last_price_open ,
                            'hour1': last_price_1 ,
                            'hour2': last_price_2 ,
                            'hour3': last_price_3 ,
                            'hour4': last_price_4 ,
                            'hour5': last_price_5 ,
                            'Close': close_price, 
                            'preCloseopen': close_pricepre_open,
                            'preClose1': close_pricepre_1,
                            'preClose2': close_pricepre_2,
                            'preClose3': close_pricepre_3,
                            'preClose4': close_pricepre_4,
                            'preClose5': close_pricepre_5,
                            'preprice': last_price_4-last_price_1
                        })
                else:
                    sumdata = sumdata + last_price_3-last_price_1
                    listdata.append({'No':j,
                        'Open': last_price_open ,
                        'hour1': last_price_1 ,
                        'hour2': last_price_2 ,
                        'hour3': last_price_3 ,
                        'hour4': last_price_4 ,
                        'hour5': last_price_5 ,
                        'Close': close_price, 
                        'preCloseopen': close_pricepre_open,
                        'preClose1': close_pricepre_1,
                        'preClose2': close_pricepre_2,
                        'preClose3': close_pricepre_3,
                        'preClose4': close_pricepre_4,
                        'preClose5': close_pricepre_5,
                        'preprice': last_price_3-last_price_1
                    })
            else:
                sumdata = sumdata + last_price_2-last_price_1
                listdata.append({'No':j,
                    'Open': last_price_open ,
                    'hour1': last_price_1 ,
                    'hour2': last_price_2 ,
                    'hour3': last_price_3 ,
                    'hour4': last_price_4 ,
                    'hour5': last_price_5 ,
                    'Close': close_price, 
                    'preCloseopen': close_pricepre_open,
                    'preClose1': close_pricepre_1,
                    'preClose2': close_pricepre_2,
                    'preClose3': close_pricepre_3,
                    'preClose4': close_pricepre_4,
                    'preClose5': close_pricepre_5,
                    'preprice': last_price_2-last_price_open
                })
        else:
            if(last_price_2 > close_pricepre_2):
                if(last_price_3 > close_pricepre_3):
                    if(last_price_4 > close_pricepre_4):
                        if(last_price_5 > close_pricepre_5):
                            sumdata = sumdata - close_price+last_price_1
                            listdata.append({'No':j,
                                'Open': last_price_open ,
                                'hour1': last_price_1 ,
                                'hour2': last_price_2 ,
                                'hour3': last_price_3 ,
                                'hour4': last_price_4 ,
                                'hour5': last_price_5 ,
                                'Close': close_price, 
                                'preCloseopen': close_pricepre_open,
                                'preClose1': close_pricepre_1,
                                'preClose2': close_pricepre_2,
                                'preClose3': close_pricepre_3,
                                'preClose4': close_pricepre_4,
                                'preClose5': close_pricepre_5,
                                'preprice': - close_price+last_price_1
                            })
                        else:
                            sumdata = sumdata - last_price_5+last_price_1
                            listdata.append({'No':j,
                                'Open': last_price_open ,
                                'hour1': last_price_1 ,
                                'hour2': last_price_2 ,
                                'hour3': last_price_3 ,
                                'hour4': last_price_4 ,
                                'hour5': last_price_5 ,
                                'Close': close_price, 
                                'preCloseopen': close_pricepre_open,
                                'preClose1': close_pricepre_1,
                                'preClose2': close_pricepre_2,
                                'preClose3': close_pricepre_3,
                                'preClose4': close_pricepre_4,
                                'preClose5': close_pricepre_5,
                                'preprice': - last_price_5+last_price_1
                            })
                    else:
                        sumdata = sumdata - last_price_4+last_price_1
                        listdata.append({'No':j,
                            'Open': last_price_open ,
                            'hour1': last_price_1 ,
                            'hour2': last_price_2 ,
                            'hour3': last_price_3 ,
                            'hour4': last_price_4 ,
                            'hour5': last_price_5 ,
                            'Close': close_price, 
                            'preCloseopen': close_pricepre_open,
                            'preClose1': close_pricepre_1,
                            'preClose2': close_pricepre_2,
                            'preClose3': close_pricepre_3,
                            'preClose4': close_pricepre_4,
                            'preClose5': close_pricepre_5,
                            'preprice': - last_price_4+last_price_1
                        })
                else:
                    sumdata = sumdata - last_price_3+last_price_1
                    listdata.append({'No':j,
                        'Open': last_price_open ,
                        'hour1': last_price_1 ,
                        'hour2': last_price_2 ,
                        'hour3': last_price_3 ,
                        'hour4': last_price_4 ,
                        'hour5': last_price_5 ,
                        'Close': close_price, 
                        'preCloseopen': close_pricepre_open,
                        'preClose1': close_pricepre_1,
                        'preClose2': close_pricepre_2,
                        'preClose3': close_pricepre_3,
                        'preClose4': close_pricepre_4,
                        'preClose5': close_pricepre_5,
                        'preprice': - last_price_3+last_price_1
                    })
            else:
                sumdata = sumdata - last_price_2+last_price_1
                listdata.append({'No':j,
                    'Open': last_price_open ,
                    'hour1': last_price_1 ,
                    'hour2': last_price_2 ,
                    'hour3': last_price_3 ,
                    'hour4': last_price_4 ,
                    'hour5': last_price_5 ,
                    'Close': close_price, 
                    'preCloseopen': close_pricepre_open,
                    'preClose1': close_pricepre_1,
                    'preClose2': close_pricepre_2,
                    'preClose3': close_pricepre_3,
                    'preClose4': close_pricepre_4,
                    'preClose5': close_pricepre_5,
                    'preprice': - last_price_2+last_price_open
                })

            
    df_close = pd.DataFrame(listdata)
    df_close.to_csv(path2+'output_pred2'+codem+"close"+'.csv')
    #train_df1.to_csv(path2+'output_pred_traindf'+codem+"close"+'.csv')
    df.to_csv(path2+'output_df1'+codem+"close"+'.csv')



    print(sumdata)
    print("happy")