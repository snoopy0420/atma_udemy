import os
import sys
import yaml
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 定数の読み込み
CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
FIGURE_DIR_NAME = yml['SETTING']['DIR_FIGURE']
DIR_HOME = yml['SETTING']['DIR_HOME']
DIR_MODEL = yml['SETTING']['DIR_MODEL']
DIR_FIGURE = yml['SETTING']['DIR_FIGURE']

# 自作モジュールの読み込み
sys.path.append(DIR_HOME)
from src.model import Model
from src.util import Util, Metric


class model_LSTM_mult(Model):

    def __init__(self, 
                 run_fold_name: str,
                 params: dict,
                 logger,
                 ) -> None:
        """コンストラクタ
        run_fold_name: runの名前とfoldの番号を組み合わせた名前
        params: ハイパーパラメータ
        """
        super().__init__(run_fold_name, params, logger)
        # カラム
        self.key_cols = params.get("key_cols") # list
        self.target_col = params.get("target_col") # str
        # ハイパーパラメータ
        self.seq_length = params.get('seq_length')
        self.n_steps = params.get('n_steps')
        self.batch_size = params.get('batch_size', 64)
        self.shuffle = params.get('shuffle', True)
        self.hidden_size = params.get('hidden_size', 64)
        self.num_epochs = params.get('num_epochs', 20)
        self.learning_rate = params.get('learning_rate', 0.001)
        # オブジェクト
        self.model = None
        self.scaler = None
        self.scaler_for_inverse = None
        self.base_dir = os.path.join(DIR_MODEL, self.run_fold_name)

    
    def create_sequences_for_forecast(self, data, key_cols, feat_cols, seq_length, n_steps, is_test=False):
        """
        予測対象の23時間分のデータが揃っている場合のみデータセットを作成
        return:
            list_x: 特徴量
            list_y: 予測対象
            list_key:  key_cols
        """
        list_x, list_y = [], []
        list_key = []
        # station_idごとにデータを作成
        for station_id in data['station_id'].unique():
            station_data = data[data['station_id'] == station_id]
            station_data = station_data.set_index('datetime')
            if not is_test:
                # 日付ごとにデータを取得
                for day in station_data.index.normalize().unique():
                    day = pd.to_datetime(day)
                    day_data = station_data.loc[:day+pd.Timedelta(hours=0)]  # dayの0時までのデータ
                    if len(day_data) >= seq_length:  # 過去データがシーケンス長よりも多い場合
                        x = day_data.iloc[-seq_length:]  # シーケンス長分のデータを取得
                        if x[self.target_col].isnull().sum() > 0: # ターゲットカラムに欠損値がある場合はスキップ
                            continue
                        x = x[feat_cols].values
                        # 正解データ(dayの1~23時のデータ)の作成
                        next_day_data = station_data.loc[day+pd.Timedelta(hours=1): day+pd.Timedelta(hours=23)] # 予測対象の23時間分のデータ
                        if len(next_day_data) == n_steps:  # 予測対象の23時間分が揃っている場合
                            y = next_day_data[self.target_col].values
                            # ターゲットカラムに欠損値がある場合はスキップ
                            if np.isnan(y).sum() > 0:
                                continue
                            list_x.append(x)
                            list_y.append(y)
                            # list_key.append(next_day_data.reset_index()[key_cols])
            else:
                # 最新断面のlist_xのみを作成
                day_data = station_data
                if len(day_data) >= seq_length:
                    x = day_data[feat_cols].values[-seq_length:]
                    list_x.append(x)
                    next_days = pd.date_range(day_data.index.max()+pd.Timedelta(hours=1), day_data.index.max()+pd.Timedelta(hours=23), freq='h')
                    df_ = pd.DataFrame({"datetime": next_days, "station_id": station_id})
                    list_key.append(df_)
                    list_key = pd.concat(list_key, axis=0)
        print(len(list_x), len(list_y))
        return np.array(list_x), np.array(list_y), list_key
    

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])  # 最後のタイムステップの出力
            return out


    def train(self, tr: pd.DataFrame) -> None:
        """
        学習に使用できる期間のデータを受け取り、モデルの学習を行い、学習済のモデルを保存する
        tr: 学習データ[key_cols, target_col, 特徴量]
        """
        # 並び替え
        tr = tr.sort_values(self.key_cols)

        # key_colsの削除
        # tr = tr.drop(self.key_cols, axis=1)

        # 標準化
        tr_scale = tr.copy()
        feat_cols = [col for col in tr.columns.to_list() if col not in self.key_cols]
        scale_cols = feat_cols + [self.target_col]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_for_inverse = MinMaxScaler(feature_range=(0, 1))
        tr_scale[scale_cols] = self.scaler.fit_transform(tr[scale_cols])
        _ = self.scaler_for_inverse.fit_transform(tr[[self.target_col]])

        # シーケンスの作成
        tr_X, tr_y, tr_key = self.create_sequences_for_forecast(tr_scale, self.key_cols, feat_cols, self.seq_length, self.n_steps)
        print(tr_X.shape, tr_y.shape)

        # デバイスを確認 (GPUがあればGPU、なければCPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # tensorに変換
        # NumPy配列を明示的にfloat32に変換
        # tr_X = tr_X.astype(np.float32)
        # tr_y = tr_y.astype(np.float32)
        # tr_X_tensor = torch.tensor(tr_X, dtype=torch.float32)
        # tr_y_tensor = torch.tensor(tr_y, dtype=torch.float32)
        # # バッチサイズを設定
        batch_size = 100
        # バッチごとにテンソルに変換
        tr_X_tensors = []
        tr_y_tensors = []
        for i in range(0, len(tr_X), batch_size):
            tr_X_tensors.append(torch.tensor(tr_X[i:i+batch_size], dtype=torch.float32).to(device))  # GPUに移動
            tr_y_tensors.append(torch.tensor(tr_y[i:i+batch_size], dtype=torch.float32).to(device))  # GPUに移動
        tr_X_tensor = torch.cat(tr_X_tensors, dim=0)
        tr_y_tensor = torch.cat(tr_y_tensors, dim=0)
        print(tr_X_tensor.shape, tr_y_tensor.shape)

        # データローダーを作成
        gc.collect()
        tr_dataset = TensorDataset(tr_X_tensor, tr_y_tensor)
        print(len(tr_dataset))
        tr_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        print(len(tr_loader))

        # モデルの初期化
        self.input_size = tr_X_tensor.shape[2]  # 入力の次元
        print(self.input_size)
        self.model = self.LSTMModel(self.input_size, self.hidden_size, self.n_steps).to(device)
        print("model")
        criterion = nn.L1Loss().to(device)
        print("criterion")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("start training")
        # トレーニングループ
        train_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, (inputs, targets) in enumerate(tr_loader):
            
                # バッチごとにデータをGPUに移動（すでに移動してある場合は不要）
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        # 学習曲線の保存
        self.plot_learning_curve(train_losses)


    def predict(self, va):
        """
        vaの最新断面から23期先までの予測を行う

        Args:
            va: 予測データ[key_cols, target_col, 特徴量]
        return: 
            df: columns[key_cols, target_col]
        """
        # vaの最新日付を抽出
        target_date = va["datetime"].max().date()

        # 並び替え
        va = va.sort_values(self.key_cols)

        # key_colsの削除
        # key = va[self.key_cols]
        # va = va.drop(self.key_cols, axis=1)

        # 標準化
        va_scale = va.copy()
        feat_cols = [col for col in va.columns.to_list() if col not in self.key_cols]
        scale_cols = feat_cols + [self.target_col]
        va_scale[scale_cols] = self.scaler.transform(va[scale_cols])

        # シーケンスの作成
        va_X, va_y, va_key = self.create_sequences_for_forecast(va_scale, self.key_cols, feat_cols, self.seq_length, self.n_steps, is_test=True)

        # デバイスを確認 (GPUがあればGPU、なければCPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # テンソルに変換
        va_X_tensor = torch.tensor(va_X, dtype=torch.float32).to(device)

        # 予測
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            va_pred = self.model(va_X_tensor).cpu().numpy()

        # 予測結果を逆変換
        va_pred = self.scaler_for_inverse.inverse_transform(va_pred)

        # 予測結果を整形[key_cols, pred]
        va_key[self.target_col] = va_pred.reshape(-1, 1)

        return va_key
    
    def save_model(self) -> None:
        """
        モデルを保存する
        """
        os.makedirs(self.base_dir, exist_ok=True)
        path_model = os.path.join(self.base_dir, 'model.pkl')
        path_scaler = os.path.join(self.base_dir, 'scaler.pkl')
        path_scaler_for_inverse = os.path.join(self.base_dir, 'scaler_for_inverse.pkl')
        Util.dump(self.model, path_model)
        # torch.save(self.model.state_dict(), path_model)  # モデルのパラメータを保存
        Util.dump(self.scaler, path_scaler)
        Util.dump(self.scaler_for_inverse, path_scaler_for_inverse)

    def load_model(self) -> None:
        """
        モデルを読み込む
        """
        path_model = os.path.join(self.base_dir, 'model.pkl')
        path_scaler = os.path.join(self.base_dir, 'scaler.pkl')
        path_scaler_for_inverse = os.path.join(self.base_dir, 'scaler_for_inverse.pkl')
        # self.model = self.LSTMModel(self.input_size, self.hidden_size, self.n_steps)
        # self.model.load_state_dict(torch.load(path_model))
        self.model = Util.load(path_model)
        self.model.eval()  # 評価モードに切り替え
        self.scaler = Util.load(path_scaler)
        self.scaler_for_inverse = Util.load(path_scaler_for_inverse)


    def plot_learning_curve(self, train_losses):
        """
        学習曲線をプロット
        """
        # 学習曲線のプロット
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM Training Loss Curve")
        plt.legend()
        save_path = os.path.join(self.base_dir, 'learning_curve.png')
        plt.savefig(save_path)
        plt.close()

