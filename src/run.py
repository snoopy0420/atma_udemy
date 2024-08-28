import sys
import os
import shutil
import datetime
import yaml
import json
import collections as cl
import warnings
from model_lgb import ModelLGB
from model_xgb import ModelXGB
from model_nn import ModelNN
from runner import Runner
from util import Submission, Util
import pandas as pd
import numpy as np


# tensorflowの警告抑制
import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')


# configの読み込み
CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']
TARGET = yml['SETTING']['TARGET']


def exist_check(path, run_name):
    """学習ファイルの存在チェックと実行確認
    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)

    # 通常の実行確認
    print('特徴量ディレクトリ:{} で実行しますか？[Y/n]'.format(FEATURE_DIR_NAME))
    x = input('>> ')
    if x != 'Y':
        print('終了します')
        sys.exit(0)

def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def my_makedirs_remove(path):
    """引数のpathディレクトリを新規作成する（存在している場合は削除→新規作成）
    path:ディレクトリ名
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_model_config(key_list, value_list, dir_name, run_name):
    """実装詳細を管理する。どんな「特徴量/パラメータ/cv/setting/」で学習させたモデルかを管理するjsonファイルを出力する
    params→ key_list:キーのリスト、value_list:キーに対するバリューのリスト
    """
    def set_default(obj):
        """json出力の際にset型のオブジェクトをリストに変更する
        """
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    
    conf_dict = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        conf_dict[v] = data

    json_file = open(dir_name + run_name  + '_param.json', 'w', encoding="utf-8")
    json.dump(conf_dict, json_file, indent=4, default=set_default, ensure_ascii=False)


    
########### 以下で諸々の設定をする############################################

def get_cv_info(random_state=42) -> dict:
    """CVの設定
    """
    # methodはThresholdクラスの関数を文字列で指定、
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定(それよりもn_split=1にするのがいいかも)
    # StratifiedKFold or GroupKFold or StratifiedGroupKFold の場合はcv_target_gr, cv_target_sfに対象カラム名を設定する
    cv_setting = {
        'method': 'KFold',
        'n_splits': 5,
        'random_state': 123,
        'shuffle': True,
        'cv_target': None
    }
    return cv_setting


def get_run_name(cv_setting, model_type):
    """run名の作成
    """
    run_name = model_type
    suffix = '_' + datetime.datetime.now().strftime("%m%d%H%M")
    model_info = ''
    run_name = run_name + '_' + cv_setting.get('method') + model_info + suffix
    return run_name


def get_file_info():
    """ファイル名の設定
    """
    file_setting = {
        'feature_dir_name': FEATURE_DIR_NAME,   # 特徴量の読み込み先ディレクトリ
        'model_dir_name': MODEL_DIR_NAME,       # モデルの保存先ディレクトリ
        'train_file_name': "vs_train.pkl",          # 学習に使用するtrainファイル名
        'test_file_name': 'vs_test.pkl',            # 予測に使用するtestファイル名
    }
    return file_setting


def get_run_info():
    """学習の設定
    """
    run_setting = {
        'target': TARGET,       # 目的変数
        'calc_shap': False,     # shap値を計算するか否か
        'save_train_pred': False,    # trainデータに対する予測値を保存するか否か(閾値の最適化に使用)
        "hopt": False,           # パラメータチューニング、lgb_hopt,xgb_hopt,nn_hopt,False
        "target_enc": False,     # target encoding をするか否か
        "target_enc_col": ["OwnerID"]        # target encodingするカラムをリストで指定
    }
    return run_setting




if __name__ == '__main__':

    
######## 学習・推論 ################################################
    # 使用する特徴量の指定
    # features_list.txtからコピペ、適宜取捨選択
    features = [
'ID',
'OwnerID',
'OwnerSince',
'TimeToReply',
'IdentityVerified',
'ListingsCount',
'HasPicture',
'RoomType',
'MaximumAccommodates',
'InstantBookable',
'Bedrooms',
'AreaCategory',
'Latitude',
'Longitude',
'Availability',
'Vacancy30',
'Vacancy60',
'Vacancy90',
'Vacancy365',
'UserRatingOverall',
'UserRatingInformation',

    ]


######### LightGBM #############################################################

    lgb_features = features

    # CV設定の読み込み
    cv_setting = get_cv_info(random_state=86)
    # run nameの設定
    run_name = get_run_name(cv_setting, model_type="lgb")
    dir_name = MODEL_DIR_NAME + run_name + '/'
    # runディレクトリの作成。ここにlogなどが吐かれる
    my_makedirs(dir_name)  
    # ファイルの設定を読み込む
    file_setting = get_file_info()
    # 学習の設定を読み込む
    run_setting = get_run_info()
    run_setting["target_enc"] = True
    run_setting["cat_cols"] = ["OwnerID", "City", "Beds_per_Bedroom", "Bedrooms", "RoomType", "Country"] # "AreaCategory"
    # run_setting["hopt"] = "lgb_hopt" # パラメータチューニングを行う 


    params = {
        "boosting_type": "gbdt",
        "objective": "fair",
        "metric": "None",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "colsample_bytree": 0.5, # feature_fraction
        "subsample": 0.5,
        "reg_lambda": 5,
        "random_state": 71,
        "num_boost_round": 5000,
        "verbose_eval": False,
        "early_stopping_rounds": 100,
        'max_depth': 3,
        "min_data_in_leaf": 10,
        "num_leaves": 31,
    }
    run_setting = {
        "key": ["ID"],       # データセットのID
        'target': TARGET_COL,       # 目的変数
        'calc_shap': False,     # shap値を計算するか否か
        'save_train_pred': False,    # trainデータに対する予測値を保存するか否か(閾値の最適化に使用)
        "hopt": False,           # パラメータチューニング、lgb_hopt,xgb_hopt,nn_hopt,False
        "target_enc": False,     # target encoding をするか否か
        "target_enc_col": ["OwnerID"]        # target encodingするカラムをリストで指定
    }

    # runnerクラスをインスタンス化
    runner = Runner(run_name, ModelLGB, df_train, df_test, params, run_setting, logger)

    # 今回の学習で使用する特徴量名を取得
    use_feature_name = runner.get_feature_name() 
    # 今回の学習で使用するパラメータを取得
    use_params = runner.get_params()
    # モデルのconfigをjsonで保存
    key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    save_model_config(key_list, value_list, dir_name, run_name)
    
    # runnerの学習
    runner.run_train_cv()  

    # feature_importanceを計算・描画
    ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name)  
     # learning curveを描画
    ModelLGB.plot_learning_curve(dir_name, run_name, eval_metric="mape") 

    # 予測
    runner.run_predict_cv()  

    # submissionファイルの作成
    lgb_preds = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl') # テストデータに対する予測値の読み込み
    lgb_preds = np.expm1(lgb_preds) # 対数変換を戻す
    Submission.create_submission(run_name, dir_name, lgb_preds)  # submit作成



######## xgboost ###############################################
 
    # 特徴量
    xgb_features = features

    # CV設定の読み込み
    cv_setting = get_cv_info(random_state=86)
    # run name設定の読み込み
    run_name = get_run_name(cv_setting, model_type="xgb")
    dir_name = MODEL_DIR_NAME + run_name + '/'
    # runディレクトリの作成。
    my_makedirs(dir_name)  
    # ファイルの設定を読み込む
    file_setting = get_file_info()
    # 学習の設定を読み込む
    run_setting = get_run_info()
    run_setting["cat_cols"] = ["OwnerID", "City", "Beds_per_Bedroom", "Bedrooms", "RoomType", "Country"]
    # run_setting["hopt"] = "xgb_hopt"

    # xgbパラメータを設定する
    params = {
        'booster': 'gbtree',
        'objective': 'reg:pseudohubererror',
        "eval_metric": "mape",
        'eta': 0.01,
        'gamma': 0.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'min_child_weight': 1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 71,
        "verbose": True,
        'num_round': 5000,
        'early_stopping_rounds': 100,
    }

    # インスタンス生成
    runner = Runner(run_name, ModelXGB, xgb_features, params, file_setting, cv_setting, run_setting)

    # 今回の学習で使用した特徴量名を取得
    use_feature_name = runner.get_feature_name() 
    # 今回の学習で使用したパラメータを取得
    use_params = runner.get_params()
    # モデルのconfigをjsonで保存
    key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    save_model_config(key_list, value_list, dir_name, run_name)
    
    # 学習
    runner.run_train_cv()

    ModelXGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
    ModelXGB.plot_learning_curve(dir_name, run_name, eval_metric="mape")  # learning curveを描画

    # 予測
    runner.run_predict_cv()  # 予測

    # submissionファイルの作成
    xgb_preds = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
    xgb_preds = np.expm1(xgb_preds) # 対数変換を戻す
    Submission.create_submission(run_name, dir_name, xgb_preds)  # submit作成

    



##### ニューラルネットワーク ###########################################################

    nn_features = features

    # CV設定の読み込み
    cv_setting = get_cv_info(random_state=53)
    # run nameの設定
    run_name = get_run_name(cv_setting, model_type="nn")
    dir_name = MODEL_DIR_NAME + run_name + '/'
    # runディレクトリの作成。ここにlogなどが吐かれる
    my_makedirs(dir_name)  
    # ファイルの設定を読み込む
    file_setting = get_file_info()
    file_setting["train_file_name"] = "nn_train.pkl"
    file_setting["test_file_name"] = "nn_test.pkl"
    # 学習の設定を読み込む
    run_setting = get_run_info()
    run_setting["cat_cols"] = ["OwnerID", "City", "Beds_per_Bedroom", "Bedrooms", "RoomType", "Country"]
    # run_setting["hopt"] = "nn_hopt"

    # モデルのパラメータ
    params = {
        "num_classes": 1, 
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.1,
        'batch_norm': 'before_act',
        "output_activation": None,
        'optimizer': {'type': 'adam', 'lr': 0.1},
        "loss": 'mean_absolute_percentage_error',
        "metrics": "mean_absolute_percentage_error", # カスタム評価関数も使える
        'batch_size': 64,
    }
    # params = {
    #     "num_classes": 1,
    #     "input_dropout": 0.1,
    #     "hidden_layers": 4.0,
    #     "hidden_units": 160.0,
    #     "hidden_activation": "relu",
    #     "hidden_dropout": 0.30000000000000004,
    #     "batch_norm": "no",
    #     "output_activation": "sigmoid",
    #     "optimizer": {
    #         "lr": 0.009838711682220185,
    #         "type": "sgd"
    #     },
    #     "loss": "binary_crossentropy",
    #     "metrics": "accuracy",
    #     "batch_size": 64.0
    # }

    # インスタンス生成
    runner = Runner(run_name, ModelNN, nn_features, params, file_setting, cv_setting, run_setting)

    # 今回の学習で使用した特徴量名を取得
    use_feature_name = runner.get_feature_name() 
    # 今回の学習で使用したパラメータを取得
    use_params = runner.get_params()
    # モデルのconfigをjsonで保存
    key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    save_model_config(key_list, value_list, dir_name, run_name)
    
    # 学習
    runner.run_train_cv()

    # 学習曲線を描画
    ModelNN.plot_learning_curve(dir_name, run_name)  

    # 予測
    runner.run_predict_cv()  

    # submissionファイルの作成
    nn_preds = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
    nn_preds = np.expm1(nn_preds)
    Submission.create_submission(run_name, dir_name, nn_preds)  # submit作成



##### アンサンブル ####################################################################

    run_name = get_run_name(cv_setting, "ensemble")

    # アンサンブル
    # em_train_probs = xgb_train_probs*0.35 + lgb_train_probs*0.35 + nn_train_probs*0.3
    # em_probs = xgb_probs*0.35 + lgb_probs*0.35 + nn_probs*0.3
    # em_preds = get_label(train_labels, em_train_probs, em_probs)
    em_preds = xgb_preds*0.35 + lgb_preds*0.35 + nn_preds*0.3

    Submission.create_submission(run_name, dir_name, em_preds)  # submit作成
