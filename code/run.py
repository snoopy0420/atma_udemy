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
#from model_nn import ModelNN
from runner import Runner
from util import Submission, Util
import pandas as pd
import numpy as np


# # tensorflowの警告抑制
# import os
# os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# warnings.filterwarnings("ignore")
# warnings.simplefilter('ignore')


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
        'train_file_name': "train.pkl",          # 学習に使用するtrainファイル名
        'test_file_name': 'test.pkl',            # 予測に使用するtestファイル名
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
        "cat_cols": "all"        # target encodingするカラムをリストで指定
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
'Beds',
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
'UserRatingCleanliness',
'UserRatingCheckin',
'UserRatingCommunication',
'UserRatingLocation',
'UserRatingPrice',
'NumberOfReviews',
'NumberOfReviews1m',
'NumberOfReviews1y',
'FirstReview',
'LastReview',
'ReviewsPerMonth',
'NaN_num',
'OwnerSince_year',
'OwnerSince_month',
'FirstReview_year',
'FirstReview_month',
'LastReview_year',
'LastReview_month',
'ReplyRate_100%',
'AcceptanceRate_100%',
'Veri_None',
'Veri_',
'Veri_weibo',
'Veri_offline_government_id',
'Veri_photographer',
'Veri_sesame',
'Veri_jumio',
'Veri_facebook',
'Veri_sent_id',
'Veri_work_email',
'Veri_selfie',
'Veri_google',
'Veri_email',
'Veri_reviews',
'Veri_manual_offline',
'Veri_zhima_selfie',
'Veri_identity_manual',
'Veri_phone',
'Veri_manual_online',
'Veri_kba',
'Veri_government_id',
'Veri_sesame_offline',
'Verifications_num',
'Amenities_num',
'Cluster',
'Distance_0',
'Distance_1',
'Distance_2',
'Distance_3',
'Distance_4',
'Distance_5',
'Distance_6',
'Distance_7',
'Distance_8',
'Distance_9',
'diff_FirstReview_LastReview',
'Bathrooms_num',
'Bathrooms_type',
'OwnerDetail_lang',
'Description_lang',
'bert_svd_OwnerDetail_0',
'bert_svd_OwnerDetail_1',
'bert_svd_OwnerDetail_2',
'bert_svd_OwnerDetail_3',
'bert_svd_OwnerDetail_4',
'bert_svd_OwnerDetail_5',
'bert_svd_OwnerDetail_6',
'bert_svd_OwnerDetail_7',
'bert_svd_OwnerDetail_8',
'bert_svd_OwnerDetail_9',
'bert_svd_OwnerDetail_10',
'bert_svd_OwnerDetail_11',
'bert_svd_OwnerDetail_12',
'bert_svd_OwnerDetail_13',
'bert_svd_OwnerDetail_14',
'bert_svd_OwnerDetail_15',
'bert_svd_OwnerDetail_16',
'bert_svd_OwnerDetail_17',
'bert_svd_OwnerDetail_18',
'bert_svd_OwnerDetail_19',
'bert_svd_Description_0',
'bert_svd_Description_1',
'bert_svd_Description_2',
'bert_svd_Description_3',
'bert_svd_Description_4',
'bert_svd_Description_5',
'bert_svd_Description_6',
'bert_svd_Description_7',
'bert_svd_Description_8',
'bert_svd_Description_9',
'bert_svd_Description_10',
'bert_svd_Description_11',
'bert_svd_Description_12',
'bert_svd_Description_13',
'bert_svd_Description_14',
'bert_svd_Description_15',
'bert_svd_Description_16',
'bert_svd_Description_17',
'bert_svd_Description_18',
'bert_svd_Description_19',
'bert_svd_conbine_PT_Am_BT_0',
'bert_svd_conbine_PT_Am_BT_1',
'bert_svd_conbine_PT_Am_BT_2',
'bert_svd_conbine_PT_Am_BT_3',
'bert_svd_conbine_PT_Am_BT_4',
'bert_svd_conbine_PT_Am_BT_5',
'bert_svd_conbine_PT_Am_BT_6',
'bert_svd_conbine_PT_Am_BT_7',
'bert_svd_conbine_PT_Am_BT_8',
'bert_svd_conbine_PT_Am_BT_9',
'bert_svd_conbine_PT_Am_BT_10',
'bert_svd_conbine_PT_Am_BT_11',
'bert_svd_conbine_PT_Am_BT_12',
'bert_svd_conbine_PT_Am_BT_13',
'bert_svd_conbine_PT_Am_BT_14',
'bert_svd_conbine_PT_Am_BT_15',
'bert_svd_conbine_PT_Am_BT_16',
'bert_svd_conbine_PT_Am_BT_17',
'bert_svd_conbine_PT_Am_BT_18',
'bert_svd_conbine_PT_Am_BT_19',
'bert_svd_Review_0',
'bert_svd_Review_1',
'bert_svd_Review_2',
'bert_svd_Review_3',
'bert_svd_Review_4',
'bert_svd_Review_5',
'bert_svd_Review_6',
'bert_svd_Review_7',
'bert_svd_Review_8',
'bert_svd_Review_9',
'bert_svd_Review_10',
'bert_svd_Review_11',
'bert_svd_Review_12',
'bert_svd_Review_13',
'bert_svd_Review_14',
'bert_svd_Review_15',
'bert_svd_Review_16',
'bert_svd_Review_17',
'bert_svd_Review_18',
'bert_svd_Review_19',
'Review_count',

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
    # run_setting["hopt"] = "lgb_hopt" # パラメータチューニングを行う


    params = {
        "boosting_type": "gbdt",
        "objective": "fair", # regression
        "metric": "None",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "colsample_bytree": 0.5, # feature_fraction
        "reg_lambda": 5,
        "random_state": 71,
        "num_boost_round": 5000,
        "verbose_eval": False,
        "early_stopping_rounds": 100,
        'max_depth': 3,
        "min_data_in_leaf": 10,
        "num_leaves": 31,
    }

    # runnerクラスをインスタンス化
    runner = Runner(run_name, ModelLGB, lgb_features, params, file_setting, cv_setting, run_setting)

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
    ModelLGB.plot_learning_curve(dir_name, run_name, eval_metiric="mape") 

    # 予測
    runner.run_predict_cv()  

    # submissionファイルの作成
    lgb_preds = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl') # テストデータに対する予測値の読み込み
    lgb_preds = np.expm1(lgb_preds) # 対数変換を戻す
    Submission.create_submission(run_name, dir_name, lgb_preds)  # submit作成



######## xgboost ###############################################
 
    # # 特徴量
    # xgb_features = features

    # # CV設定の読み込み
    # cv_setting = get_cv_info()

    # # run name設定の読み込み
    # run_name = get_run_name(cv_setting, model_type="xgb")
    # dir_name = MODEL_DIR_NAME + run_name + '/'

    # # ログを吐くディレクトリを作成する
    # # exist_check(MODEL_DIR_NAME, run_name)  # これは使わない、使うと実行が終わらない
    # my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # # ファイルの設定を読み込む
    # file_setting = get_file_info()

    # # 学習の設定を読み込む
    # run_setting = get_run_info()
    # run_setting["hopt"] = "xgb_hopt"
    # # run_setting["calc_shap"] = True

    # # xgbパラメータを設定する
    # params = {
    #     'booster': 'gbtree',
    #     'objective': 'reg:squarederror',
    #     "eval_metric": "mae",
    #     'eta': 0.3,
    #     'gamma': 0.0,
    #     'alpha': 0.0,
    #     'lambda': 1.0,
    #     'min_child_weight': 1,
    #     'max_depth': 5,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'random_state': 71,
    #     'num_round': 1000,
    #     "verbose": False,
    #     'early_stopping_rounds': 100,
    # }

    # # インスタンス生成
    # runner = Runner(run_name, ModelXGB, xgb_features, params, file_setting, cv_setting, run_setting)

    # # 今回の学習で使用した特徴量名を取得
    # use_feature_name = runner.get_feature_name() 

    # # 今回の学習で使用したパラメータを取得
    # use_params = runner.get_params()

    # # モデルのconfigをjsonで保存
    # key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
    # value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
    # save_model_config(key_list, value_list, dir_name, run_name)
    
    # # 学習
    # if cv_setting.get('method') == 'None':
    #     runner.run_train_all()  # 全データで学習
    #     runner.run_predict_all()  # 予測
    # else:
    #     runner.run_train_cv()  # 学習
    #     ModelXGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
    #     ModelXGB.plot_learning_curve(run_name)  # learning curveを描画
    #     runner.run_predict_cv()  # 予測

    # # submissionファイルの作成
    # xgb_preds = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
    # Submission.create_submission(run_name, dir_name, xgb_preds)  # submit作成

    



# ##### ニューラルネットワーク ###########################################################

#     nn_features = features

#     # CV設定の読み込み
#     cv_setting = get_cv_info(random_state=53)

#     # run nameの設定
#     run_name = get_run_name(cv_setting, model_type="nn")
#     dir_name = MODEL_DIR_NAME + run_name + '/'

#     my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

#     # ファイルの設定を読み込む
#     file_setting = get_file_info()
    
#     # 学習の設定を読み込む
#     run_setting = get_run_info()
#     run_setting["hopt"] = False

#     # モデルのパラメータ
#     params = {
#         "num_classes": 1, 
#         'input_dropout': 0.0,
#         'hidden_layers': 3,
#         'hidden_units': 96,
#         'hidden_activation': 'relu',
#         'hidden_dropout': 0.2,
#         'batch_norm': 'before_act',
#         "output_activation": "sigmoid",
#         'optimizer': {'type': 'adam', 'lr': 0.000005},
#         "loss": "binary_crossentropy", 
#         "metrics": "accuracy", # カスタム評価関数も使える
#         'batch_size': 64,
#     }
#     # params = {
#     #     "num_classes": 1,
#     #     "input_dropout": 0.1,
#     #     "hidden_layers": 4.0,
#     #     "hidden_units": 160.0,
#     #     "hidden_activation": "relu",
#     #     "hidden_dropout": 0.30000000000000004,
#     #     "batch_norm": "no",
#     #     "output_activation": "sigmoid",
#     #     "optimizer": {
#     #         "lr": 0.009838711682220185,
#     #         "type": "sgd"
#     #     },
#     #     "loss": "binary_crossentropy",
#     #     "metrics": "accuracy",
#     #     "batch_size": 64.0
#     # }

#     runner = Runner(run_name, ModelNN, nn_features, params, file_setting, cv_setting, run_setting)

#     # 今回の学習で使用した特徴量名を取得
#     use_feature_name = runner.get_feature_name() 

#     # 今回の学習で使用したパラメータを取得
#     use_params = runner.get_params()

#     # モデルのconfigをjsonで保存
#     key_list = ['load_features', 'use_features', 'model_params', 'file_setting', 'cv_setting', "run_setting"]
#     value_list = [features, use_feature_name, use_params, file_setting, cv_setting, run_setting]
#     save_model_config(key_list, value_list, dir_name, run_name)
    
#     # 学習
#     if cv_setting.get('method') == 'None':
#         runner.run_train_all()  # 全データで学習
#         runner.run_predict_all()  # 予測
#     else:
#         runner.run_train_cv()  # 学習
#         ModelNN.plot_learning_curve(run_name)  # 学習曲線を描画
#         runner.run_predict_cv()  # 予測

#     # submissionファイルの作成
#     # 今回は,出力が確率なので,閾値の最適化後にラベル変換
#     train_labels = pd.read_pickle(FEATURE_DIR_NAME + f'{file_setting.get("train_file_name")}')[run_setting.get("target")]
#     nn_train_probs = Util.load_df_pickle(dir_name + f'{run_name}-train_preds.pkl')
#     nn_probs = Util.load_df_pickle(dir_name + f'{run_name}-pred.pkl')
#     nn_preds = get_label(train_labels, nn_train_probs, nn_probs)

#     Submission.create_submission(run_name, dir_name, nn_preds)  # submit作成



# ##### アンサンブル ####################################################################

#     run_name = get_run_name(cv_setting, "ensemble")

#     # アンサンブル
#     em_train_probs = xgb_train_probs*0.35 + lgb_train_probs*0.35 + nn_train_probs*0.3
#     em_probs = xgb_probs*0.35 + lgb_probs*0.35 + nn_probs*0.3
#     em_preds = get_label(train_labels, em_train_probs, em_probs)

#     Submission.create_submission(run_name, dir_name, em_preds)  # submit作成
