# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm
import yaml
from ta import add_all_ta_features
from scipy.stats import spearmanr

# test
class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2019-02-28"
    # 評価期間開始日
    VAL_START = "2019-03-01"
    # 評価期間終了日
    VAL_END = "2019-06-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None
    feature = None

    dtw = None
    cate = ["Result_FinancialStatement ReportType", "topix", "section"]
    results = {}

    @classmethod
    def get_inputs(cls, dataset_dir):
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"]
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END]
                _val_X = feats[cls.VAL_START : cls.VAL_END]
                _test_X = feats[cls.TEST_START :]

                _train_y = labels[: cls.TRAIN_END]
                _val_y = labels[cls.VAL_START : cls.VAL_END]
                _test_y = labels[cls.TEST_START :]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        def yoy(se: pd.Series) -> pd.Series: 
            #前年比計算
            df = pd.DataFrame()
            df["se"] = se
            df["diff"] = se.diff()
            df["abs"] = np.abs(se).shift(1)
            df["yoy"] = df["diff"]/df["abs"]
            df["yoy"] = df["yoy"].fillna(0)
            return df["yoy"]

        # 各種情報読み込み&個別の銘柄に絞る
        stocks_fin = dfs["stock_fin"]
        stock_fin = stocks_fin.loc[stocks_fin["Local Code"] == code]
        stock_fin.sort_values("datetime", inplace=True)

        stocks_list = dfs["stock_list"]
        stock_list = stocks_list.loc[stocks_list["Local Code"] == code]

        stocks_price = dfs["stock_price"]
        stock_price = stocks_price[stocks_price["Local Code"] == code]
        stock_price.sort_values("datetime", inplace=True)

        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90

        with open('fin_feat.yaml') as file:
            fin_feat = yaml.safe_load(file.read())
        stock_fin = stock_fin[fin_feat]
        stock_fin = stock_fin.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()

        # 欠損値処理
        stock_fin = stock_fin.fillna(0)

        price_feat = ["EndOfDayQuote Open", "EndOfDayQuote High", "EndOfDayQuote Low", "EndOfDayQuote ExchangeOfficialClose", "EndOfDayQuote Volume"]
        stock_price = stock_price[price_feat]
        stock_price = stock_price.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()

        stock_price = add_all_ta_features(stock_price, open="EndOfDayQuote Open", high="EndOfDayQuote High",
                                          low="EndOfDayQuote Low", close="EndOfDayQuote ExchangeOfficialClose", volume="EndOfDayQuote Volume")


        # # 欠損値処理
        stock_price = stock_price.fillna(0)
        
        issuedshare = stock_list["IssuedShareEquityQuote IssuedShare"].values[0]
        section = stock_list["Section/Products"].values[0]
        topix = stock_list["Size (New Index Series)"].values[0]

        stock_price["Market_capitalization"] = np.log(stock_price["EndOfDayQuote ExchangeOfficialClose"] * issuedshare)
        # 元データのカラムを削除
        stock_price = stock_price.drop(["EndOfDayQuote Open", "EndOfDayQuote High", "EndOfDayQuote Low", "EndOfDayQuote Volume"], axis=1)

        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        stock_price = stock_price.loc[stock_price.index.isin(stock_fin.index)]
        stock_fin = stock_fin.loc[stock_fin.index.isin(stock_price.index)]

        ######## ファンダメンタル分析

        # 成長性の特徴量
        stock_fin["Result_FinancialStatement NetAssets_yoy"] = yoy(stock_fin["Result_FinancialStatement NetAssets"])
        stock_fin["Result_FinancialStatement OperatingIncome_yoy"] = yoy(stock_fin["Result_FinancialStatement OperatingIncome"])
        stock_fin["Result_FinancialStatement OrdinaryIncome_yoy"] = yoy(stock_fin["Result_FinancialStatement OrdinaryIncome"])
        stock_fin["Result_FinancialStatement TotalAssets_yoy"] = yoy(stock_fin["Result_FinancialStatement TotalAssets"])
        stock_fin["eps_yoy"] = yoy(stock_fin["Result_FinancialStatement NetIncome"]/issuedshare)

        # 収益性の特徴量
        stock_fin["Result_FinancialStatement OperatingIncome rate"] = (stock_fin["Result_FinancialStatement OperatingIncome"]/stock_fin["Result_FinancialStatement NetSales"]).fillna(0)
        stock_fin["Result_FinancialStatement OrdinaryIncome rate"] = (stock_fin["Result_FinancialStatement OrdinaryIncome"]/stock_fin["Result_FinancialStatement NetSales"]).fillna(0)
    
        stock_fin["Result_FinancialStatement OperatingIncome rate_yoy"] = yoy(stock_fin["Result_FinancialStatement OperatingIncome rate"])
        stock_fin["Result_FinancialStatement OrdinaryIncome rate_yoy"] = yoy(stock_fin["Result_FinancialStatement OrdinaryIncome rate"])

        # 割安性の特徴量
        stock_fin["bps"] = (stock_fin["Result_FinancialStatement NetAssets"]/issuedshare).fillna(0)

        # 安定性の特徴量
        stock_fin["ROA"] = (stock_fin["Result_FinancialStatement NetIncome"]/stock_fin["Result_FinancialStatement TotalAssets"]).fillna(0)

        # 予想情報による達成率
        stock_fin["Forecast_FinancialStatement NetIncome"] = stock_fin["Forecast_FinancialStatement NetIncome"].shift(1)
        stock_fin["Result_FinancialStatement NetIncome_achieverate"] = stock_fin["Result_FinancialStatement NetIncome"]/stock_fin["Forecast_FinancialStatement NetIncome"]
        stock_fin["Result_FinancialStatement NetIncome_achieverate"] = stock_fin["Result_FinancialStatement NetIncome_achieverate"].replace([np.inf, -np.inf], 1).fillna(1)

        stock_fin["Forecast_FinancialStatement NetSales"] = stock_fin["Forecast_FinancialStatement NetSales"].shift(1)
        stock_fin["Result_FinancialStatement NetSales_achieverate"] = stock_fin["Result_FinancialStatement NetSales"]/stock_fin["Forecast_FinancialStatement NetSales"]
        stock_fin["Result_FinancialStatement NetSales_achieverate"] = stock_fin["Result_FinancialStatement NetSales_achieverate"].replace([np.inf, -np.inf], 1).fillna(1)

        stock_fin["Forecast_FinancialStatement OperatingIncome"] = stock_fin["Forecast_FinancialStatement OperatingIncome"].shift(1)
        stock_fin["Forecast_FinancialStatement OperatingIncome_achieverate"] = stock_fin["Result_FinancialStatement OperatingIncome"]/stock_fin["Forecast_FinancialStatement OperatingIncome"]
        stock_fin["Forecast_FinancialStatement OperatingIncome_achieverate"] = stock_fin["Forecast_FinancialStatement OperatingIncome_achieverate"].replace([np.inf, -np.inf], 1).fillna(1)

        stock_fin["Forecast_FinancialStatement OrdinaryIncome"] = stock_fin["Forecast_FinancialStatement OrdinaryIncome"].shift(1)
        stock_fin["Forecast_FinancialStatement OrdinaryIncome_achieverate"] = stock_fin["Result_FinancialStatement OrdinaryIncome"]/stock_fin["Forecast_FinancialStatement OrdinaryIncome"]
        stock_fin["Forecast_FinancialStatement OrdinaryIncome_achieverate"] = stock_fin["Forecast_FinancialStatement OrdinaryIncome_achieverate"].replace([np.inf, -np.inf], 1).fillna(1)

        stock_fin = stock_fin.fillna(0)

        # データを結合
        feats = pd.concat([stock_price, stock_fin], axis=1)

        feats["pbr"] = (feats["EndOfDayQuote ExchangeOfficialClose"]/feats["bps"]).fillna(0)
        feats["per"] = (feats["EndOfDayQuote ExchangeOfficialClose"]/(stock_fin["Result_FinancialStatement NetIncome"]/issuedshare)).fillna(0)
        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], 0)
        feats = feats.fillna(0)

        # 銘柄コードを設定
        feats["code"] = code
        feats["section"] = section
        feats["topix"] = topix

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats

    @classmethod
    def create_model(cls, dfs, inputs, codes, label):
        if cls.feature is None:
            # 特徴量を取得
            buff = []
            for code in tqdm(codes):
                buff.append(cls.get_features_for_predict(cls.dfs, code))
            feature = pd.concat(buff)
            cls.feature = feature
        else:
            feature = cls.feature

        with open('drop_feat.yaml') as file:
            drop_feat = yaml.safe_load(file.read())

        feature = feature.drop(drop_feat, axis=1)

        feature[cls.cate] = feature[cls.cate].astype("category")

        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, test_X, test_y = cls.get_features_and_label(
            dfs, codes, feature, label
        )
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train) 

        # モデル作成
        # LightGBM parameters
        with open('lgb_params.yaml') as file:
            params = yaml.safe_load(file.read())

        def spearman(preds, data):
            correlation, pvalue = spearmanr(preds, data.get_label())
            return "spearman", correlation, True

        result = {}
        # モデルの学習
        model = lgb.train(params,
                        train_set=lgb_train, # トレーニングデータの指定
                        valid_sets=[lgb_eval,lgb_train], # 検証データの指定
                        valid_names=['eval', 'train'],
                        verbose_eval=True,
                        categorical_feature=cls.cate,
                        evals_result=result,
                        num_boost_round=5000,
                        feval=spearman
                        )
        cls.results[label] = result

        return model

    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        try:
            for label in labels:
                m = os.path.join(model_path, f"my_model_{label}.pkl")
                with open(m, "rb") as f:
                    # pickle形式で保存されているモデルを読み込み
                    cls.models[label] = pickle.load(f)
            return True
        except Exception as e:
            print(e)
            return False

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path="../model"):
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            print(label)
            model = cls.create_model(cls.dfs, inputs=inputs, codes=codes, label=label)
            cls.save_model(model, label, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feature = pd.concat(buff)
        
        with open('drop_feat.yaml') as file:
            drop_feat = yaml.safe_load(file.read())

        feature = feature.drop(drop_feat, axis=1)

        feature[cls.cate] = feature[cls.cate].astype("category")

        # 日付と銘柄コードに絞り込み
        df = feature.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(
            str
        )

        # 出力対象列を定義
        output_columns = ["code"]

        # 目的変数毎に予測
        for label in labels:
            # 予測実施
            df[label] = cls.models[label].predict(feature)
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()