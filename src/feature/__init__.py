import gc
import os
import hashlib
import pandas as pd
import CONST
from pathlib import Path
from abc import ABCMeta, abstractmethod
import feature._001_utils as utils
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatureBase:
    """

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """

        """
        self.trn_base = utils.load_trn_base()
        self.tst_base = utils.load_tst_base()

    @property
    @abstractmethod
    def fin(self):
        pass

    @abstractmethod
    def create_feature_impl(self, df):
        raise NotImplementedError

    @abstractmethod
    def post_process(self, trn, tst):
        raise NotImplementedError

    def get_feature_dir(self):
        trn_dir = os.path.join(CONST.TRNFEATDIR)
        tst_dir = os.path.join(CONST.TSTFEATDIR)

        if not os.path.exists(trn_dir): os.makedirs(trn_dir)
        if not os.path.exists(tst_dir): os.makedirs(tst_dir)

        return trn_dir, tst_dir

    def create_feature(self, devmode=False):
        trn_dir, tst_dir = self.get_feature_dir()

        trn_feature_files = list(Path(trn_dir).glob('{}_*.f'.format(self.__class__.__name__)))
        tst_feature_files = list(Path(tst_dir).glob('{}_*.f'.format(self.__class__.__name__)))

        if len(trn_feature_files) > 0 and len(tst_feature_files) > 0 and devmode is False:
            print("There are cache dir for feature [{}] (train_cache_dir=[{}], test_cache_dir=[{}])".format(
                self.__class__.__name__, trn_dir, tst_dir
            ))

            return trn_feature_files, tst_feature_files

        print("Start computing feature [{}] (train_cache_dir=[{}], test_cache_dir=[{}])".format(
            self.__class__.__name__, trn_dir, tst_dir
        ))

        in_data = [pd.read_csv(f) for f in self.fin] if isinstance(self.fin, list) else pd.read_csv(self.fin)
        feat = self.create_feature_impl(in_data)
        del in_data
        gc.collect()

        # trn, tstに分ける
        feat_T1 = pd.concat([
            feat[['Season', 'TeamID']].rename(columns={'TeamID': 'T1TeamID'}),
            feat[[c for c in feat.columns if c not in ['Season', 'TeamID']]].add_prefix('T1')
        ], axis=1)
        feat_T2 = pd.concat([
            feat[['Season', 'TeamID']].rename(columns={'TeamID': 'T2TeamID'}),
            feat[[c for c in feat.columns if c not in ['Season', 'TeamID']]].add_prefix('T2')
        ], axis=1)

        trn = self.trn_base.copy()
        trn = trn.merge(feat_T1, on=['Season', 'T1TeamID'], how='left')
        trn = trn.merge(feat_T2, on=['Season', 'T2TeamID'], how='left')

        tst = self.tst_base.copy()
        tst = tst.merge(feat_T1, on=['Season', 'T1TeamID'], how='left')
        tst = tst.merge(feat_T2, on=['Season', 'T2TeamID'], how='left')

        trn, tst = self.post_process(trn, tst)

        # 保存する特徴量のカラムのプレフィックスとしてクラス名を追加
        feature_cols = [c for c in trn.columns if c not in CONST.EX_COLS]
        # nullが無いことを確認
        assert trn[feature_cols].notnull().all().all()
        assert tst[feature_cols].notnull().all().all()

        rename_dict = dict(zip(feature_cols, [f'{self.__class__.__name__}_{c}' for c in feature_cols]))
        trn = trn.rename(columns=rename_dict)
        tst = tst.rename(columns=rename_dict)

        feature_cols = list(rename_dict.values())

        assert 2096 == trn.shape[0]
        assert 11390 == tst.shape[0]

        # Save ...
        if not devmode:
            utils.to_feather(trn[feature_cols], trn_dir)
            utils.to_feather(tst[feature_cols], tst_dir)
            trn_feature_files = list(Path(trn_dir).glob('{}_*.f'.format(self.__class__.__name__)))
            tst_feature_files = list(Path(tst_dir).glob('{}_*.f'.format(self.__class__.__name__)))

            return trn_feature_files, tst_feature_files
        else:
            return trn, tst
