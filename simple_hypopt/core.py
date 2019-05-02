#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:25:49 2019

@author: takumi_uchida
"""

import random
import pandas as pd
import numpy as np

class hypopt:
    def __init__(self, f, M, HPs, T=100, random_seed=None):
        """
        This is a simple hyperparameter optimization class.
        This concept base on SMBO[*1].        
        The arguments and attributs of this class 
        corresponde to section2 of [*1].
        
        [*1]https://www.lri.fr/~kegl/research/PDFs/BeBaBeKe11.pdf
        
        ARGUMENTs
        ----------------
        f [function]:
            evaluation function. f(x) is loss of test set. 
            x is a hyperparameter vector. 
        M [object having fit() and predict()]:
            learn H by fit(), and predict loss of test set by predict(X).
            X is a hyperparameter vectors.
        HPs [array of dictionary]:
            all pattern of hyperparameters to seek.
        T [int]:
            number of iteration. default is 100.
        
        EXAMPLEs
        ----------------
        from sklearn.datasets import load_boston
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression

        dataset = load_boston()
        
        def f(**hyper_parameters):
            model = GradientBoostingRegressor(**hyper_parameters)
            scores = cross_val_score(model, 
                                     X=dataset.data, y=dataset.target, 
                                     cv=4, scoring='neg_mean_absolute_error')
            return scores.mean()
        
        M = LinearRegression()
                                                
        HPs = [{'learning_rate':0.10, 'n_estimators':1, 'max_features':'sqrt'}
              ,{'learning_rate':0.08, 'n_estimators':2, 'max_features':'log2'}
              ,{'learning_rate':0.06, 'n_estimators':3, 'max_features':'sqrt'}
              ,{'learning_rate':0.04, 'n_estimators':4, 'max_features':'log2'}
              ,{'learning_rate':0.02, 'n_estimators':5, 'max_features':'sqrt'}
              ,{'learning_rate':0.01, 'n_estimators':6, 'max_features':'log2'}
              ]
        
        self = hypopt(f, M, HPs)
        
        
        ATTRIBUTEs
        ----------------
        S [function]:
            suggest next hyperparameter x to evaluate by f(x).
            S use M to suggest x.

        H [dictionary]:
            History of hyperparameter vector and loss of test set.
            ex) {'HP_index':[3, 1],
                 'score': [2.3, 4.3]}            
        """
        random.seed(random_seed)

        self.f   = f
        self.M   = M
        self.HPs = HPs
        self.T   = T

        self.n_HPs = len(self.HPs)
        self._fit_HPs_vector()        
        
        k = min(3, self.n_HPs)
        indexes = random.sample(range(self.n_HPs), k=k)
        score = [f(**self.HPs[i]) for i in indexes]
        self.H = {'HP_index': indexes, 'score':score}
        
        self._M_fit()
    
    def fit(self):
        for t in range(self.T):
            print('t={}'.format(t))
            is_updated = self._update()
            if not is_updated:
                return None
        return None
    
    def get_best_HP(self):
        best_score_H_index  = self.H['score'].index(max(self.H['score']))
        best_score_HP_index = self.H['HP_index'][best_score_H_index]
        return self.HPs[best_score_HP_index]
        
    def S(self):
        """
        self.HPs_vector について、self.Mで予測しテストスコアを予測させる。
        予測結果をindex_prescoreで管理し、予測テストスコアが最大になるindexを選出する。
        """
        prescore = self.M.predict(self.HPs_vector)
        index_prescore = [(i, prescore) for i,prescore in enumerate(prescore)]        
        index_prescore = sorted(index_prescore, key=lambda x:x[1], reverse=True)
        for i in index_prescore:
            index = index_prescore[i][0]
            if index not in self.H['HP_index']:
                return index
            else:
                return None # break
        
    def _fit_HPs_vector(self):
        _df = pd.DataFrame(self.HPs)
        frame = []
        for col in _df.columns:
            if isinstance(_df[col][0], str):
                frame.append(pd.get_dummies(_df[col], prefix=col))
            else:
                frame.append(_df[[col]])
        self.HPs_vector = np.array(pd.concat(frame, axis=1))
    
    def _M_fit(self):
        self.M.fit(self.HPs_vector[self.H['HP_index']], self.H['score'])

    def _update(self):
        """
        self.Sで選出されたHPs_vectorのindexと、
        実際のテストスコア（=self.f()の実行結果）をself.Hに追加する。
        最後に更新されたself.Hに対してself.Mを再学習する。
        更新があった場合はTrueを返却し、更新がなかった場合はFalseを返却する。
        """
        next_HP_index = self.S()
        if next_HP_index is not None:
            self.H['HP_index'].append(next_HP_index)
            self.H['score'].append(self.f(**self.HPs[next_HP_index]))
            self._M_fit()
            return True
        return False


