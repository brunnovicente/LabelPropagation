# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calcular_tempo(t):
    horas = int(t / 3600)
    tempo = t % 3600
    minutos = int(tempo / 60)
    segundos = int(tempo % 60)
    return str(horas) + ' Horas, '+str(minutos)+' minutos e '+str(segundos)+' segundos.'
        

tsne = TSNE(n_components=2)
sca = MinMaxScaler()

bases = ['covtype']#, 'epilepsia', 'pendigits', 'reuters']

for base in bases:
    print('BASE: ', base)
    dados = pd.read_csv('C:/Users/brunn/Google Drive/bases/classificacao/'+base+'.csv')
    
    resultado_acuracia_tran = pd.DataFrame()
    resultado_recall_tran = pd.DataFrame()
    resultado_precisao_tran = pd.DataFrame()
    resultado_fscore_tran = pd.DataFrame()
        
    resultado_acuracia_ind = pd.DataFrame()
    resultado_recall_ind = pd.DataFrame()
    resultado_precisao_ind = pd.DataFrame()
    resultado_fscore_ind = pd.DataFrame()
    
    X = sca.fit_transform(dados.drop(['y'], axis=1))
    y = dados['y'].values
    if X.shape[0] > 100000:
        X, a, y, ay = train_test_split(X, y, train_size=0.1, stratify=y)
    if X.shape[0] > 30000:
        X, a, y, ay = train_test_split(X, y, train_size=0.2, stratify=y)
    
    for i in np.arange(0.01,0.11, 0.01):
        print('--- Rotulados: ', int(i*100),'%')
        
        ACURACIA = []
        RECALL = []
        PRECISAO = []
        FSCORE = []
        
        ACURACIAi = []
        RECALLi = []
        PRECISAOi = []
        FSCOREi = []
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y)
        X_train = pd.DataFrame(X_train)
        
        for j in np.arange(10):
            label_prop_model = LabelPropagation()
            L, U, yl, yu = train_test_split(X_train, y_train, train_size=i, stratify=y_train)
            yc = y_train.copy()
            yc[U.index.values] = -1
            
            t1 = time.time()
            label_prop_model.fit(X_train, yc)
            t2 = time.time()
            print('... Teste ',j, ' - Tempo: ', calcular_tempo(t2 - t1))
            
            #Transductive Score
            y_pred_t = label_prop_model.predict(U)
            ACURACIA.append(accuracy_score(yu, y_pred_t))
            RECALL.append(recall_score(yu, y_pred_t, average='weighted'))
            PRECISAO.append(precision_score(yu, y_pred_t, average='weighted'))
            FSCORE.append(f1_score(yu, y_pred_t, average='weighted'))
            
            #Inductive Score
            y_pred_i = label_prop_model.predict(X_test)
            ACURACIAi.append(accuracy_score(y_test, y_pred_i))
            RECALLi.append(recall_score(y_test, y_pred_i, average='weighted'))
            PRECISAOi.append(precision_score(y_test, y_pred_i, average='weighted'))
            FSCOREi.append(f1_score(y_test, y_pred_i, average='weighted'))
       
        resultado_acuracia_tran[np.int(i*100)] = ACURACIA
        resultado_recall_tran[np.int(i*100)] = RECALL
        resultado_precisao_tran[np.int(i*100)] = PRECISAO
        resultado_fscore_tran[np.int(i*100)] = FSCORE
        
        resultado_acuracia_ind[np.int(i*100)] = ACURACIAi
        resultado_recall_ind[np.int(i*100)] = RECALLi
        resultado_precisao_ind[np.int(i*100)] = PRECISAOi
        resultado_fscore_ind[np.int(i*100)] = FSCOREi
    
        resultado_acuracia_tran.to_csv('lp_acuracia_tran_'+base+'.csv', index=False)
        resultado_recall_tran.to_csv('lp_recall_tran_'+base+'.csv', index=False)
        resultado_precisao_tran.to_csv('lp_precisa_tran_'+base+'.csv', index=False)
        resultado_fscore_tran.to_csv('lp_fscore_tran_'+base+'.csv', index=False)
        
        resultado_acuracia_ind.to_csv('lp_acuracia_ind_'+base+'.csv', index=False)
        resultado_recall_ind.to_csv('lp_recall_ind_'+base+'.csv', index=False)
        resultado_precisao_ind.to_csv('lp_precisa_ind_'+base+'.csv', index=False)
        resultado_fscore_ind.to_csv('lp_fscore_ind_'+base+'.csv', index=False)
    

