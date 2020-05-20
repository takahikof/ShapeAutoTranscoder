# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import sys
import time
import random
import math
import copy
import threading
import pickle
import warnings
import os

sys.path.append( 'utils' );
import common_util as cu
from shaperep import *
import model_dnn

###########
minibatch_size = 8;              # DNNのミニバッチサイズ
n_epoch = 201;                   # DNNの学習反復回数
dropout_rate = 0.0;              # dropout確率 (0.0:dropoutしない dropout層をネットワークに追加する必要があるので注意)
###########



if( __name__ == "__main__" ) :

    argv = sys.argv
    if( len( argv ) != 10 ) :
        print( "Usage: python " + argv[ 0 ] + " <Di:ShapeRepPickleTrain> <Fi:ModelListTrain> <Fi:LabelTrain>" );
        print( " <Di:ShapeRepPickleTest> <Fi:ModelListTest> <Fi:LabelTest> <Pi:NumDimEmbed>" );
        print( " <Pi:DeviceID> <D:Checkpoint>" );
        print( " <Pi:DeviceID> : e.g., /gpu:0, /gpu:1, /cpu:0" );
        quit();

    ##### コマンドライン引数を読み込む #####
    in_shaperep_filename_train = argv[ 1 ];
    in_list_filename_train = argv[ 2 ];
    in_label_filename_train = argv[ 3 ];
    in_shaperep_filename_test = argv[ 4 ];
    in_list_filename_test = argv[ 5 ];
    in_label_filename_test = argv[ 6 ];
    n_dim_embed = int( argv[ 7 ] );
    device_id = argv[ 8 ];
    checkpoint = argv[ 9 ];

    np.set_printoptions( threshold = np.inf ); # print時に省略しない
    if not sys.warnoptions : # disable warnings
        warnings.simplefilter("ignore");
        os.environ["PYTHONWARNINGS"] = "ignore";

    ##### リストファイルとラベルファイルを読み込む #####
    print( "loading files..." );
    modellist_train = cu.readList( in_list_filename_train );
    modellist_test = cu.readList( in_list_filename_test );
    n_models_train = len( modellist_train );
    n_models_test = len( modellist_test );

    ctg2id = cu.generateCategory2ID( in_label_filename_train ); # カテゴリ名->カテゴリ番号への対応
    n_class = len( ctg2id );
    Label_train = cu.generateLabelVec( modellist_train, in_label_filename_train, ctg2id ); # ラベル番号を格納したベクトル
    Label_test = cu.generateLabelVec( modellist_test, in_label_filename_test, ctg2id );    # ラベル番号を格納したベクトル


    ##### 3Dモデルの形状表現のpickleファイルを読み込む #####
    with open( in_shaperep_filename_train, mode='rb' ) as f :
        sr_train = pickle.load( f );
    with open( in_shaperep_filename_test, mode='rb' ) as f :
        sr_test = pickle.load( f );

    print( "Number of training 3D models: " + str( sr_train.n_model ) );
    print( "Number of testing 3D models: " + str( sr_test.n_model ) );


    ##### DNN構築・学習 (学習中に，テスト用データを使って検索精度評価も行う) #####
    print( "constructing DNN..." );
    config = tf.ConfigProto( allow_soft_placement = True, log_device_placement = False );
    config.gpu_options.allow_growth = True;
    graph = tf.Graph();
    with graph.as_default() :
        sess = tf.Session( config = config );

        dnn = model_dnn.DNN( sess, device_id, minibatch_size,
                             n_class, n_models_train, n_dim_embed, dropout_rate, sr_train, checkpoint );

        dnn.train( sr_train, sr_test, Label_train, Label_test, n_epoch, ctg2id,
                   in_list_filename_train, in_label_filename_train,
                   in_list_filename_test, in_label_filename_test );



    quit();
