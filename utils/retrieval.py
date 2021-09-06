# -*- coding: utf-8 -*-
import numpy as np
import time
import random
import math
import sys
import os

from sklearn.metrics import pairwise_distances
from sklearn.metrics import label_ranking_average_precision_score
sys.path.append( 'utils' );
import common_util as cu


def parse_label_file( list_file, label_file ) :
    modellist = cu.readList( list_file );
    ctg2id = cu.generateCategory2ID( label_file );
    label = cu.generateLabelVec( modellist, label_file, ctg2id );
    n_class = len( ctg2id );
    return label, n_class;

def retrieval( feat, list_file, label_file, metric ) :

    label, n_class = parse_label_file( list_file, label_file );

    n_model = feat.shape[ 0 ];

    if( metric == "L2" ) :
        metric = "euclidean";
        D = pairwise_distances( feat, metric=metric );
    elif( metric == "L1" ) :
        metric = "cityblock";
        D = pairwise_distances( feat, metric=metric );
    elif( metric == "COS" ) :
        metric = "cosine";
        D = pairwise_distances( feat, metric=metric );
    else :
        print( "error: invalid metric: " + metric );

    S = -1.0 * D; # convert to similarity
    J = create_groundtruth_matrix( label );

    # 検索0件目を除外するために対角成分を除去
    S = remove_diag_elements( S );
    J = remove_diag_elements( J );

    # NN
    idx = np.argmax( S, axis=1 );
    idx = idx + np.arange( 0, n_model * ( n_model - 1 ), ( n_model - 1 ) );
    J_top1 = np.take( J, idx );
    nn = np.mean( J_top1 );

    # MAP
    map = label_ranking_average_precision_score( J, S );

    return( nn, map );


def remove_diag_elements( A ) :
    # 正方行列Aの対角成分を削除 (サイズ NxN が Nx(N-1)になる)
    # https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)


def create_groundtruth_matrix( label ) :
    # ラベルベクトルから正解行列を作成．
    # ラベルベクトルlabelのi番目の要素は，データiのカテゴリ番号．
    # 正解行列Mの(i,j)の要素は，データiとデータjが同じカテゴリである場合1, そうでなければ0．
    # https://stackoverflow.com/questions/48157476/efficiently-compute-pairwise-equal-for-numpy-arrays
    l = np.reshape( label, ( -1, 1 ) );
    a = np.ascontiguousarray( l );
    b = np.ascontiguousarray( l );
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]));
    x1D = a.view(void_dt).ravel();
    y1D = b.view(void_dt).ravel();
    M = (x1D[:,None] == y1D).astype(np.int32);
    return M;
