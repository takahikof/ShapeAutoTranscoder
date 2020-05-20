# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import random
import math

def makeDir( dirname ) :
  if not os.path.exists( dirname ) :
    os.makedirs( dirname );


def randomSamplingRow( mat, n ) :
    n_rows = mat.shape[ 0 ];
    perm = np.arange( n_rows );
    np.random.shuffle( perm );
    if( n > n_rows ) :
        n = n_rows;
    return mat[ perm[ 0:n ], : ];

def readList( list_filename ) :
    file = open( list_filename );
    list = file.readlines();
    file.close();
    for i in range( len( list ) ) :
        list[ i ] = list[ i ].rstrip("\n"); # 改行コード削除
    return list;

def readMap( map_filename ) :
    map = {};
    map_file = open( map_filename, 'r' );
    for line in map_file :
        itemlist = line[:-1].split(' '); # 半角スペースで区切られているとする
        map[ itemlist[ 0 ] ] = itemlist[ 1 ];
    map_file.close();
    return map;

def generateCategory2ID( label_filename ) :
    # labelファイルから，カテゴリ名 -> カテゴリ番号への対応表を作成
    model2ctg = readMap( label_filename );

    # カテゴリ名を番号に置換
    ctg2id = {};
    i = 0;
    for value in model2ctg.values() :
        if( value not in ctg2id ) :
            ctg2id[ value ] = i;
            i += 1;

    return ctg2id;

def generateLabelMat( modellist, label_filename, ctg2id ) :
    # labelファイルから，カテゴリ名 -> カテゴリ番号への対応表を作成
    model2ctg = readMap( label_filename );

    # 教師行列 (各行がone-hot vector)を作成
    labelmat = np.zeros( ( len( modellist ), len( ctg2id ) ), dtype = np.float32 );
    i = 0;
    for model in modellist :
        if( model in model2ctg ) :
            id = ctg2id[ model2ctg[ model ] ];
            labelmat[ i, id ] = 1.0;
            i += 1;
        else :
            print( model + " doesn't exist in the label file : " + label_filename );

    return labelmat;

def generateLabelVec( modellist, label_filename, ctg2id ) :
    # labelファイルから，カテゴリ名 -> カテゴリ番号への対応表を作成
    model2ctg = readMap( label_filename );

    # ラベルベクトル(各要素が各サンプルのカテゴリ番号)を作成
    labelvec = np.zeros( len( modellist ), dtype = np.int32 );

    i = 0;
    for model in modellist :
        if( model in model2ctg ) :
            id = ctg2id[ model2ctg[ model ] ];
            labelvec[ i ] = id;
            i += 1;
        else :
            print( model + " doesn't exist in the label file : " + label_filename );

    return labelvec;

def generatePairIndex( labelvec ) :
    # labelvec[i]にはデータiのカテゴリ番号が格納されている

    pos_idx = [];
    neg_idx = [];

    # pos_idx[ i ][ 0 ] にはデータiの番号 (=iと同じ値)を格納
    # pos_idx[ i ][ 1 ] にはデータiと正例ペアになり得るデータ番号を格納

    for i in range( len( labelvec ) ) :
        pos_idx.append( [ i ] );
        neg_idx.append( [ i ] );

    # 物体カテゴリレベルでのマッチを正例ペアとする
    for i in range( len( labelvec ) ) :
        tmppos = [];
        tmpneg = [];
        for j in range( len( labelvec ) ) :
            if( i != j ) : # 自分自身とはペアを組まない
                if( labelvec[ i ] == labelvec[ j ] ) :
                    tmppos.append( j );
                else :
                    tmpneg.append( j );
        pos_idx[ i ].append( np.array( tmppos ).astype( np.int32 ) );
        neg_idx[ i ].append( np.array( tmpneg ).astype( np.int32 ) );

    return( pos_idx, neg_idx );

def hasInfNan( m ) :
    n = np.isnan( m );
    i = np.isinf( m );
    result = np.sum( n+i );
    if( result > 0 ) :
        return True;
    else :
        return False;
