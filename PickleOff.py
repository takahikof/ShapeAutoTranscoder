# -*- coding: utf-8 -*-

import os
import sys
import pickle

sys.path.append( 'utils' );
import common_util as cu
from renderer import *
from off import *

if( __name__ == "__main__" ) :

    argv = sys.argv
    if( len( argv ) != 4 ) :
        print( "Usage: python " + argv[ 0 ] + " <Di:Off> <Fi:ModelList> <Fo:Pickle>" );
        quit();

    ##### コマンドライン引数を読み込む #####
    in_off_dirname = argv[ 1 ];
    in_list_filename = argv[ 2 ];
    out_pickle_filename = argv[ 3 ];

    ##### ファイルを読み込む #####
    print( "loading files..." );
    modellist = cu.readList( in_list_filename );
    n_models = len( modellist );

    dirname = os.path.dirname( out_pickle_filename );
    if( dirname != "" ) :
        cu.makeDir( os.path.dirname( out_pickle_filename ) );

    ##### OFFファイルを読み込む #####
    meshes = [];
    for i in range( n_models ) :
        print( str( i ) + " / " + str( n_models ) );
        mesh = Mesh( in_off_dirname + "/" + modellist[ i ] + ".off" );
        meshes.append( mesh );

    ##### OFFファイルのpickleを書き出す #####
    with open( out_pickle_filename, 'wb' ) as f :
        pickle.dump( meshes, f );

    quit();
