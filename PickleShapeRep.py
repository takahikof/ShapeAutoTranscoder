# -*- coding: utf-8 -*-

import os
import sys
import pickle

sys.path.append( 'utils' );
import common_util as cu
from renderer import *
from pointsampler import *
from voxelizer import *
from off import *
from shaperep import *


ALIGN_UPRIGHT = False; # SH17LSの3Dモデルのuprightベクトルの向きを，ModelNet10/40のuprightベクトルの向きに揃えるかどうか


render_eye_filename = "data/icosahedron.off"; # 視点3Dモデル
render_eye_position = "V";                   # 視点の位置 : "V":頂点, "F":面の重心
render_projection = "O";                     # "O":平行投影, "P":透視投影
render_resolution = 64;                      # レンダリング解像度 (レンダリング画像の1辺の画素数)
point_num = 1024;                            # 1モデル当たりの点数
voxel_resolution = 32;                       # ボクセル解像度 (3D画像の1辺の画素数)


if( __name__ == "__main__" ) :

    argv = sys.argv
    if( len( argv ) != 3 ) :
        print( "Usage: python " + argv[ 0 ] + " <Fi:OffPickle> <Fo:ShapeRepPickle>" );
        quit();

    ##### コマンドライン引数を読み込む #####
    in_off_filename = argv[ 1 ];
    out_shaperep_filename = argv[ 2 ];

    ##### ファイルを読み込む #####
    print( "loading files..." );
    with open( in_off_filename, mode='rb' ) as f :
        off = pickle.load( f );
    print( "Number of 3D models: " + str( len( off ) ) );

    dirname = os.path.dirname( out_shaperep_filename );
    if( dirname != "" ) :
        cu.makeDir( os.path.dirname( out_shaperep_filename ) );

    ##### 多視点レンダラを作成 #####
    eyes = Mesh( render_eye_filename, radius=1.1 );
    renderer = Renderer( eyes, render_eye_position, render_resolution, render_projection );
    print( "Number of views for rendering: " + str( renderer.n_views ) );
    print( "Rendering resolution: " + str( renderer.resolution ) );

    ##### 点群サンプラを作成 #####
    pointsampler = PointSampler( point_num );
    print( "Number of points per 3D model: " + str( pointsampler.n_point ) );

    ##### ボクセライザを作成 #####
    voxelizer = Voxelizer( voxel_resolution, "B" );
    print( "Voxel resolution: " + str( voxelizer.resolution ) );
    print( "Voxel type: " + voxelizer.mode );

    ##### 複数の形状表現に変換 #####
    sr = ShapeRep( off, renderer, pointsampler, voxelizer, ALIGN_UPRIGHT );

    ##### 形状表現のpickleを書き出す #####
    with open( out_shaperep_filename, 'wb' ) as f :
        pickle.dump( sr, f );

    quit();
