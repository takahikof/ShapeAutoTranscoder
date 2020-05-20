# -*- coding: utf-8 -*-
import numpy as np
from renderer import *
from pointsampler import *
from voxelizer import *
from off import *

class ShapeRep :
    def __init__( self, off, renderer, pointsampler, voxelizer, align_upright ) :

        self.n_model = len( off );

        self.mvimg = [];
        self.oript = [];
        self.voxel = [];

        self.render_n_view = renderer.n_views;
        self.render_resolution = renderer.resolution;
        self.point_n_point = pointsampler.n_point;
        self.voxel_resolution = voxelizer.resolution;
        self.voxel_type = voxelizer.mode;

        for i in range( self.n_model ) :
            print( str( i ) + " / " + str( self.n_model ) );

            if( align_upright == True ) :
                off[ i ].rotate_x_90(); # x軸周りに90度回転

            self.mvimg.append( renderer.multiview_rendering( off[ i ] ) );
            self.oript.append( pointsampler.sample( off[ i ] ) );
            self.voxel.append( voxelizer.sample( off[ i ] ) );
                        


    def append( self, sp ) : # リストに連結する

        self.n_model += sp.n_model;

        # エラーチェック
        if( self.render_n_view != sp.render_n_view ) :
            print( "error: render_n_view must be equal." );
            quit();
        if( self.render_resolution != sp.render_resolution ) :
            print( "error: render_resolution must be equal." );
            quit();
        if( self.point_n_point != sp.point_n_point ) :
            print( "error: point_n_point must be equal." );
            quit();
        if( self.voxel_resolution != sp.voxel_resolution ) :
            print( "error: voxel_resolution must be equal." );
            quit();
        if( self.voxel_type != sp.voxel_type ) :
            print( "error: voxel_type must be equal." );
            quit();

        for i in range( sp.n_model ) :
            self.mvimg.append( sp.mvimg[ i ] );
            self.oript.append( sp.oript[ i ] );
            self.voxel.append( sp.voxel[ i ] );
