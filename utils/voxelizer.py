# -*- coding: utf-8 -*-
import numpy as np
from off import *
from pointsampler import *

class Voxelizer : # ポリゴンスープ/点群をボクセル化する
    def __init__( self, resolution, mode ) :
        self.resolution = resolution;

        if( mode != "B" and mode != "D" ) : # "B": binary, "D": density
            print( "invalid voxelization mode: " + mode );
            quit();

        self.mode = mode;

    def sample( self, mesh ) : # ポリゴンスープをバイナリボクセル化
        vox = np.zeros( ( self.resolution, self.resolution, self.resolution ), dtype=np.float32 );

        # 頂点の座標値は正規化済み([-1,1]の範囲に収まっている)としてボクセル化

        # 頂点の座標値を[0,1]に変換
        vertices = mesh.vert.copy();
        vertices = ( vertices + 1.0 ) / 2.0;

        boxhalf = ( 1.0 / self.resolution ) / 2.0;
        boxhalfsize = [ boxhalf, boxhalf, boxhalf ];

        for t in range( mesh.n_face ) :
            triangle = vertices[ mesh.face[ t ] ];
            bbox_min = np.min( triangle, axis=0 );
            bbox_max = np.max( triangle, axis=0 );

            x_from = int( bbox_min[ 0 ] * self.resolution );
            y_from = int( bbox_min[ 1 ] * self.resolution );
            z_from = int( bbox_min[ 2 ] * self.resolution );
            x_to = int( bbox_max[ 0 ] * self.resolution );
            y_to = int( bbox_max[ 1 ] * self.resolution );
            z_to = int( bbox_max[ 2 ] * self.resolution );

            if( x_to == self.resolution ) :
                x_to = self.resolution - 1;
            if( y_to == self.resolution ) :
                y_to = self.resolution - 1;
            if( z_to == self.resolution ) :
                z_to = self.resolution - 1;

            for x in range( x_from, x_to + 1 ) :
                for y in range( y_from, y_to + 1 ) :
                    for z in range( z_from, z_to + 1 ) :

                        if( vox[ x, y, z ] == 1.0 ) : # 体積が既にある場合は判定しない
                            continue;

                        bcx = ( 1.0 / self.resolution ) * x + boxhalf;
                        bcy = ( 1.0 / self.resolution ) * y + boxhalf;
                        bcz = ( 1.0 / self.resolution ) * z + boxhalf;
                        boxcenter = [ bcx, bcy, bcz ];

                        if( check( boxcenter, boxhalfsize, triangle ) ) :
                            vox[ x, y, z ] = 1.0;

        # 座標系を修正
        vox = np.swapaxes( vox, 0, 2 );
        vox = np.flip( vox, 0 );
        vox = np.swapaxes( vox, 0, 1 );

        return vox;


    def sample_from_pointset( self, points ) : # 点群をボクセル化
        vox = np.zeros( ( self.resolution, self.resolution, self.resolution ), dtype=np.float32 );

        # 点群pointsの座標値は正規化済み([-1,1]の範囲に収まっている)としてボクセル化
        for i in range( points.shape[ 0 ] ) : # 各点について
            # 点iが属するボクセル座標値を計算
            v = ( points[ i ] + 1.0 ) / 2.0 * float( self.resolution );

            v = ( v + 0.5 ).astype( np.int32 );

            if( v[ 0 ] < 0 ) :
                v[ 0 ] = 0;
            if( v[ 1 ] < 0 ) :
                v[ 1 ] = 0;
            if( v[ 2 ] < 0 ) :
                v[ 2 ] = 0;
            if( v[ 0 ] >= self.resolution ) :
                v[ 0 ] = self.resolution - 1;
            if( v[ 1 ] >= self.resolution ) :
                v[ 1 ] = self.resolution - 1;
            if( v[ 2 ] >= self.resolution ) :
                v[ 2 ] = self.resolution - 1;

            vox[ v[ 0 ], v[ 1 ], v[ 2 ] ] += 1.0;

        if( self.mode == "B" ) :
            vox = np.where( vox > 0.0, 1.0, 0.0 );
        elif( self.mode == "D" ) :
            vox /= np.max( vox ); # 最大画素値が1となるように正規化

        return vox;

###################
# boxcenterとboxhalfsizeで指定された3D矩形領域と，三角形triangleが交差するかどうか判定
# ベタ移植： http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
def check( boxcenter, boxhalfsize, triangle ) :
    # move everything so that the boxcenter is in (0,0,0)
    v0 = triangle[0] - boxcenter;
    v1 = triangle[1] - boxcenter;
    v2 = triangle[2] - boxcenter;

    # compute triangle edges
    e0 = v1 - v0;
    e1 = v2 - v1;
    e2 = v0 - v2;

    # Bullet 3: test the 9 tests first (this was faster)

    fex = np.abs( e0[0] );
    fey = np.abs( e0[1] );
    fez = np.abs( e0[2] );

    # AXISTEST_X01(e0[Z], e0[Y], fez, fey);
    a = e0[2];
    b = e0[1];
    fa = fez;
    fb = fey;
    p0 = a*v0[1] - b*v0[2];
    p2 = a*v2[1] - b*v2[2];
    if( p0 < p2 ) :
        min=p0;
        max=p2;
    else :
        min=p2;
        max=p0;
    rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];
    if( min > rad or max < -rad ) :
        return False;

    # AXISTEST_Y02(e0[Z], e0[X], fez, fex);
    a = e0[2];
    b = e0[0];
    fa = fez;
    fb = fex;
    p0 = -a*v0[0] + b*v0[2];
    p2 = -a*v2[0] + b*v2[2];
    if( p0 < p2 ) :
        min=p0;
        max=p2;
    else :
        min=p2;
        max=p0;
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];
    if( min > rad or max < -rad ) :
        return False;

    # AXISTEST_Z12(e0[Y], e0[X], fey, fex);
    a = e0[1];
    b = e0[0];
    fa = fey;
    fb = fex;
    p1 = a*v1[0] - b*v1[1];
    p2 = a*v2[0] - b*v2[1];
    if( p2 < p1 ) :
        min=p2;
        max=p1;
    else :
        min=p1
        max=p2;
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];
    if( min > rad or max < -rad ) :
        return False;


    fex = np.abs( e1[0] );
    fey = np.abs( e1[1] );
    fez = np.abs( e1[2] );

    # AXISTEST_X01(e1[Z], e1[Y], fez, fey);
    a = e1[2];
    b = e1[1];
    fa = fez;
    fb = fey;
    p0 = a*v0[1] - b*v0[2];
    p2 = a*v2[1] - b*v2[2];
    if( p0 < p2 ) :
        min=p0;
        max=p2;
    else :
        min=p2;
        max=p0;
    rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];
    if( min > rad or max < -rad ) :
        return False;

    # AXISTEST_Y02(e1[Z], e1[X], fez, fex);
    a = e1[2];
    b = e1[0];
    fa = fez;
    fb = fex;
    p0 = -a*v0[0] + b*v0[2];
    p2 = -a*v2[0] + b*v2[2];
    if( p0 < p2 ) :
        min=p0;
        max=p2;
    else :
        min=p2;
        max=p0;
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];
    if( min > rad or max < -rad ) :
        return False;

    # AXISTEST_Z0(e1[Y], e1[X], fey, fex);
    a = e1[1];
    b = e1[0];
    fa = fey;
    fb = fex;
    p0 = a*v0[0] - b*v0[1];
    p1 = a*v1[0] - b*v1[1];
    if( p0 < p1 ) :
        min=p0;
        max=p1;
    else :
        min=p1;
        max=p0;
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];
    if( min > rad or max < -rad ) :
        return False;


    fex = np.abs( e2[0] );
    fey = np.abs( e2[1] );
    fez = np.abs( e2[2] );

    # AXISTEST_X2(e2[Z], e2[Y], fez, fey);
    a = e2[2];
    b = e2[1];
    fa = fez;
    fb = fey;
    p0 = a*v0[1] - b*v0[2];
    p1 = a*v1[1] - b*v1[2];
    if( p0 < p1 ) :
        min=p0;
        max=p1;
    else :
        min=p1;
        max=p0;
    rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];
    if( min > rad or max < -rad ) :
        return False;

    # AXISTEST_Y1(e2[Z], e2[X], fez, fex);
    a = e2[2];
    b = e2[0];
    fa = fez;
    fb = fex;
    p0 = -a*v0[0] + b*v0[2];
    p1 = -a*v1[0] + b*v1[2];
    if( p0 < p1 ) :
        min=p0;
        max=p1;
    else :
        min=p1;
        max=p0;
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];
    if( min > rad or max < -rad ) :
        return False;

    # AXISTEST_Z12(e2[Y], e2[X], fey, fex);
    a = e2[1];
    b = e2[0];
    fa = fey;
    fb = fex;
    p1 = a*v1[0] - b*v1[1];
    p2 = a*v2[0] - b*v2[1];
    if( p2 < p1 ) :
        min=p2;
        max=p1;
    else :
        min=p1
        max=p2;
    rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];
    if( min > rad or max < -rad ) :
        return False;


    # Bullet 1: first test overlap in the {x,y,z}-directions
    # find min, max of the triangle each direction, and test for overlap in
    # that direction -- this is equivalent to testing a minimal AABB around
    # the triangle against the AABB

    # test in X-direction

    # FINDMINMAX(v0[X],v1[X],v2[X],min,max);
    x0 = v0[0];
    x1 = v1[0];
    x2 = v2[0];
    min = x0;
    max = x0;
    if( x1 < min ) :
        min=x1;
    if( x1 > max ) :
        max=x1;
    if( x2 < min ) :
        min=x2;
    if( x2 > max ) :
        max=x2;

    if( min > boxhalfsize[0] or max < -boxhalfsize[0] ) :
        return False;

    # test in Y-direction

    # FINDMINMAX(v0[Y],v1[Y],v2[Y],min,max);
    x0 = v0[1];
    x1 = v1[1];
    x2 = v2[1];
    min = x0;
    max = x0;
    if( x1 < min ) :
        min=x1;
    if( x1 > max ) :
        max=x1;
    if( x2 < min ) :
        min=x2;
    if( x2 > max ) :
        max=x2;

    if( min > boxhalfsize[1] or max < -boxhalfsize[1] ) :
        return False;

    # test in Z-direction

    # FINDMINMAX(v0[Z],v1[Z],v2[Z],min,max);
    x0 = v0[2];
    x1 = v1[2];
    x2 = v2[2];
    min = x0;
    max = x0;
    if( x1 < min ) :
        min=x1;
    if( x1 > max ) :
        max=x1;
    if( x2 < min ) :
        min=x2;
    if( x2 > max ) :
        max=x2;

    if( min > boxhalfsize[2] or max < -boxhalfsize[2] ) :
        return False;


    # Bullet 2: test if the box intersects the plane of the triangle
    # compute plane equation of triangle: normal*x+d=0
    normal = np.cross( e0, e1 );

    # -NJMP- (line removed here)
    if( planeBoxOverlap( normal, v0, boxhalfsize ) == False ) :
        return False;

    return True;


def planeBoxOverlap( normal, vert, maxbox ) :
    vmin = [ 0.0, 0.0, 0.0 ];
    vmax = [ 0.0, 0.0, 0.0 ];
    for q in range( 3 ) :
        v = vert[q];
        if( normal[q] > 0.0 ) :
            vmin[q] = -maxbox[q] - v;
            vmax[q] =  maxbox[q] - v;
        else :
            vmin[q] =  maxbox[q] - v;
            vmax[q] = -maxbox[q] - v;
    if( np.dot( normal, vmin ) > 0.0 ) :
        return False;
    if( np.dot( normal, vmax ) >= 0.0 ) :
        return True;
    return False;




if( __name__ == "__main__" ) :

    mesh = Mesh( "T1.off" );

    """
    # ポリゴンスープをボクセル化
    voxelizer = Voxelizer( 32, "B" );
    voxel = voxelizer.sample( mesh );
    """

    # ポリゴンスープを点群化してからボクセル化
    voxelizer = Voxelizer( 32, "B" );
    pointsampler = PointSampler( 1024 );
    oript = pointsampler.sample( mesh );
    voxel = voxelizer.sample_from_pointset( oript[ :, 0:3 ] );

    quit();
