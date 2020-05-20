# -*- coding: utf-8 -*-
import numpy as np

class Mesh :
    def __init__( self, in_filepath, radius=1.0 ) :
        self.n_vert = 0;  # 頂点数
        self.n_face = 0;  # 面数
        # self.n_edge = 0; # 辺数
        self.vert = None; # 各頂点の座標．サイズは(n_vert x 3)．
        self.face = None; # 各面を構成する頂点ID．3角形を想定するのでサイズは(n_face x 3)．
        self.norm_face = None; # 各面の法線ベクトル
        self.norm_vert = None; # 各頂点の法線ベクトル
        self.area_face = None; # 各面の面積
        self.area_total = 0;   # 表面積

        # print( "loading " + in_filepath );
        
        self.load_off( in_filepath );
        self.calc_norm();
        self.normalize_pos_scale( radius );
        self.calc_area();

    ##### OFF形式ファイルを読み込む #####
    def load_off( self, in_filepath ) :
        f = open( in_filepath );
        data = f.read();  # ファイル終端まで全て読んだデータを返す
        f.close();
         
        lines = data.split('\n') # 改行で区切る
        
        headers = lines[ 1 ].split();
        self.n_vert = int( headers[ 0 ] ); # 頂点数
        self.n_face = int( headers[ 1 ] ); # 面数
        
        # 頂点座標を読み込む        
        self.vert = np.zeros( ( self.n_vert, 3 ), dtype=np.float32 );
        for i in range( self.n_vert ) :
            v = lines[ 2 + i ].split();
            self.vert[ i, 0 ] = float( v[ 0 ] );
            self.vert[ i, 1 ] = float( v[ 1 ] );
            self.vert[ i, 2 ] = float( v[ 2 ] );

        # 面を構成する頂点を読み込む        
        self.face = np.zeros( ( self.n_face, 3 ), dtype=np.int32 );
        for i in range( self.n_face ) :
            f = lines[ 2 + self.n_vert + i ].split();
            if( int( f[ 0 ] ) != 3 ) :
                print( "error: supports only triangles." );
                quit();
            self.face[ i, 0 ] = int( f[ 1 ] );
            self.face[ i, 1 ] = int( f[ 2 ] );
            self.face[ i, 2 ] = int( f[ 3 ] );

    ##### OFF形式ファイルを書き出す #####
    def save_off( self, out_filepath ) :
        f = open( out_filepath, 'w' );        
        f.write( "OFF\n" );
        f.write( str( self.n_vert ) + " " + str( self.n_face ) + " 0\n" );
        for i in range( self.n_vert ) :
            f.write( str( self.vert[ i ][ 0 ] ) + " " + str( self.vert[ i ][ 1 ] ) + " " + str( self.vert[ i ][ 2 ] ) + "\n" );
        for i in range( self.n_face ) :
            f.write( "3 " + str( self.face[ i ][ 0 ] ) + " " + str( self.face[ i ][ 1 ] ) + " " + str( self.face[ i ][ 2 ] ) + "\n" ); 
        f.close();

    ##### 法線ベクトルを計算  #####
    def calc_norm( self ) :
        # 面の法線を計算    
        self.norm_face = np.zeros( ( self.n_face, 3 ), dtype=np.float32 );
        for i in range( self.n_face ) :
            v0 = self.vert[ self.face[ i, 0 ] ];
            v1 = self.vert[ self.face[ i, 1 ] ];
            v2 = self.vert[ self.face[ i, 2 ] ];
            vv1 = v1 - v0;
            vv2 = v2 - v1;        
            cross = np.cross( vv1, vv2 );
            
            norm = np.linalg.norm( cross );
            if( norm < 1e-6 ) :
                normvec = [ 0.0, 0.0, 0.0 ];
            else :               
                normvec = cross / norm;
            self.norm_face[ i ] = normvec;
        
        # 頂点->その頂点が属する面IDリスト
        vert2face = [];
        for i in range( self.n_vert ) :
            vert2face.append( [] );
        for i in range( self.n_face ) :
            vert2face[ self.face[ i, 0 ] ].append( i );
            vert2face[ self.face[ i, 1 ] ].append( i );
            vert2face[ self.face[ i, 2 ] ].append( i );        
        
        # 頂点の法線を計算    
        self.norm_vert = np.zeros( ( self.n_vert, 3 ), dtype=np.float32 );
        for i in range( self.n_vert ) :
            normvec = [ 0.0, 0.0, 0.0 ];
            for j in range( len( vert2face[ i ] ) ) :
                normvec += self.norm_face[ vert2face[ i ][ j ] ];
            
            norm = np.linalg.norm( normvec );
            if( norm < 1e-6 ) :
                normvec = [ 0.0, 0.0, 0.0 ];
            else :            
                normvec = normvec / norm;
            self.norm_vert[ i ] = normvec;
                
        """
        tmp = np.concatenate( ( self.vert, self.norm_vert ), axis = 1 );
        np.savetxt( "out.xyz", tmp );     
        """    

    ##### 位置とスケールの正規化 #####        
    def normalize_pos_scale( self, radius ) :

        # 位置の正規化        
        # 頂点群の重心で正規化する場合
        # mean = np.mean( self.vert, axis=0 );
        # self.vert = self.vert - mean;
        
        # 頂点群のbounding boxで正規化する場合
        bbcenter = ( np.max( self.vert, 0 ) + np.min( self.vert, 0 ) ) / 2.0;
        self.vert = self.vert - bbcenter;
        
        # スケールの正規化 (半径radiusの球に収まるようにスケーリング)
        norms = np.linalg.norm( self.vert, axis=1 );
        max_norm = np.max( norms );
        self.vert = self.vert * ( radius / max_norm );
        
    ##### 面の重心を計算 #####
    def calc_face_gravity_center( self ) :
        face_gc = np.zeros( ( self.n_face, 3 ), dtype=np.float32 );
        for i in range( self.n_face ) :
            mean = [ 0.0, 0.0, 0.0 ];
            for j in range( 3 ) :
                mean += self.vert[ self.face[ i, j ] ];
            face_gc[ i ] = mean / 3.0;
        return face_gc;
        
    ##### 面の面積を計算 #####
    def calc_area( self ) :
        self.area_face = np.zeros( self.n_face, dtype=np.float32 );
        self.area_total = 0.0;
        for i in range( self.n_face ) :
            a = self.vert[ self.face[ i, 0 ] ];
            b = self.vert[ self.face[ i, 1 ] ];
            c = self.vert[ self.face[ i, 2 ] ];
            v1 = a - b;      
            v2 = c - b;
            tmp = np.cross( v1, v2 ); 
            tmp = 0.5 * np.linalg.norm( tmp );
            self.area_face[ i ] = tmp;
            self.area_total += tmp;
    
    ##### ランダムに平行移動，回転 (データ拡張用) #####
    def randomize( self ) :
        
        flag_scale = True; 
        flag_rotate = True;
        flag_translate = True;
        
        if( flag_scale ) : # 非等方変形
            # [range_from, range_to]の範囲で拡大率を決定
            range_from = 0.8;
            range_to = 1.2;
            # range_from = 0.9;
            # range_to = 1.1;
            sc_x = ( np.random.random() * ( range_to - range_from ) ) + range_from;
            sc_y = ( np.random.random() * ( range_to - range_from ) ) + range_from;
            sc_z = ( np.random.random() * ( range_to - range_from ) ) + range_from;

            self.vert[:,0] *= sc_x;
            self.vert[:,1] *= sc_y;
            self.vert[:,2] *= sc_z;
            
            self.norm_face[:,0] *= 1.0 / sc_x;
            self.norm_face[:,1] *= 1.0 / sc_y;
            self.norm_face[:,2] *= 1.0 / sc_z;
            self.norm_face /= np.linalg.norm( self.norm_face, axis=1, keepdims=True ) + 1e-8;
            
            self.norm_vert[:,0] *= 1.0 / sc_x;
            self.norm_vert[:,1] *= 1.0 / sc_y;
            self.norm_vert[:,2] *= 1.0 / sc_z;
            self.norm_vert /= np.linalg.norm( self.norm_vert, axis=1, keepdims=True ) + 1e-8;
            
            # スケールの正規化 (半径1の球に収まるようにスケーリング)
            norms = np.linalg.norm( self.vert, axis=1 );
            max_norm = np.max( norms );
            self.vert *= 1.0 / max_norm;

        if( flag_rotate ) :  # 回転
            degree = 20.0;
            # degree = 10.0;
            rot_mat = get_random_rotation( degree );
            self.vert = np.matmul( self.vert, rot_mat );
            self.norm_face = np.matmul( self.norm_face, rot_mat );
            self.norm_vert = np.matmul( self.norm_vert, rot_mat );
           
        if( flag_translate ) :  # 平行移動
            # [range_from, range_to]の範囲で平行移動
            range_from = -0.2;
            range_to = 0.2;
            # range_from = -0.1;
            # range_to = 0.1;
            tr_x = ( np.random.random() * ( range_to - range_from ) ) + range_from;
            tr_y = ( np.random.random() * ( range_to - range_from ) ) + range_from;
            tr_z = ( np.random.random() * ( range_to - range_from ) ) + range_from;
            self.vert[:,0] += tr_x;
            self.vert[:,1] += tr_y;
            self.vert[:,2] += tr_z;



    ##### x軸周りに90度回転 (SH17LSのuprightベクトルをModelNet10/40と同じにするため) #####
    def rotate_x_90( self ) :

        # Y/Z -> -Z/Y

        # 頂点を回転        
        self.vert[ :, 2 ] *= -1.0;
        z = np.copy( self.vert[ :, 2 ] );
        self.vert[ :, 2 ] = self.vert[ :, 1 ];
        self.vert[ :, 1 ] = z;
        
        # 面の法線を回転        
        self.norm_face[ :, 2 ] *= -1.0;
        z = np.copy( self.norm_face[ :, 2 ] );
        self.norm_face[ :, 2 ] = self.norm_face[ :, 1 ];
        self.norm_face[ :, 1 ] = z;

        # 頂点の法線を回転        
        self.norm_vert[ :, 2 ] *= -1.0;
        z = np.copy( self.norm_vert[ :, 2 ] );
        self.norm_vert[ :, 2 ] = self.norm_vert[ :, 1 ];
        self.norm_vert[ :, 1 ] = z;
        
        return;
    
    ##### vertex shaderに渡すバイト列を生成 (面を構成する頂点の座標と法線)#####
    def generate_array_for_rendering( self ) :
        idx = np.reshape( self.face, [ 1, -1 ] ).squeeze();
        vertices = self.vert[ idx ];
        normvecs = self.norm_vert[ idx ];
        bytes = np.hstack( [ vertices, normvecs ] );
        bytes = bytes.astype('f4').tobytes();
        return bytes;        




def get_random_rotation( angle=180.0 ) :
    # [-angle, +angle]の範囲から回転角度を決定
    rad_from = -angle * ( np.pi / 180.0 );
    rad_to = angle * ( np.pi / 180.0 );

    # 各軸の回転角度 [rad]をランダムに決定
    x_rotate = ( np.random.random() * ( rad_to - rad_from ) ) + rad_from;
    y_rotate = ( np.random.random() * ( rad_to - rad_from ) ) + rad_from;
    z_rotate = ( np.random.random() * ( rad_to - rad_from ) ) + rad_from;

    # x軸周りの回転行列
    rot_mat_x = [ [ 1.0, 0.0, 0.0 ],
                  [ 0.0, np.cos(x_rotate), -1.0 * np.sin(x_rotate) ],
                  [ 0.0, np.sin(x_rotate), np.cos(x_rotate) ] ];
               
    # y軸周りの回転行列
    rot_mat_y = [ [ np.cos(y_rotate), 0.0, np.sin(y_rotate) ],
                  [ 0.0, 1.0, 0.0 ],
                  [ -1.0 * np.sin(y_rotate), 0.0, np.cos(y_rotate) ] ];

    # z軸周りの回転行列
    rot_mat_z = [ [ np.cos(z_rotate), -1.0 * np.sin(z_rotate), 0.0 ],
                  [ np.sin(z_rotate), np.cos(z_rotate), 0.0 ],
                  [ 0.0, 0.0, 1.0 ] ];

    rot_mat = np.matmul( np.matmul( rot_mat_x, rot_mat_y ), rot_mat_z );           
    return rot_mat;


if( __name__ == "__main__" ) :

    mesh = Mesh( "icosahedron.off" );
    quit();
    