# -*- coding: utf-8 -*-
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44
from off import *

##### vertex shader (照明付きレンダリング用) #####
vertex_shader_light='''
        #version 330
        
        uniform mat4 Mvp;
        
        in vec3 in_vert;
        in vec3 in_norm;
        
        out vec3 v_vert;
        out vec3 v_norm;
        
        void main() {
            v_vert = in_vert;
            v_norm = in_norm;
            gl_Position = Mvp * vec4(v_vert, 1.0);
        }
''';

##### fragment shader (照明付きレンダリング用) #####
fragment_shader_light='''
        #version 330
        
        uniform vec4 Color;
        uniform vec3 Light;
        
        in vec3 v_vert;
        in vec3 v_norm;
        
        out vec4 f_color;
        
        void main() {
        
            // 法線向きの反転に対処
            float lum = dot(normalize(v_norm), normalize(v_vert - Light));
            lum = abs( lum );
            lum = acos(lum) / 1.570796325;
            lum = 1.0 - clamp(lum, 0.0, 1.0);
            

            // 通常の照明つきレンダリング            
            // float lum = dot(normalize(v_norm), normalize(v_vert - Light));
            // lum = acos(lum) / 3.14159265;
            // lum = clamp(lum, 0.0, 1.0);
            // lum = lum * lum;
            
            lum = smoothstep(0.0, 1.0, lum);
            lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
            lum = lum * 0.8 + 0.2;
        
            vec3 color = vec3( 1.0, 1.0, 1.0 ); // 面の色は白
            color = color * (1.0 - Color.a) + Color.rgb * Color.a;
            f_color = vec4(color * lum, 1.0);
        }
''';
###########################

##### vertex shader (深さ値レンダリング用) #####
vertex_shader_depth='''
        #version 330
        
        uniform mat4 Mvp;
        
        in vec3 in_vert;
        in vec3 in_norm;        
        
        out vec3 v_vert;
        
        void main() {
            v_vert = in_vert + in_norm * 1e-8; // コンパイルエラー回避のためにin_normを無理やり使用
            gl_Position = Mvp * vec4(v_vert, 1.0);
        }
''';

##### fragment shader (深さ値レンダリング用) #####
fragment_shader_depth='''
        #version 330
                
        in vec3 v_vert;
                
        out vec4 f_color;
        
        void main() {
            f_color = vec4( gl_FragCoord.z );
        }
''';


##### vertex shader (法線マップレンダリング用) #####
vertex_shader_normal='''
        #version 330
        
        uniform mat4 Mvp;
        uniform mat3 Rot;
        
        in vec3 in_vert;
        in vec3 in_norm;
        
        out vec3 v_vert;
        out vec3 v_norm;
        
        void main() {
            v_vert = in_vert;
            v_norm = Rot * in_norm;
            gl_Position = Mvp * vec4( v_vert, 1.0 );
        }
''';

##### fragment shader (法線マップレンダリング用) #####
fragment_shader_normal='''
        #version 330
        
        in vec3 v_vert;
        in vec3 v_norm;
        
        out vec4 f_color;
        
        void main() {
            // 法線ベクトル要素は[-1,1]の値を取りうるので，色の取りうる範囲[0,1]へ変換
            vec3 n = ( v_norm + 1.0 ) / 2.0;
            f_color = vec4( n, 1.0 );            
        }
''';
###########################


class Renderer :
    def __init__( self, eyes, eyepos, resolution, projection ) :

        if( eyepos == "V" ) : # 頂点からレンダリングする場合
            self.n_views = eyes.n_vert;
            self.views = eyes.vert;    
        elif( eyepos == "F" ) : # 面の重心からレンダリングする場合
            self.n_views = eyes.n_face;
            self.views = eyes.calc_face_gravity_center();
            norms = np.linalg.norm( self.views, axis=1 );
            max_norm = np.max( norms );
            self.views = self.views * ( 1.0 / max_norm );            
        else :
            print( "error: invalid camera position : " + camera );
            quit();

        self.resolution = resolution;

        # contextを生成
        self.ctx = moderngl.create_standalone_context(); 

        # シェーダプログラムを生成
        self.prog_light = self.ctx.program( vertex_shader=vertex_shader_light,
                                  fragment_shader=fragment_shader_light );
        self.prog_depth = self.ctx.program( vertex_shader=vertex_shader_depth,
                                  fragment_shader=fragment_shader_depth );
        self.prog_normal = self.ctx.program( vertex_shader=vertex_shader_normal,
                                   fragment_shader=fragment_shader_normal );
        
        # Framebuffers
        self.fbo = self.ctx.framebuffer( self.ctx.renderbuffer( ( self.resolution, self.resolution ) ),
                                    self.ctx.depth_renderbuffer( ( self.resolution, self.resolution ) ) );

        # input buffer
        self.buffer = self.ctx.buffer( data=None, reserve=350000000, dynamic=True ); # 350MByteのバッファを確保  (ModelNet40 train setにこのくらいのバッファが必要な3Dモデルがある)      
        
        # vertex arrays
        self.vao_light = self.ctx.simple_vertex_array( self.prog_light, self.buffer, 'in_vert', 'in_norm' );
        self.vao_depth = self.ctx.simple_vertex_array( self.prog_depth, self.buffer, 'in_vert', 'in_norm' );
        self.vao_normal = self.ctx.simple_vertex_array( self.prog_normal, self.buffer, 'in_vert', 'in_norm' );

        if( projection == "O" ) : # 平行投影の場合
            self.projection = Matrix44.orthogonal_projection(  1.0,   # left
                                                              -1.0,   # right
                                                              -1.0,   # top
                                                               1.0,   # bottom
                                                               0.0,   # near
                                                               2.0 ); # far
        elif( projection == "P" ) : # 透視投影の場合
            self.projection = Matrix44.perspective_projection( 90.0,     # fovy  
                                                                1.0,     # aspect
                                                                0.01,    # near
                                                                2.0 );   # far
        else :
            print( "error: invalid projection setting : " + projection );
            quit();
        
        self.count = 0;
                
    def render( self, camera, n_render_vert ) :               
    
        # upright = [ 0.0, 1.0, 0.0 ];
        upright = [ 0.0, 0.0, -1.0 ]; # ModelNet10/40だとこれがupright
        
        cos = np.dot( camera, upright ) / ( np.linalg.norm( camera ) * np.linalg.norm( upright ) );
        
        if( np.abs( np.abs( cos ) - 1.0 ) < 1e-6 ) : # カメラ位置とuprightベクトルが平行の場合はuprightベクトルを変更
            upright = [ 1.0, 0.0, 0.0 ];  
            
        lookat = Matrix44.look_at(
            ( camera[0], camera[1], camera[2] ),      # camera position
            ( 0.0, 0.0, 0.0 ),                        # camera direction
            ( upright[0], upright[1], upright[2] ) ); # upright vector
   
        mvp = self.projection * lookat;

        # 法線レンダリング用回転行列
        rot = lookat[0:3,0:3]; # 4x4のlookat行列から3x3の回転行列を抜粋
        
        # 各シェーダプログラムに光源と投影行列を設定
        self.prog_light.get( 'Light', 0 ).value = ( camera[0], camera[1], camera[2] );   # 光源位置
        self.prog_light.get( 'Color', 0 ).value = (1.0, 1.0, 1.0, 0.25);     # 光の色
        self.prog_light.get( 'Mvp', 0 ).write( mvp.astype('f4').tobytes() ); # 投影行列
        
        ### self.prog_depth.get( 'Mvp', 0 ).write( mvp.astype('f4').tobytes() ); # 投影行列
        ### self.prog_normal.get( 'Mvp', 0 ).write( mvp.astype('f4').tobytes() ); # 投影行列
        ### self.prog_normal.get( 'Rot', 0 ).write( rot.astype('f4').tobytes() ); # lookat行列
    
        self.fbo.use();
        self.ctx.enable( moderngl.DEPTH_TEST );

        # 照明付きレンダリング
        self.ctx.clear( 1.0, 1.0, 1.0 );
        self.vao_light.render( vertices = n_render_vert );
        data_light = self.fbo.read( components=1, alignment=1 );        
        
        # 深さ値レンダリング
        ### self.ctx.clear( 1.0, 1.0, 1.0 );
        ### self.vao_depth.render( vertices = n_render_vert );
        ### data_depth = self.fbo.read( components=1, alignment=1 );
        
        # 法線マップレンダリング
        ### self.ctx.clear( 1.0, 1.0, 1.0 );
        ### self.vao_normal.render( vertices = n_render_vert );
        ### data_normal = self.fbo.read( components=3, alignment=1 );
        
        ### return ( data_light, data_depth, data_normal );
        return data_light;

       
    def multiview_rendering( self, target ) :
            
        target_array = target.generate_array_for_rendering();
        n_render_vert = target.n_face * 3; # vertex shaderが処理する頂点数
        self.buffer.write( target_array );

        ### rendered_imgs = np.zeros( ( self.n_views, self.resolution, self.resolution, 5 ), dtype=np.float32 );        
        rendered_imgs = np.zeros( ( self.n_views, self.resolution, self.resolution, 1 ), dtype=np.float32 );        
        
        for i in range( self.n_views ) :
            # print( "view: " + str( i ) );
            
            ### light, depth, normal = self.render( self.views[ i ], n_render_vert );
            light = self.render( self.views[ i ], n_render_vert );
            
            light = np.frombuffer( light, dtype=np.uint8 );
            ### depth = np.frombuffer( depth, dtype=np.uint8 );
            ### normal = np.frombuffer( normal, dtype=np.uint8 );

            """            
            # 画像ファイルとして書き出す
            light = np.reshape( light, [ self.resolution, self.resolution ] );
            depth = np.reshape( depth, [ self.resolution, self.resolution ] );
            normal = np.reshape( normal, [ self.resolution, self.resolution, 3 ] );
            img = Image.fromarray( light );
            img.save( "output_light_" + str(self.count) + "_" + str(i) + ".png" );
            img = Image.fromarray( depth );
            img.save( "output_depth_" + str(self.count) + "_" + str(i) + ".png" );
            img = Image.fromarray( normal );
            img.save( "output_normal_" + str(self.count) + "_" + str(i) + ".png" );                                     
            """            
            
            light = np.reshape( light, [ self.resolution, self.resolution, 1 ] );
            ### depth = np.reshape( depth, [ self.resolution, self.resolution, 1 ] );
            ### normal = np.reshape( normal, [ self.resolution, self.resolution, 3 ] );
            
            light = light.astype( np.float32 ) / 255.0;
            ### depth = depth.astype( np.float32 ) / 255.0;
            ### normal = normal.astype( np.float32 ) / 255.0;
            
            ### rendered_imgs[ i ] = np.dstack( [ light, depth, normal ] );  
            rendered_imgs[ i ] = light;

        self.count += 1;

        return rendered_imgs;            


def save_rendered_image( img, prefix, minval=0.0, maxval=1.0 ) :
    # レンダリングした画像(5チャンネル)を深さ，陰影，法線の3枚の画像ファイルに書き出す
    res = img.shape[ 0 ]; # 画像は正方形とする
    
    img = ( img - minval ) / ( maxval - minval ); # [0,1]にスケーリング
    
    tmpimg = ( img * 255.0 ).astype( np.uint8 );    
    light = tmpimg[ :, :, 0 ];
    ### depth = tmpimg[ :, :, 1 ];
    ### normal = tmpimg[ :, :, 2:5 ];
    
    light = np.reshape( light, [ res, res ] );
    ### depth = np.reshape( depth, [ res, res ] );
    ### normal = np.reshape( normal, [ res, res, 3 ] );
    
    img = Image.fromarray( light );
    img.save( prefix + "_" + "light" + ".png" );
    ### img = Image.fromarray( depth );
    ### img.save( prefix + "_" + "depth" + ".png" );
    ### img = Image.fromarray( normal );
    ### img.save( prefix + "_" + "normal" + ".png" );
        