# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import time
import os
import random
import math
import copy
from ops_dnn import *

sys.path.append( 'utils' );
import common_util as cu
import retrieval

# 学習係数
BASE_LEARNING_RATE = 0.002
DECAY_STEP = 10000
DECAY_RATE = 0.975
MIN_LEARNING_RATE = 0.000001

# Batch normalization移動平均の減衰
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

alpha = 0.01;
do_rotate = False;
# alpha = 0.3;
# do_rotate = True;


class DNN( object ) :
    #########################
    def __init__( self, sess, device, minibatch_size,
                  n_class, n_models_train, n_dim_embed, dropout_rate, sr, checkpoint ) :

        self.sess = sess;
        self.device = device;
        self.minibatch_size = minibatch_size;          # ミニバッチ当たりの3Dモデル数

        self.render_n_view = sr.render_n_view;          # 視点数
        self.render_resolution = sr.render_resolution;  # レンダリング画像の解像度
        self.render_n_channel = 1;                      # レンダリング画像のチャネル数

        self.point_n_point = sr.point_n_point;          # 1モデル当たりの点の数
        self.point_n_channel = 6;                       # 点群モデルのチャンネル数

        self.voxel_resolution = sr.voxel_resolution;    # ボクセル画像の解像度
        self.voxel_n_channel = 1;                       # ボクセル画像のチャネル数

        self.n_class = n_class;
        self.n_models_train = n_models_train;
        self.dropout_keep_prob = 1.0 - dropout_rate;

        self.n_dim_embed = n_dim_embed;                 # 埋込次元数

        self.checkpoint_dir = checkpoint;               # 学習したDNNの読込/保存ディレクトリ

        self.build_model();

    #########################
    def build_model( self ) :

        with tf.device( self.device ) :

            # 点群オートエンコーダへの入力
            self.oript_in = tf.placeholder( tf.float32, [ self.minibatch_size * self.point_n_point, self.point_n_channel ], name='oript_in' );
            self.oript_gt = tf.placeholder( tf.float32, [ self.minibatch_size * self.point_n_point, self.point_n_channel ], name='oript_gt' );

            # ボクセルオートエンコーダへの入力
            self.voxel_in = tf.placeholder( tf.float32, [ self.minibatch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, self.voxel_n_channel ], name='voxel_in' );
            self.voxel_gt = tf.placeholder( tf.float32, [ self.minibatch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, self.voxel_n_channel ], name='voxel_gt' );

            # 多視点画像オートエンコーダへの入力
            self.mvimg_in = tf.placeholder( tf.float32, [ self.minibatch_size * self.render_n_view, self.render_resolution, self.render_resolution, self.render_n_channel ], name='mvimg_in' );
            self.mvimg_gt = tf.placeholder( tf.float32, [ self.minibatch_size * self.render_n_view, self.render_resolution, self.render_resolution, self.render_n_channel ], name='mvimg_gt' );

            # 全DNNで共通の入力
            self.flag_train = tf.placeholder( tf.bool, name='flag_train' );
            self.keep_prob = tf.placeholder( tf.float32, name='keep_prob' );
            self.batch = tf.Variable(0);                                     # 計何バッチ学習したか
            self.bn_decay = self.get_bn_decay( self.batch );                 # BatchNormalization用decay
            self.learning_rate = self.get_learning_rate( self.batch );       # 学習率

            # エンコーダの構造
            self.oript_feat = self.oript_encoder( self.oript_in );
            self.voxel_feat = self.voxel_encoder( self.voxel_in );
            self.mvimg_feat = self.mvimg_encoder( self.mvimg_in );

            # switching
            self.oript_feat_switched, self.voxel_feat_switched, self.mvimg_feat_switched = self.switching_layer( self.oript_feat, self.voxel_feat, self.mvimg_feat );

            # デコーダの構造
            self.oript_out = self.oript_decoder( self.oript_feat_switched );
            self.voxel_out = self.voxel_decoder( self.voxel_feat_switched );
            self.mvimg_out = self.mvimg_decoder( self.mvimg_feat_switched );

            # 点群オートエンコーダの損失関数
            self.oript_loss = get_set_reconstruction_loss( self.oript_gt, self.oript_out, self.minibatch_size, self.point_n_point, "L2" );

            # ボクセルオートエンコーダの損失関数 (参考： Generative and Discriminative Voxel Modeling with Convolutional Neural Networks)
            vpred = self.voxel_out * 0.98 + 0.01;  # [0.01, 0.99]の範囲
            vgt = self.voxel_gt;  # density voxelの場合は[0,1]の範囲.binary voxelの場合は0または1.
            gamma = 0.97;
            self.voxel_loss = tf.reduce_mean( - gamma * vgt * tf.log( vpred ) - (1.0-gamma) * ( 1.0 - vgt ) * tf.log( 1.0 - vpred ) );

            # 多視点画像オートエンコーダの損失関数
            ipred = tf.reshape( self.mvimg_out, [ self.minibatch_size * self.render_n_view, -1 ] );
            igt = tf.reshape( self.mvimg_gt, [ self.minibatch_size * self.render_n_view, -1 ] );
            ndim = igt.get_shape().as_list()[-1];
            self.mvimg_loss = tf.reduce_mean( tf.reduce_sum( tf.abs( igt - ipred ), reduction_indices=[1] ) / ndim );

            self.overall_loss = self.oript_loss + self.voxel_loss + self.mvimg_loss;

            self.center_loss = self.get_center_loss( self.oript_feat_switched, self.voxel_feat_switched, self.mvimg_feat_switched );
            self.overall_loss += alpha * self.center_loss;

            self.saver = tf.train.Saver();


    #########################
    def oript_encoder( self, input ) :

        # Graph convolution PointNet 参考: FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation
        knn = 16;
        nn_idx = tf_nearest_neighbor( input[ :, 0:3 ], knn, self.minibatch_size, self.point_n_point ); # 近傍探索
        local_cov = tf_local_covariance( input, nn_idx, self.minibatch_size, self.point_n_point ); # 局所領域の共分散
        h = tf.concat( [ input, local_cov ], 1 ); # 有向点群に共分散を連結

        h = linear( h, 64, "oript_enc_fc1" );
        h = relu( batch_norm_for_fc( h, self.flag_train, self.bn_decay, "oript_enc_bn1" ) );

        h = linear( h, 64, "oript_enc_fc2" );
        h = relu( batch_norm_for_fc( h, self.flag_train, self.bn_decay, "oript_enc_bn2" ) );

        h = linear( h, 64, "oript_enc_fc3" );
        h = relu( batch_norm_for_fc( h, self.flag_train, self.bn_decay, "oript_enc_bn3" ) );

        h = tf_local_pooling( h, nn_idx, "M", self.minibatch_size, self.point_n_point ); # 局所領域のプーリング

        h = linear( h, 128, "oript_enc_fc4" );
        h = relu( batch_norm_for_fc( h, self.flag_train, self.bn_decay, "oript_enc_bn4" ) );

        h = tf_local_pooling( h, nn_idx, "M", self.minibatch_size, self.point_n_point ); # 局所領域のプーリング

        h = linear( h, 1024, "oript_enc_fc5" );
        h = relu( batch_norm_for_fc( h, self.flag_train, self.bn_decay, "oript_enc_bn5" ) );

        h = aggregate( h, "M", self.minibatch_size, self.point_n_point ); # global pooling

        h = linear( h, 512, "oript_enc_fc6" );

        h = relu( batch_norm_for_fc( h, self.flag_train, self.bn_decay, "oript_enc_bn6" ) );

        h = linear( h, self.n_dim_embed, "oript_enc_fc7" );

        h = l2normalize( h );

        return h;

    #########################
    def oript_decoder( self, input ) :

        # 全結合デコーダ
        h = linear( input, 512, "oript_dec_fc1" );
        h = relu( h );

        h = linear( h, 1024, "oript_dec_fc2" );
        h = relu( h );

        h = linear( h, self.point_n_point * self.point_n_channel, "oript_dec_fc3" );
        h = tf.reshape( h, [ self.minibatch_size * self.point_n_point, self.point_n_channel ] );

        # 法線のノルムを1に正規化
        pos = h[ :, 0:3 ];
        normal = h[ :, 3:6 ];
        normal = l2normalize( normal );
        h = tf.concat( [ pos, normal ], 1 );

        return h;

    #########################
    def voxel_encoder( self, input ) :

        # Residual 3D CNN
        h = conv3d( input, 16, 3, 1, "voxel_enc_conv1" );
        h = relu( h );
        h = tf.nn.max_pool3d( h, [ 1, 2, 2, 2, 1 ], [ 1, 2, 2, 2, 1 ], 'SAME' );

        h = res_block_3d( h, 32, self.flag_train, self.bn_decay, "voxel_enc_res1" );
        h = res_block_3d( h, 32, self.flag_train, self.bn_decay, "voxel_enc_res2" );
        h = tf.nn.max_pool3d( h, [ 1, 2, 2, 2, 1 ], [ 1, 2, 2, 2, 1 ], 'SAME' );

        h = res_block_3d( h, 64, self.flag_train, self.bn_decay, "voxel_enc_res3" );
        h = res_block_3d( h, 64, self.flag_train, self.bn_decay, "voxel_enc_res4" );
        h = tf.nn.max_pool3d( h, [ 1, 2, 2, 2, 1 ], [ 1, 2, 2, 2, 1 ], 'SAME' );

        h = res_block_3d( h, 128, self.flag_train, self.bn_decay, "voxel_enc_res5" );
        h = res_block_3d( h, 128, self.flag_train, self.bn_decay, "voxel_enc_res6" );
        h = tf.nn.max_pool3d( h, [ 1, 2, 2, 2, 1 ], [ 1, 2, 2, 2, 1 ], 'SAME' );

        h = tf.reshape( h, [ self.minibatch_size, -1 ] ); # flatten

        h = linear( h, 512, "voxel_enc_fc1" );
        h = relu( h );

        h = linear( h, self.n_dim_embed, "voxel_enc_fc2" );

        h = l2normalize( h );

        return h;

    #########################
    def voxel_decoder( self, input ) :

        h = linear( input, 8 * 8 * 8 * 128, "voxel_dec_fc1" );
        h = relu( h );
        h = tf.reshape( h, [ self.minibatch_size, 8, 8, 8, 128 ] );

        h = deconv3d( h, [ self.minibatch_size, 8, 8, 8, 64 ], 3, 1, "voxel_dec_deconv0" );
        h = relu( h );

        h = deconv3d( h, [ self.minibatch_size, 16, 16, 16, 32 ], 3, 2, "voxel_dec_deconv1" );
        h = relu( h );

        h = deconv3d( h, [ self.minibatch_size, 16, 16, 16, 16 ], 3, 1, "voxel_dec_deconv2" );
        h = relu( h );

        h = deconv3d( h, [ self.minibatch_size, 32, 32, 32, 8 ], 3, 2, "voxel_dec_deconv3" );
        h = relu( h );

        h = deconv3d( h, [ self.minibatch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, self.voxel_n_channel ], 3, 1, "voxel_dec_deconv4" );

        h = tf.nn.sigmoid( h );

        return h;


    #########################
    def mvimg_encoder( self, input ) :

        # Residual 2D CNN
        h = conv2d( input, 32, 5, 1, "mvimg_enc_conv1" );
        h = relu( h );
        h = max_pool( h );

        h = res_block( h, 64, self.flag_train, self.bn_decay, "mvimg_enc_res1" );
        h = res_block( h, 64, self.flag_train, self.bn_decay, "mvimg_enc_res2" );
        h = max_pool( h );

        h = res_block( h, 128, self.flag_train, self.bn_decay, "mvimg_enc_res3" );
        h = res_block( h, 128, self.flag_train, self.bn_decay, "mvimg_enc_res4" );
        h = max_pool( h );

        h = res_block( h, 256, self.flag_train, self.bn_decay, "mvimg_enc_res5" );
        h = res_block( h, 256, self.flag_train, self.bn_decay, "mvimg_enc_res6" );
        h = max_pool( h );

        h = tf.reshape( h, [ self.minibatch_size, -1 ] ); # flatten

        h = linear( h, 512, "mvimg_enc_fc1" );
        h = relu( h );

        h = linear( h, self.n_dim_embed, "mvimg_enc_fc2" );

        h = l2normalize( h );

        return h;

    #########################
    def mvimg_decoder( self, input ) :

        h = linear( input, 4 * 4 * 512 * self.render_n_view, "mvimg_dec_fc1" );
        h = relu( h );
        h = tf.reshape( h, [ self.minibatch_size * self.render_n_view, 4, 4, 512 ] );

        h = deconv2d( h, [ self.minibatch_size * self.render_n_view, 8, 8, 256 ], 3, 2, "mvimg_dec_deconv1" );
        h = relu( batch_norm_for_conv2d( h, self.flag_train, self.bn_decay, "mvimg_dec_bn1" ) );

        h = deconv2d( h, [ self.minibatch_size * self.render_n_view, 16, 16, 128 ], 3, 2, "mvimg_dec_deconv2" );
        h = relu( batch_norm_for_conv2d( h, self.flag_train, self.bn_decay, "mvimg_dec_bn2" ) );

        h = deconv2d( h, [ self.minibatch_size * self.render_n_view, 32, 32, 64 ], 3, 2, "mvimg_dec_deconv3" );
        h = relu( batch_norm_for_conv2d( h, self.flag_train, self.bn_decay, "mvimg_dec_bn3" ) );

        h = deconv2d( h, [ self.minibatch_size * self.render_n_view, 64, 64, 32 ], 3, 2, "mvimg_dec_deconv4" );
        h = relu( batch_norm_for_conv2d( h, self.flag_train, self.bn_decay, "mvimg_dec_bn4" ) );

        h = deconv2d( h, [ self.minibatch_size * self.render_n_view, self.render_resolution, self.render_resolution, self.render_n_channel ], 3, 1, "mvimg_dec_deconv5" );

        h = tf.nn.sigmoid( h );

        return h;

    #########################
    def switching_layer( self, ofeat, vfeat, ifeat ) :

        n_dim = ofeat.get_shape().as_list()[ 1 ];

        x = tf.concat( [ ofeat, vfeat, ifeat ], 1 );
        x = tf.reshape( x, [ -1, n_dim ] );
        x = tf.split( x, self.minibatch_size );

        x_perm = [];
        for i in range( self.minibatch_size ) :
            perm = tf.random_shuffle( tf.eye( 3 ) );
            x_perm.append( tf.matmul( perm, x[ i ] ) );

        x_perm = tf.concat( x_perm, 0 );
        x_perm = tf.reshape( x_perm, [ -1, n_dim * 3 ] );
        x_perm = tf.split( x_perm, 3, axis=1 );

        return x_perm[0], x_perm[1], x_perm[2];


    def get_center_loss( self, ofeat, vfeat, ifeat ) :

        center = ( ofeat + vfeat + ifeat ) / 3.0;
        center = l2normalize( center );

        center_neg = [];
        for i in range( self.minibatch_size - 1 ) :
            center_neg.append( tf.reshape( center[ i + 1 ], [ 1, -1 ] ) );
        center_neg.append( tf.reshape( center[ 0 ], [ 1, -1 ] ) );
        center_neg = tf.concat( center_neg, 0 );

        odist_pos = sqrt( tf.reduce_sum( tf.square( ofeat - center ), reduction_indices=[1] ) ); # 正例ペア間距離
        odist_neg = sqrt( tf.reduce_sum( tf.square( ofeat - center_neg ), reduction_indices=[1] ) ); # 負例ペア間距離
        vdist_pos = sqrt( tf.reduce_sum( tf.square( vfeat - center ), reduction_indices=[1] ) ); # 正例ペア間距離
        vdist_neg = sqrt( tf.reduce_sum( tf.square( vfeat - center_neg ), reduction_indices=[1] ) ); # 負例ペア間距離
        idist_pos = sqrt( tf.reduce_sum( tf.square( ifeat - center ), reduction_indices=[1] ) ); # 正例ペア間距離
        idist_neg = sqrt( tf.reduce_sum( tf.square( ifeat - center_neg ), reduction_indices=[1] ) ); # 負例ペア間距離

        margin = 1.0;
        oloss = tf.reduce_mean( tf.maximum( 0.0, margin + odist_pos - odist_neg ) );
        vloss = tf.reduce_mean( tf.maximum( 0.0, margin + vdist_pos - vdist_neg ) );
        iloss = tf.reduce_mean( tf.maximum( 0.0, margin + idist_pos - idist_neg ) );
        loss = ( oloss + vloss + iloss ) / 3.0;

        # Min-CV loss 参考： Learning 3D Keypoint Descriptors for Non-Rigid Shape Matching
        mu, sigma = tf.nn.moments( tf.concat( [ odist_pos, vdist_pos, idist_pos ], 0 ), axes=0 );
        loss += 1.0 * sigma / mu;

        return loss;

    #########################
    def train( self, sr_train, sr_test, Label_train, Label_test, n_epoch, ctg2id,
               in_list_filename_train, in_label_filename_train,
               in_list_filename_test, in_label_filename_test ) :

        with tf.device( self.device ) :

            self.modellist_train = cu.readList( in_list_filename_train );
            self.modellist_test = cu.readList( in_list_filename_test );
            self.id2ctg = {v: k for k, v in ctg2id.items()}; # カテゴリ番号->カテゴリ名への対応


            optimizer = tf.train.AdamOptimizer( self.learning_rate );
            optim = optimizer.minimize( self.overall_loss, global_step = self.batch );

            self.sess.run( tf.global_variables_initializer() );

            # 学習済みパラメタがあれば読み込む
            if( self.load() ) :
                print( "Pretrained parameters exist." );
            else :
                print( "Pretrained parameters do not exist. Start training with randomly initialized parameters." );

            n_models_train = sr_train.n_model;
            n_models_test = sr_test.n_model;

            n_batches_train = int( max( n_models_train / self.minibatch_size, 1 ) );
            n_batches_test = int( max( n_models_test / self.minibatch_size, 1 ) );

            # loss確認時はバッチ数を少なめに設定
            n_batches_train_valid = int( max( n_batches_train / 5.0, 1 ) );
            n_batches_test_valid = int( max( n_batches_test / 5.0, 1 ) );

            print( "Number of training epochs : " + str( n_epoch ) );
            print( "Number of minibatches per epoch : " + str( n_batches_train ) );
            print( "Mini batch size : " + str( self.minibatch_size ) );


            for epoch in range( n_epoch ) :

                print( "Epoch: " + str( epoch ) );

                if( epoch % 1 == 0 ) : # 定期的にloss計算
                    # training setのloss
                    overallloss_total = 0.0;
                    oriptloss_total = 0.0;
                    voxelloss_total = 0.0;
                    mvimgloss_total = 0.0;
                    centerloss_total = 0.0;
                    for batch in range( n_batches_train_valid ) :
                        mb_oript, mb_voxel, mb_mvimg, mb_label, _ = self.generateMiniBatch( sr_train, Label_train );

                        result = self.sess.run( [ self.overall_loss, self.oript_loss, self.voxel_loss, self.mvimg_loss, self.center_loss ],
                                                  feed_dict={ self.oript_in : mb_oript, self.oript_gt : mb_oript,
                                                              self.voxel_in : mb_voxel, self.voxel_gt : mb_voxel,
                                                              self.mvimg_in : mb_mvimg, self.mvimg_gt : mb_mvimg,
                                                              self.flag_train : False, self.keep_prob : 1.0 } );

                        overallloss_total += result[ 0 ];
                        oriptloss_total += result[ 1 ];
                        voxelloss_total += result[ 2 ];
                        mvimgloss_total += result[ 3 ];
                        centerloss_total += result[ 4 ];


                    print( "overallloss_train: " + str( overallloss_total / n_batches_train_valid ) \
                           # + " oriptloss_train: " + str( oriptloss_total / n_batches_train_valid ) \
                           # + " voxelloss_train: " + str( voxelloss_total / n_batches_train_valid ) \
                           # + " mvimgloss_train: " + str( mvimgloss_total / n_batches_train_valid ) \
                           # + " centerloss_train: " + str( centerloss_total / n_batches_train_valid ) \
                         );


                    # test setのloss
                    overallloss_total = 0.0;
                    oriptloss_total = 0.0;
                    voxelloss_total = 0.0;
                    mvimgloss_total = 0.0;
                    centerloss_total = 0.0;
                    for batch in range( n_batches_test_valid ) :
                        mb_oript, mb_voxel, mb_mvimg, mb_label, _ = self.generateMiniBatch( sr_test, Label_test );

                        result = self.sess.run( [ self.overall_loss, self.oript_loss, self.voxel_loss, self.mvimg_loss, self.center_loss ],
                                                  feed_dict={ self.oript_in : mb_oript, self.oript_gt : mb_oript,
                                                              self.voxel_in : mb_voxel, self.voxel_gt : mb_voxel,
                                                              self.mvimg_in : mb_mvimg, self.mvimg_gt : mb_mvimg,
                                                              self.flag_train : False, self.keep_prob : 1.0 } );

                        overallloss_total += result[ 0 ];
                        oriptloss_total += result[ 1 ];
                        voxelloss_total += result[ 2 ];
                        mvimgloss_total += result[ 3 ];
                        centerloss_total += result[ 4 ];

                    print( "overallloss_test: " + str( overallloss_total / n_batches_test_valid ) \
                           # + " oriptloss_test: " + str( oriptloss_total / n_batches_test_valid ) \
                           # + " voxelloss_test: " + str( voxelloss_total / n_batches_test_valid ) \
                           # + " mvimgloss_test: " + str( mvimgloss_total / n_batches_test_valid ) \
                           # + " centerloss_test: " + str( centerloss_total / n_batches_test_valid ) \
                         );

                if( epoch % 5 == 0 ) : # 定期的に精度評価
                    print( "extracting features for evaluation..." );
                    feat_train = self.extractEmbeddedFeatures( sr_train );
                    feat_test = self.extractEmbeddedFeatures( sr_test );

                    if( cu.hasInfNan( feat_train[ 0 ] ) or cu.hasInfNan( feat_train[ 1 ] ) or cu.hasInfNan( feat_train[ 2 ] ) \
                        or cu.hasInfNan( feat_test[ 0 ] ) or cu.hasInfNan( feat_test[ 1 ] ) or cu.hasInfNan( feat_test[ 2 ] ) ) :
                        print( "!!!!!!!!!! error: Inf or Nan has been detected !!!!!!!!!");

                    print( "Combination by vector concatenation:" );
                    f_train = np.concatenate( feat_train, axis=1 );
                    f_test = np.concatenate( feat_test, axis=1 );
                    print( "evaluating retrieval accuracy of testing shapes..." );
                    map = self.evaluateRetrievalAccuracy( f_test, in_list_filename_test, in_label_filename_test );
                    print( "MAP: " + "{:.3f}".format( map ) );

                curr_batch = self.sess.run( self.batch );
                curr_lr = self.sess.run( self.learning_rate );
                curr_bndecay = self.sess.run( self.bn_decay );

                # print( "batch : " + str( curr_batch ) );
                # print( "lr : " + str( curr_lr ) );
                # print( "bndecay : " + str( curr_bndecay ) );

                # 学習
                start = time.time();
                for batch in range( n_batches_train ) :
                    mb_oript_gt, mb_voxel_gt, mb_mvimg_gt, mb_label, _ = self.generateMiniBatch( sr_train, Label_train );
                    mb_oript_in, mb_voxel_in, mb_mvimg_in = self.randomizeMiniBatch( mb_oript_gt, mb_voxel_gt, mb_mvimg_gt );

                    # Update network
                    self.sess.run( optim, feed_dict={ self.oript_in : mb_oript_in, self.oript_gt : mb_oript_gt,
                                                      self.voxel_in : mb_voxel_in, self.voxel_gt : mb_voxel_gt,
                                                      self.mvimg_in : mb_mvimg_in, self.mvimg_gt : mb_mvimg_gt,
                                                      self.flag_train : True, self.keep_prob : self.dropout_keep_prob } );
                print( "took " + "{:.3f}".format( time.time() - start ) + " sec" );

            self.save( n_epoch );


    #########################
    def randomizeMiniBatch( self, mb_oript, mb_voxel, mb_mvimg ) :
        minibatch_oript = [];
        minibatch_voxel = [];
        minibatch_mvimg = [];
        for i in range( self.minibatch_size ) :

                o_copy = copy.deepcopy( mb_oript[ i * self.point_n_point : (i+1) * self.point_n_point ] );
                o_copy = self.randomizeOript( o_copy );
                minibatch_oript.append( o_copy );

                v_copy = copy.deepcopy( mb_voxel[ i ] );
                v_copy = self.randomizeVoxel( v_copy );
                minibatch_voxel.append( v_copy );

                i_copy = copy.deepcopy( mb_mvimg[ i * self.render_n_view : (i+1) * self.render_n_view ] );
                i_copy = self.randomizeMvimg( i_copy );
                minibatch_mvimg.append( i_copy );

        minibatch_oript = np.array( minibatch_oript ).astype( np.float32 );
        minibatch_voxel = np.array( minibatch_voxel ).astype( np.float32 );
        minibatch_mvimg = np.array( minibatch_mvimg ).astype( np.float32 );
        minibatch_oript = np.reshape( minibatch_oript, [ -1, self.point_n_channel ] );
        minibatch_voxel = np.reshape( minibatch_voxel, [ -1, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, self.voxel_n_channel ] );
        minibatch_mvimg = np.reshape( minibatch_mvimg, [ -1, self.render_resolution, self.render_resolution, self.render_n_channel ] );

        return( minibatch_oript, minibatch_voxel, minibatch_mvimg );


    #########################
    def generateMiniBatch( self, sr, label ) :
        n_model = sr.n_model;
        minibatch_oript = [];
        minibatch_voxel = [];
        minibatch_mvimg = [];
        minibatch_label = [];
        minibatch_idx = [];

        for i in range( self.minibatch_size ) :
            # 1つの3Dモデルをランダムに選択
            shape_idx = np.random.randint( n_model );
            minibatch_idx.append( shape_idx );

            o = sr.oript[ shape_idx ];
            v = sr.voxel[ shape_idx ];
            i = sr.mvimg[ shape_idx ];

            minibatch_oript.append( o );
            minibatch_voxel.append( v );
            minibatch_mvimg.append( i );

            """
            # ラベル
            l = np.zeros( [ 1, self.n_class ], dtype="float32" );
            l[ 0, label[ shape_idx ] ] = 1.0;
            minibatch_label.append( l );
            """
            # 擬似ラベル
            l = np.zeros( [ 1, self.n_models_train ], dtype="float32" );
            l[ 0, shape_idx ] = 1.0;
            minibatch_label.append( l );


        minibatch_oript = np.array( minibatch_oript ).astype( np.float32 );
        minibatch_voxel = np.array( minibatch_voxel ).astype( np.float32 );
        minibatch_mvimg = np.array( minibatch_mvimg ).astype( np.float32 );
        minibatch_label = np.array( minibatch_label ).astype( np.float32 );
        minibatch_oript = np.reshape( minibatch_oript, [ -1, self.point_n_channel ] );
        minibatch_voxel = np.reshape( minibatch_voxel, [ -1, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution, self.voxel_n_channel ] );
        minibatch_mvimg = np.reshape( minibatch_mvimg, [ -1, self.render_resolution, self.render_resolution, self.render_n_channel ] );
        minibatch_label = np.squeeze( minibatch_label );

        return( minibatch_oript, minibatch_voxel, minibatch_mvimg, minibatch_label, minibatch_idx );


    #########################
    def get_learning_rate( self, batch ) :
        learning_rate = tf.train.exponential_decay(
                            BASE_LEARNING_RATE,  # Base learning rate.
                            batch * self.minibatch_size,  # Current index into the dataset.
                            DECAY_STEP,          # Decay step.
                            DECAY_RATE,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, MIN_LEARNING_RATE) # CLIP THE LEARNING RATE!
        return learning_rate

    #########################
    def get_bn_decay( self, batch ) :
        bn_momentum = tf.train.exponential_decay(
                          BN_INIT_DECAY,
                          batch * self.minibatch_size,
                          BN_DECAY_DECAY_STEP,
                          BN_DECAY_DECAY_RATE,
                          staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay


    #########################
    def extractEmbeddedFeatures( self, sr ) :
        X_oript = [];
        X_voxel = [];
        X_mvimg = [];
        n_models = sr.n_model;
        n_batch = int( math.ceil( float( n_models ) / float( self.minibatch_size ) ) );  # ミニバッチ数 (厳密なバッチ数)

        for i in range( n_batch ) :

            idx_from = i * self.minibatch_size;

            # 3Dモデル数がminibatch_sizeで割り切れない場合は，ミニバッチにダミーデータを追加する必要がある
            n_over = idx_from + self.minibatch_size - n_models;
            if( n_over <= 0 ) :
                n_real = self.minibatch_size;
                n_dummy = 0;
            else :
                n_real = n_models - idx_from;
                n_dummy = n_over;

            # ミニバッチを1つ作成
            mb_oript = [];
            mb_voxel = [];
            mb_mvimg = [];
            for j in range( n_real ) :
                mb_oript.append( sr.oript[ idx_from + j ] );
                mb_voxel.append( sr.voxel[ idx_from + j ] );
                mb_mvimg.append( sr.mvimg[ idx_from + j ] );
            for j in range( n_dummy ) :
                mb_oript.append( np.zeros( [ self.point_n_point, self.point_n_channel ], dtype="float32" ) );
                mb_voxel.append( np.zeros( [ self.voxel_resolution, self.voxel_resolution, self.voxel_resolution ], dtype="float32" ) );
                mb_mvimg.append( np.zeros( [ self.render_n_view, self.render_resolution, self.render_resolution, self.render_n_channel ], dtype="float32" ) );

            mb_oript = np.array( mb_oript ).astype( np.float32 );
            mb_oript = np.reshape( mb_oript, self.oript_in.get_shape().as_list() );
            mb_voxel = np.array( mb_voxel ).astype( np.float32 );
            mb_voxel = np.reshape( mb_voxel, self.voxel_in.get_shape().as_list() );
            mb_mvimg = np.array( mb_mvimg ).astype( np.float32 );
            mb_mvimg = np.reshape( mb_mvimg, self.mvimg_in.get_shape().as_list() );

            result = self.sess.run( [ self.oript_feat, self.voxel_feat, self.mvimg_feat ],
                                    feed_dict={ self.oript_in : mb_oript,
                                                self.voxel_in : mb_voxel,
                                                self.mvimg_in : mb_mvimg,
                                                self.keep_prob : 1.0, self.flag_train : False } );
            X_oript.append( result[ 0 ] );
            X_voxel.append( result[ 1 ] );
            X_mvimg.append( result[ 2 ] );

        X_oript = np.asarray( X_oript );
        X_voxel = np.asarray( X_voxel );
        X_mvimg = np.asarray( X_mvimg );
        X_oript = np.reshape( X_oript, [ n_batch * self.minibatch_size, -1 ] ); # ベクトル群に平坦化
        X_voxel = np.reshape( X_voxel, [ n_batch * self.minibatch_size, -1 ] ); # ベクトル群に平坦化
        X_mvimg = np.reshape( X_mvimg, [ n_batch * self.minibatch_size, -1 ] ); # ベクトル群に平坦化
        X_oript = X_oript[ 0:n_models, : ]; # ダミーを除外
        X_voxel = X_voxel[ 0:n_models, : ]; # ダミーを除外
        X_mvimg = X_mvimg[ 0:n_models, : ]; # ダミーを除外

        return X_oript, X_voxel, X_mvimg;


    #########################
    def evaluateRetrievalAccuracy( self, feat, in_list_filename, in_label_filename ) :
        nn, map = retrieval.retrieval( feat, in_list_filename, in_label_filename, "COS" );
        return map;

    #########################
    def randomizeOript( self, oript ) :

        n_rot = np.random.randint( 4 ); # 0,1,2,3のいずれか
        if( np.random.randint( 2 ) == 0 ) : # 0,1のいずれか
            do_flip = True;
        else :
            do_flip = False;

        ##### 上向き軸(z軸)周りの90度刻み回転 (X/Y -> Y/-X) #####
        if( do_rotate ) :
            for i in range( n_rot ) :
                oript[ :, 0 ] *= -1.0;
                x = np.copy( oript[ :, 0 ] );
                oript[ :, 0 ] = oript[ :, 1 ];
                oript[ :, 1 ] = x;

                # 法線も回転
                oript[ :, 3 ] *= -1.0;
                x = np.copy( oript[ :, 3 ] );
                oript[ :, 3 ] = oript[ :, 4 ];
                oript[ :, 4 ] = x;

        ##### 反転 #####
        if( do_flip ) :
            oript[ :, 0 ] *= -1.0;

        ##### ランダム平行移動 #####
        translate_max = 0.1;
        x = np.random.uniform( - translate_max, translate_max );
        y = np.random.uniform( - translate_max, translate_max );
        z = np.random.uniform( - translate_max, translate_max );
        oript[ :, 0 ] += x;
        oript[ :, 1 ] += y;
        oript[ :, 2 ] += z;

        ##### 位置と法線に加法性ノイズ #####
        noise = np.random.normal( 0.0, 0.01, ( self.point_n_point, self.point_n_channel ) );
        oript += noise;

        return oript;

    #########################
    def randomizeMvimg( self, mvimg ) :

        n_rot = np.random.randint( 4 ); # 0,1,2,3のいずれか
        if( np.random.randint( 2 ) == 0 ) : # 0,1のいずれか
            do_flip = True;
        else :
            do_flip = False;

        ##### 左右反転 #####
        if( do_flip ) :
            for i in range( self.render_n_view ) :
                mvimg[ i ] = np.flip( mvimg[ i ], 1 );

        ##### ランダム平行移動 #####
        def jitter( v, axis, insidx, delidx ) :
            v = np.insert( v, insidx, 1.0, axis=axis ); # 背景色は白
            v = np.delete( v, delidx, axis=axis );
            return v;

        offset_max = 8; # 移動量の最大値

        for i in range( self.render_n_view ) :

            # x軸方向の平行移動
            offset = np.random.randint( offset_max + 1 );
            direction = np.random.randint( 2 ); # 0,1のいずれか
            if( offset != 0 ) :
                if( direction == 0 ) : # 正の方向へ平行移動する場合
                    insidx = [ 0 ] * offset;
                    delidx = range( self.render_resolution, self.render_resolution + offset );
                else :                 # 負の方向へ平行移動する場合
                    insidx = [ self.render_resolution ] * offset;
                    delidx = range( 0, offset );

                mvimg[ i ] = jitter( mvimg[ i ], 0, insidx, delidx );

            # y軸方向の平行移動
            offset = np.random.randint( offset_max + 1 );
            direction = np.random.randint( 2 ); # 0,1のいずれか
            if( offset != 0 ) :
                if( direction == 0 ) : # 正の方向へ平行移動する場合
                    insidx = [ 0 ] * offset;
                    delidx = range( self.render_resolution, self.render_resolution + offset );
                else :                 # 負の方向へ平行移動する場合
                    insidx = [ self.render_resolution ] * offset;
                    delidx = range( 0, offset );

                mvimg[ i ] = jitter( mvimg[ i ], 1, insidx, delidx );

        return mvimg;


    #########################
    def randomizeVoxel( self, voxel ) :

        n_rot = np.random.randint( 4 ); # 0,1,2,3のいずれか
        if( np.random.randint( 2 ) == 0 ) : # 0,1のいずれか
            do_flip = True;
        else :
            do_flip = False;

        ##### 上向き軸(z軸)周りの90度刻み回転 (X/Y -> Y/-X) #####
        if( do_rotate ) :
            for i in range( n_rot ) :
                voxel = np.flip( voxel, 2 );
                voxel = np.swapaxes( voxel, 2, 0 );

        ##### 反転 #####
        if( do_flip ) :
            voxel = np.flip( voxel, 2 );

        ##### ランダム体積消去 #####
        erase_vol_max = 0.1; # 体積消去の最大割合
        erase_vol = np.random.rand() * erase_vol_max; # 今回の消去割合
        idx = np.where( voxel > 0.0 ); # 体積のある場所
        vol = voxel[ idx ] + np.random.rand( voxel[ idx ].shape[0] ); # 画素値を[1,2]の範囲のランダムな値にばらけさせる
        vol = np.where( vol > 1.0 + erase_vol, 1.0, 0.0 ); # 閾値処理
        voxel[ idx ] = vol;

        ##### ランダム平行移動 #####
        def jitter( v, axis, insidx, delidx ) :
            v = np.insert( v, insidx, 0, axis=axis );
            v = np.delete( v, delidx, axis=axis );
            return v;

        offset_max = 4; # 移動量の最大値

        # x軸方向の平行移動
        offset = np.random.randint( offset_max + 1 );
        direction = np.random.randint( 2 ); # 0,1のいずれか
        if( offset != 0 ) :
            if( direction == 0 ) : # 正の方向へ平行移動する場合
                insidx = [ 0 ] * offset;
                delidx = range( self.voxel_resolution, self.voxel_resolution + offset );
            else :                 # 負の方向へ平行移動する場合
                insidx = [ self.voxel_resolution ] * offset;
                delidx = range( 0, offset );

            voxel = jitter( voxel, 0, insidx, delidx );

        # y軸方向の平行移動
        offset = np.random.randint( offset_max + 1 );
        direction = np.random.randint( 2 ); # 0,1のいずれか
        if( offset != 0 ) :
            if( direction == 0 ) : # 正の方向へ平行移動する場合
                insidx = [ 0 ] * offset;
                delidx = range( self.voxel_resolution, self.voxel_resolution + offset );
            else :                 # 負の方向へ平行移動する場合
                insidx = [ self.voxel_resolution ] * offset;
                delidx = range( 0, offset );

            voxel = jitter( voxel, 1, insidx, delidx );

        # z軸方向の平行移動
        offset = np.random.randint( offset_max + 1 );
        direction = np.random.randint( 2 ); # 0,1のいずれか
        if( offset != 0 ) :
            if( direction == 0 ) : # 正の方向へ平行移動する場合
                insidx = [ 0 ] * offset;
                delidx = range( self.voxel_resolution, self.voxel_resolution + offset );
            else :                 # 負の方向へ平行移動する場合
                insidx = [ self.voxel_resolution ] * offset;
                delidx = range( 0, offset );

            voxel = jitter( voxel, 2, insidx, delidx );

        return voxel;


    def save( self, step ) :
        print(" [*] Saving checkpoint to " + self.checkpoint_dir );
        if not os.path.exists( self.checkpoint_dir ) :
            os.makedirs( self.checkpoint_dir );
        self.saver.save( self.sess, os.path.join( self.checkpoint_dir, "shapeAE" ), global_step=step );

    def load( self ) :
        print(" [*] Reading checkpoint from " + self.checkpoint_dir );
        ckpt = tf.train.get_checkpoint_state( self.checkpoint_dir );
        if ckpt and ckpt.model_checkpoint_path :
            ckpt_name = os.path.basename( ckpt.model_checkpoint_path );
            self.saver.restore( self.sess, os.path.join( self.checkpoint_dir, ckpt_name ) );
            return True;
        else:
            return False;
