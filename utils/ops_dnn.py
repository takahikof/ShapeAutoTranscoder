# -*- coding: utf-8 -*-
import math
import sys
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

mode_set_to_set_dist = "CHAMFER";


#################################
def print_shape( x ) :
    shape = x.get_shape().as_list();
    print( shape );

#################################
def sqrt( x ) : # sqrt(0)の微分値がinfになるため,sqrt(0)を回避
    return tf.sqrt( tf.clip_by_value( x, 1e-8, 1e8 ) );

def tanh( x ) :
    return tf.nn.tanh( x );

def relu( x ) :
    return tf.nn.relu( x );

def elu( x ) :
    return tf.nn.elu( x );

def crelu( x ) :
    x1 = tf.nn.relu( x );
    x2 = tf.nn.relu( -x );
    tensor_dim = len( x.get_shape().as_list() );
    if( tensor_dim == 4 ) :
        channel_dim = 3;
    else :
        channel_dim = 1;
    x = tf.concat( [ x1, x2 ], channel_dim );    
    return x;

def max_pool( x ) :
    return tf.nn.max_pool( x, [ 1, 2, 2, 1 ], [ 1, 2, 2, 1 ], 'SAME' );
    
def dropout( x, keep_prob ) :
    return tf.nn.dropout( x, keep_prob );
    
def bn( x, is_training, scope ) :
    with tf.variable_scope( scope ) :
        y = tf.contrib.layers.batch_norm( x, center=True, scale=True, is_training=is_training );
        return y;
    
def l2normalize( x ) :
    y = tf.nn.l2_normalize( x + 1e-8, 1 );
    return y;

def l1normalize( x ) :
    norm = tf.norm( x, ord=1, axis=1, keepdims=True );
    y = tf.div( x, norm );
    return y;

def ksparse( x, k ) :
    shape = x.get_shape().as_list();
    values, indices = tf.nn.top_k( x, k = k );
    values = values[ :, k - 1 ];
    values = tf.reshape( values, [ -1, 1 ] );
    values = tf.tile( values, [ 1, shape[1] ] );
    y = tf.where( tf.less( x , values ), tf.zeros_like( x ), x );
    return y;

def linear( x, output_size, scope ) :
    shape = x.get_shape().as_list();
    with tf.variable_scope( scope ) :
        matrix = tf.get_variable( "matrix", [shape[1], output_size], tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        bias = tf.get_variable( "bias", [output_size], initializer=tf.constant_initializer( 0.0  ))
        return( tf.matmul( x, matrix ) + bias );


def conv2d( x, output_dim, kernel_size, stride, scope ) :
    with tf.variable_scope( scope ) :
        w = tf.get_variable( "conv", [ kernel_size, kernel_size, x.get_shape()[-1], output_dim ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        conv = tf.nn.conv2d( x, w, strides=[1, stride, stride, 1], padding='SAME');
        biases = tf.get_variable( 'biases', [ output_dim ], initializer=tf.constant_initializer(0.0) );        
        shape = conv.get_shape().as_list();
        conv = tf.reshape( tf.nn.bias_add( conv, biases ), [ -1, shape[1], shape[2], shape[3] ]  );        
        return conv;

def conv3d( x, output_dim, kernel_size, stride, scope ) :
    with tf.variable_scope( scope ) :
        w = tf.get_variable( "conv", [ kernel_size, kernel_size, kernel_size, x.get_shape()[-1], output_dim ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        conv = tf.nn.conv3d( x, w, strides=[1, stride, stride, stride, 1], padding='SAME');
        biases = tf.get_variable( 'biases', [ output_dim ], initializer=tf.constant_initializer(0.0) );        
        shape = conv.get_shape().as_list();
        conv = tf.reshape( tf.nn.bias_add( conv, biases ), [ -1, shape[1], shape[2], shape[3], shape[4] ]  );        
        return conv;

def deconv2d( x, output_shape, kernel_size, stride, scope ) :
    with tf.variable_scope( scope ) :
        w = tf.get_variable( 'deconv', [ kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1] ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        deconv = tf.nn.conv2d_transpose( x, w, output_shape=output_shape, strides=[1, stride, stride, 1] );
        biases = tf.get_variable( 'biases', [ output_shape[-1] ], initializer=tf.constant_initializer(0.0) );
        deconv = tf.reshape(tf.nn.bias_add( deconv, biases ), deconv.get_shape() );
        return deconv;

def deconv3d( x, output_shape, kernel_size, stride, scope ) :
    with tf.variable_scope( scope ) :
        w = tf.get_variable( 'deconv', [ kernel_size, kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1] ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        deconv = tf.nn.conv3d_transpose( x, w, output_shape=output_shape, strides=[1, stride, stride, stride, 1] );
        biases = tf.get_variable( 'biases', [ output_shape[-1] ], initializer=tf.constant_initializer(0.0) );
        deconv = tf.reshape(tf.nn.bias_add( deconv, biases ), deconv.get_shape() );
        return deconv;







def aggregate( x, mode, minibatch_size, n_lf_permodel ) :
    # xは[ minibatch_size * n_lf_permodel, n_dim_in ]の2D tensor
    shape = x.get_shape().as_list();
    n_dim_in = shape[ 1 ];
    
    segment_ids = tf.range( minibatch_size );
    segment_ids = tf.reshape( segment_ids, [ -1, 1 ] );
    segment_ids = tf.tile( segment_ids, [ 1, n_lf_permodel ] );
    segment_ids = tf.squeeze( tf.reshape( segment_ids, [ -1, 1 ] ) );

    if( mode == "A" ) : # average pooling
        y = tf.segment_mean( x, segment_ids );
        y = tf.reshape( y, [ minibatch_size, n_dim_in ] ); # segment_mean/segment_maxするとtensorのshapeが不定になるので明らかにしておく
    elif( mode == "M" ) : # max pooling
        y = tf.segment_max( x, segment_ids );
        y = tf.reshape( y, [ minibatch_size, n_dim_in ] ); # segment_mean/segment_maxするとtensorのshapeが不定になるので明らかにしておく

    elif( mode == "B" ) : # bi-linear pooling
        # TODO
        pass;        

    return y;


def pairwise_distances( A, B, metric ) :    

    if( metric == "L2" ) : # L2 distance
        sqnormA = tf.reduce_sum( tf.multiply( A, A ), 1, keepdims=True );
        sqnormB = tf.reduce_sum( tf.multiply( B, B ), 1, keepdims=True );    
        distmat = sqnormA - 2.0 * tf.matmul( A, tf.transpose( B ) ) + tf.transpose( sqnormB ); # squared euclidean distances
        distmat = sqrt( distmat ); # euclidean distances
    elif( metric == "L1" ) : # L1 distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = tf.reduce_sum( tf.abs( expanded_a - expanded_b ), 2 );    
    elif( metric == "L05" ) : # L0.5 distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = tf.square( tf.reduce_sum( sqrt( tf.abs( expanded_a - expanded_b ) ), 2 ) );
    elif( metric == "COS" ) : # Cosine distance
        normA = sqrt( tf.reduce_sum( tf.multiply( A, A ), 1, keepdims=True ) );
        normB = sqrt( tf.reduce_sum( tf.multiply( B, B ), 1, keepdims=True ) );    
        norms = normA * tf.transpose( normB );
        distmat = tf.matmul( A, tf.transpose( B ) ) / norms;
        distmat = 1.0 - ( 1.0 + distmat ) / 2.0;
    elif( metric == "CHI" ) : # Chi-squared distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = sqrt( tf.reduce_sum( tf.square( expanded_a - expanded_b ) / ( expanded_a + expanded_b + 1e-8 ), 2 ) );        
    elif( metric == "CAM" ) : # Camberra distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = tf.reduce_sum( tf.abs( expanded_a - expanded_b ) / ( expanded_a + expanded_b + 1e-8 ), 2 );                
    else :
        print( "invalid metric : " + metric );    
        quit();
        
    return distmat;
        
def chamfer_distance( A, B, metric ) :    
    D = pairwise_distances( A, B, metric );
    D1 = tf.reduce_min( D, 1 );    
    D1 = tf.reduce_mean( D1 );
    D2 = tf.reduce_min( D, 0 );
    D2 = tf.reduce_mean( D2 );
    return( D1 + D2 );
    
def hausdorff_distance( A, B, metric ) :    
    D = pairwise_distances( A, B, metric );
    D1 = tf.reduce_min( D, 1 );    
    D1 = tf.reduce_max( D1 );
    D2 = tf.reduce_min( D, 0 );
    D2 = tf.reduce_max( D2 );
    return( tf.maximum( D1, D2 ) );



def get_set_reconstruction_loss( s1, s2, minibatch_size, n_lf_permodel, metric ) :
    # s1, s2は[ minibatch_size * n_lf_permodel, 局所特徴次元数 ]の2D tensor

    # s1とs2で局所特徴数が異なるかも知れないので計算 (EMDは同じである必要がある)
    n_lf_permodel_s1 = int( s1.get_shape().as_list()[ 0 ] / minibatch_size );
    n_lf_permodel_s2 = int( s2.get_shape().as_list()[ 0 ] / minibatch_size );
    
    print( "set_to_set distance : " + mode_set_to_set_dist + " with " + metric );

    loss = 0;
    for i in range( minibatch_size ) :
        a = s1[ i*n_lf_permodel_s1 : (i+1)*n_lf_permodel_s1 ];
        b = s2[ i*n_lf_permodel_s2 : (i+1)*n_lf_permodel_s2 ];
        
        if( mode_set_to_set_dist == "CHAMFER" ) :
            loss += chamfer_distance( a, b, metric );
        elif( mode_set_to_set_dist == "HAUSDORFF" ) :
            loss += hausdorff_distance( a, b, metric );
        elif( mode_set_to_set_dist == "EARTHMOVERS" ) :    
            loss += earthmovers_distance( a, b, metric );
        else :
            print( "Invalid set-to-set distance mode : " + mode_set_to_set_dist );
            quit();
            
    return( loss / minibatch_size );


def get_persample_reconstruction_loss( s1, s2 ) :
    # s1, s2は[ minibatch_size * n_lf_permodel, 局所特徴次元数 ]の2D tensor
    dists = tf.sqrt( tf.reduce_sum( tf.square( s1 - s2 ), reduction_indices=[1] ) + 1e-8 );
    return( tf.reduce_mean( dists ) );                

    
def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)

def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


# https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py
def arcface_loss(embedding, labels, out_num, w_init=None, s=5., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, nclass)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = labels
        # mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output_train = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        # output_test = tf.add( s_cos_t, cos_mt_temp, name='arcface_loss_output2')
        output_test = s_cos_t
    return output_train, output_test


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, nclass)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = labels
        # mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output  
    
    
def res_block( x, out_dim, flag_train, bn_decay, ID ) :

    in_dim = x.get_shape().as_list()[-1];
    if( in_dim == out_dim ) :
        shortcut = x;
    else :
        shortcut = conv2d( x, out_dim, 1, 1, ID+"_conv1" );

    """    
    h = conv2d( x, out_dim, 3, 1, ID+"_conv2" );
    h = batch_norm_for_conv2d( h, flag_train, bn_decay, ID+"_bn1" );
    h = relu( h );
    h = conv2d( h, out_dim, 3, 1, ID+"_conv3" );
    h = batch_norm_for_conv2d( h, flag_train, bn_decay, ID+"_bn2" );
    h = relu( h + shortcut );
    """    
    
    h = batch_norm_for_conv2d( x, flag_train, bn_decay, ID+"_bn1" );
    h = conv2d( h, out_dim, 3, 1, ID+"_conv2" );
    h = batch_norm_for_conv2d( h, flag_train, bn_decay, ID+"_bn2" );
    h = relu( h );
    h = conv2d( h, out_dim, 3, 1, ID+"_conv3" );
    h = batch_norm_for_conv2d( h, flag_train, bn_decay, ID+"_bn3" );
    h = h + shortcut;

    return h;


def res_block_3d( x, out_dim, flag_train, bn_decay, ID ) :

    in_dim = x.get_shape().as_list()[-1];
    if( in_dim == out_dim ) :
        shortcut = x;
    else :
        shortcut = conv3d( x, out_dim, 1, 1, ID+"_conv1" );

    h = batch_norm_for_conv3d( x, flag_train, bn_decay, ID+"_bn1" );
    h = conv3d( h, out_dim, 3, 1, ID+"_conv2" );
    h = batch_norm_for_conv3d( h, flag_train, bn_decay, ID+"_bn2" );
    h = relu( h );
    h = conv3d( h, out_dim, 3, 1, ID+"_conv3" );
    h = batch_norm_for_conv3d( h, flag_train, bn_decay, ID+"_bn3" );
    h = h + shortcut;

    return h;


def conv_for_pointnet( x, output_dim, scope ) :
    kernel_size = 1;
    stride = 1;
    padding = 'VALID';
    with tf.variable_scope( scope ) :
        w = tf.get_variable( "conv", [ kernel_size, kernel_size, x.get_shape()[-1], output_dim ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        conv = tf.nn.conv2d( x, w, strides=[1, stride, stride, 1], padding=padding );
        
        biases = tf.get_variable( 'biases', [ output_dim ], initializer=tf.constant_initializer(0.0) );        
        shape = conv.get_shape().as_list();
        conv = tf.nn.bias_add( conv, biases );
                
        return conv;


def tf_covariance( x ) : # 共分散行列を計算
    mean_x = tf.reduce_mean( x, axis=0, keepdims=True );
    x = x - mean_x;
    cov = tf.matmul( tf.transpose( x ), x ) / ( tf.cast( tf.shape(x)[0], tf.float32 ) - 1.0 );    
    return cov;

def tf_uppertriangle( x ) : # 上三角成分だけ抜粋 (対角成分を含む)    
    mask = tf.cast( tf.matrix_band_part( tf.ones_like( x ), 0, -1 ), dtype=tf.bool );
    tri = tf.boolean_mask( x, mask );
    tri = tf.reshape( tri, [ 1, -1 ] ); # ベクトル化して返す
    return tri;
    
def tf_nearest_neighbor( x, k, minibatch_size, n_pt_permodel ) : # k近傍探索
    # xは[ minibatch_size * n_pt_permodel, 特徴次元数 ]の2D tensor

    nn_idx = []; # 近傍のインデックス情報を返す
    for i in range( minibatch_size ) :
        f = x[ i * n_pt_permodel : (i+1) * n_pt_permodel ];   # データiの特徴だけ抜粋 
        dist = pairwise_distances( f, f, "L2" );
        _, idx = tf.nn.top_k( -1.0 * dist, k = k );
        nn_idx.append( idx );
    return nn_idx;


def tf_local_covariance( x, nn_idx, minibatch_size, n_pt_permodel ) : # 近傍のインデックス情報に基づいて局所領域の共分散を計算
    # xは[ minibatch_size * n_pt_permodel, 特徴次元数 ]の2D tensor
    # nn_idxに各データの近傍情報が格納されている

    ndim = x.get_shape().as_list()[ 1 ];  # 特徴次元数
    n_elem = int( ( ndim * ( ndim + 1 ) ) / 2 ); # 共分散行列の上三角成分+対角成分の数
    
    local_cov = [];
    for i in range( minibatch_size ) :
        f = x[ i * n_pt_permodel : (i+1) * n_pt_permodel ];   # データiの特徴だけ抜粋 
        nn = tf.gather( f, nn_idx[ i ] );
        mean = tf.reduce_mean( nn, axis=1, keepdims=True );
        nn = nn - mean;
        knn = nn.get_shape().as_list()[ 1 ];
        cov = tf.matmul( tf.transpose( nn, perm=[0,2,1] ), nn ) / ( tf.cast( knn, tf.float32 ) - 1.0 );
        local_cov.append( tf_uppertriangle( cov ) ); 
        
    local_cov = tf.concat( local_cov, 0 );
    local_cov = tf.reshape( local_cov, [ minibatch_size * n_pt_permodel, n_elem ] );
    return local_cov;

def tf_local_pooling( x, nn_idx, pool_mode, minibatch_size, n_pt_permodel ) : # 近傍のインデックス情報に基づいて局所領域のプーリングを計算
    # xは[ minibatch_size * n_pt_permodel, 特徴次元数 ]の2D tensor
    # nn_idxに各データの近傍情報が格納されている

    local_pool = [];
    for i in range( minibatch_size ) :
        f = x[ i * n_pt_permodel : (i+1) * n_pt_permodel ];   # データiの特徴だけ抜粋 
        nn = tf.gather( f, nn_idx[ i ] );
        
        if( pool_mode == "A" ) : # average pooling
            pool = tf.reduce_mean( nn, axis=1 );
        elif( pool_mode == "M" ) : # max pooling
            pool = tf.reduce_max( nn, axis=1 );
        else :
            print( "error: invalid pooling mode : " + pool_mode );
            quit();
            
        local_pool.append( pool );

    local_pool = tf.concat( local_pool, 0 );
    return local_pool;
 
 
 
#===========================
# kernelの計算
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    
    # sigma = 0.3;
    sigma = 0.1;
    
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast( sigma, tf.float32))
#===========================

#===========================
# mmdの計算
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
#===========================
