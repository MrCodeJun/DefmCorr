import tensorflow as tf
import pointfly as pf
# import numpy as np


def input_transform_net(points, is_training, name, dim=3, reused=None):
    """ Input (XYZ) Transform Net, input is BxNx3 points data
        Return:
            Transformation matrix of size 3xK """
    pts_num = points.get_shape()[1].value
    batch_size = points.get_shape()[0].value
    net = pf.dense(points, 64, name + 'mlp1', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = pf.dense(net, 128, name + 'mlp2', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = pf.dense(net, 1024, name + 'mlp3', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = tf.expand_dims(net, axis=-1, name=name+'expand_dim')
    net = pf.max_pool2d(net, [pts_num, 1], stride=[1, 1], name=name + 'pooling')
    net = tf.reshape(net, [batch_size, 1024])
    net = pf.dense(net, 128, name + 'mlp4', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = pf.dense(net, 64, name + 'mlp5', is_training, activation_fn=tf.nn.relu, reuse=reused)

    with tf.variable_scope(name + 'xyz_transform', reuse=reused):
        assert(dim == 3)
        weights = tf.get_variable('weights', [64, 3*dim],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*dim],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, dim])
    return transform


def input_qtransform_net(points, is_training, name, reused=None):

    batch_size = points.get_shape()[0].value
    point_num = points.get_shape()[1].value

    net = pf.dense(points, 64, name + 'mlp1', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = pf.dense(net, 128, name + 'mlp2', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = pf.dense(net, 1024, name + 'mlp3', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = tf.expand_dims(net, axis=-1, name=name + 'expand_dim')
    net = pf.max_pool2d(net, [point_num, 1], stride=[1, 1], name=name+'pooling')
    net = tf.reshape(net, [batch_size, 1, 1024])
    net = pf.dense(net, 128, name + 'mlp4', is_training, activation_fn=tf.nn.relu, reuse=reused)
    net = pf.dense(net, 64, name + 'mlp5', is_training, activation_fn=tf.nn.relu, reuse=reused)

    with tf.variable_scope(name+'q_transform', reuse=reused):
        weights = tf.get_variable('weights', [64, 4], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [4], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    biases += tf.constant([1, 0, 0, 0], dtype=tf.float32)
    weights = tf.expand_dims(weights, 0)
    weights = tf.tile(weights, [batch_size, 1, 1])
    transform = tf.matmul(net, weights)
    # [batch_size, 1, 4] each shape has a trans matrix

    transform = tf.nn.bias_add(transform, biases)
    qtransform = batch_quat_to_rotmat(transform)

    return qtransform


def batch_quat_to_rotmat(transform):
    batch_size = transform.get_shape()[0].value
    s = tf.reduce_sum(tf.pow(transform, 2), -1)
    a = tf.constant(2.0)
    s = a / s
    # h = [batch_size, 4, 4]:
    h = tf.matmul(transform, transform, transpose_a=True)

    rotation_list = []
    rotation_list.append(tf.subtract(1.0, tf.multiply(s, tf.expand_dims(tf.add(h[:, 2, 2], h[:, 3, 3]), -1))))
    rotation_list.append(tf.multiply(s, tf.expand_dims(tf.subtract(h[:, 1, 2], h[:, 3, 0]), -1)))
    rotation_list.append(tf.multiply(s, tf.expand_dims(tf.add(h[:, 1, 3], h[:, 2, 0]), -1)))

    rotation_list.append(tf.multiply(s, tf.expand_dims(tf.add(h[:, 1, 2], h[:,3, 0]), -1)))
    rotation_list.append(tf.subtract(1.0, tf.multiply(s, tf.expand_dims(tf.add(h[:, 1, 1], h[:, 3, 3]), -1))))
    rotation_list.append(tf.multiply(s, tf.expand_dims(tf.subtract(h[:, 2, 3], h[:, 1, 0]), -1)))

    rotation_list.append(tf.multiply(s, tf.expand_dims(tf.subtract(h[:, 1, 3], h[:, 2, 0]), -1)))
    rotation_list.append(tf.multiply(s, tf.expand_dims(tf.add(h[:, 2, 3], h[:, 1, 0]), -1)))
    rotation_list.append(tf.subtract(1.0, tf.multiply(s, tf.expand_dims(tf.add(h[:, 1, 1], h[:, 2, 2]), -1))))

    out = tf.reshape(tf.stack(rotation_list, axis=-1), [batch_size, 3, 3])
    out = tf.transpose(out, perm=[0, 2, 1])

    return out


def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist


def points_laplace_coord(points, K=8, D=1):
    if D == 1:
        _, indices = pf.knn_indices_general(points, points, K, False)
    else:
        _, indices_dilated = pf.knn_indices_general(points, points, K*D, True)
        indices = indices_dilated[:, :, ::D, :]
    # calculate laplacian coordinate:
    nn_points = tf.gather_nd(points, indices, name = 'nn_pts')
    points_tile = tf.expand_dims(points, axis=2, name = 'points_expand')
    points_tile = tf.tile(points_tile, [1, 1, K, 1])
    mean_coords = tf.reduce_mean(tf.subtract(points_tile, nn_points), axis=-2)
    lap_coords = tf.subtract(points, mean_coords)

    return lap_coords


class DeformNet():
    def __init__(self, is_training, batch_size, point_num, dense_weight, is_trans, is_key):
        self.points = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3), name='input')
        self.point_labels = tf.placeholder(tf.int32, shape=(batch_size, point_num), name='input_point_labels')
        self.category_labels = tf.placeholder(tf.int32, shape=(batch_size), name='category_label')
        self.is_training = is_training
        if is_key:
            self.key_list_points = tf.placeholder(tf.int32, shape=(batch_size, None, 1), name='point_key_list')
            self.key_list_temp = tf.placeholder(tf.int32, shape=(batch_size, None, 1), name='temp_key_list')

        self.temp_point_labels = tf.placeholder(tf.float32, shape=(batch_size, point_num), name='temp_point_labels')
        self.temp_points = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3), name='temp_points')

        self.is_trans = is_trans
        if is_trans is not 'no_trans':
            self.points_feats, self.global_fts, self.transform_matrix, self.points_t = self.point_feat_extract(
                                                                                                self.points, 'netE',
                                                                                                self.is_training,
                                                                                                self.is_trans)
            self.transform_matrix_transpose = tf.transpose(self.transform_matrix, perm=(0, 2, 1), name='transpose')
        else:
            self.points_feats, self.global_fts, self.points_t = self.point_feat_extract(self.points, 'netE',
                                                                                       self.is_training, self.is_trans)

        self.global_fts_tile = tf.tile(tf.expand_dims(self.global_fts, axis=1), [1, point_num, 1])
        self.feats_concat_temp = tf.concat([self.temp_points, self.global_fts_tile], axis=-1, name='temp_and_feats')
        self.points_deform = self.points_deform(self.feats_concat_temp, 'netD', self.is_training)

        self.category_pred = self.points_category(self.global_fts, 16, self.is_training, 'netCategory')


        if is_trans is not 'no_trans':
            self.points_deform_transpose = tf.matmul(self.points_deform, self.transform_matrix_transpose)
            self.shape_loss = self.shape_loss(self.points, self.points_deform_transpose, dense_weight, k_n=8)
        else:
            self.shape_loss = self.shape_loss(self.points_t, self.points_deform, dense_weight, k_n=8)

        if is_key is True:
            self.key_loss = self.key_points_loss(self.points_t, self.points_deform,
                                                 self.key_list_points, self.key_list_temp)

        self.category_loss = self.category_loss(self.category_pred, self.category_labels)


    def point_feat_extract(self, points, name, is_training, is_trans, is_reused=None):
        pts_num = points.get_shape()[1].value
        if is_trans == 'q_trans':
            transform_matrix = input_qtransform_net(points, is_training, name+'rotation')
            points_t = tf.matmul(points, transform_matrix)
        else:
            if is_trans == 'trans':
                transform_matrix = input_transform_net(points, is_training, name+'rotation')
                points_t = tf.matmul(points, transform_matrix)
            else:
                points_t = points

        net = pf.dense(points_t, 64, name+'mlp1', is_training, reuse=is_reused, activation_fn=tf.nn.relu)
        net = pf.dense(net, 128, name+'mlp2', is_training, reuse=is_reused, activation_fn=tf.nn.relu)
        net = pf.dense(net, 256, name+'mlp3', is_training, reuse=is_reused, activation_fn=tf.nn.relu)
        net = pf.dense(net, 1024, name+'mlp4', is_training, reuse=is_reused, activation_fn=tf.nn.relu)

        pfts = net

        net = tf.expand_dims(net, axis=-1, name=name+'extend_input')
        net = pf.max_pool2d(net, [pts_num, 1], stride=[1, 1], name=name+'pooling')

        net_tile = tf.tile(tf.squeeze(net, axis=-1), [1, pts_num, 1])
        concat_pfts = tf.concat([pfts, net_tile], axis=-1)

        if is_trans is not 'no_trans':
            return concat_pfts, tf.squeeze(net, axis=[1, 3]), transform_matrix, points_t
        else:
            return concat_pfts, tf.squeeze(net, axis=[1, 3]), points_t


    def points_deform(self, feats, name, is_training, is_reused=None):
        bottleneck_size = feats.get_shape()[-1].value
        feats = pf.dense(feats, bottleneck_size, name+'mlp1', is_training,
                         reuse=is_reused, activation_fn=tf.nn.relu)
        feats = pf.dense(feats, bottleneck_size//2, name+'mlp2', is_training,
                         reuse=is_reused, activation_fn=tf.nn.relu)
        feats = pf.dense(feats, bottleneck_size//4, name+'mlp3', is_training,
                         reuse=is_reused, activation_fn=tf.nn.relu)
        feats = pf.dense(feats, 3, name+'mlp4', is_training, reuse=is_reused, activation_fn=tf.nn.tanh)

        points_deform = 2*feats   # why ？？？
        return points_deform


    def points_category(self, shape_fts, category_num, is_training, name, is_resused=None):
        net = pf.dense(shape_fts, 512, name+'mlp1', is_training,
                       reuse=is_resused, activation_fn=tf.nn.relu)
        net = tf.layers.dropout(net, 0.3, training= is_training,name= name+'dp1')
        net = pf.dense(net, 256, name+'mlp2', is_training,
                       reuse=is_resused, activation_fn=tf.nn.relu)
        net = tf.layers.dropout(net, 0.3, training= is_training,name= name+'dp2')
        net = pf.dense(net, category_num, name+'mlp3', is_training,
                       reuse=is_resused, activation_fn=None)
        return net


    def shape_loss(self, point_a, point_b, densityWeight, k_n=8):
        # calculate shape loss (chamfer loss)
        square_dist = pairwise_l2_norm2_batch(point_a, point_b)
        dist = tf.sqrt(square_dist)
        min_row = tf.reduce_min(dist, axis=2)
        min_col = tf.reduce_min(dist, axis=1)
        shape_loss = tf.reduce_mean(min_row) + tf.reduce_mean(min_col)

        # calculate density loss
        square_dist2 = pairwise_l2_norm2_batch(point_a, point_a)
        dist2 = tf.sqrt(square_dist2)
        knndis = tf.nn.top_k(tf.negative(dist), k=k_n)
        knndis2 = tf.nn.top_k(tf.negative(dist2), k=k_n)
        densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

        data_loss = shape_loss + densityLoss * densityWeight

        return data_loss


    def key_points_loss(self, point_a, point_b, key_list_a, key_list_b):
        #  key_list =[batch_size,key_points_num,1]
        #  maybe need to consider the orders of the two key points set ?????

        key_points_a = tf.gather(point_a, key_list_a, name='key_points_a')
        key_points_b = tf.gather(point_b, key_list_b, name='key_points_b')

        l2 = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(key_points_a, key_points_b), 2), -1))
        key_loss = tf.reduce_mean(l2, axis=[0, 1])

        return key_loss


    def category_loss(self, pred, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
        return classify_loss


    def laplacian_loss(self, points_s, points_t):
        lap1 = points_laplace_coord(points_s)
        lap2 = points_laplace_coord(points_t)
        laplacian_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), axis=-1))
        move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(points_t, points_s)), axis=-1))

        return  laplacian_loss+move_loss

















