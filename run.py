import tensorflow as tf
import os
import sys
import argparse
import importlib
import numpy as np
import h5py
import pointfly as pf
from mayavi import mlab
import sample_points as sp

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, 'Models'))

# Default Parameters Setting
parser = argparse.ArgumentParser()
parser.add_argument("--densityWeight", type=float, default=1.0, help="density weight [default: 1.0]")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='deform_net_with_seg', help='Model name: deform_net [default: deform_net]')
parser.add_argument('--log', default='log_deform_with_seg2', help='Log dir [default: log]')
parser.add_argument('--point_num', type=int, default=2048, help='Do not set the argument')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run  [default: 200]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--decay_step', type=int, default=50000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--train_data', default='../../Data/benchmark_v0_deform',
                    help= 'Training data dir [default: ../Data]')
parser.add_argument('--is_visual', default='True', help='Visualize the data [defaul: False]')


FLAGS = parser.parse_args()
DENSE_WEIGHT= FLAGS.densityWeight
LOG_DIR = FLAGS.log
GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
POINT_NUM = FLAGS.point_num
BASE_LEARNING_RATE = FLAGS.learning_rate
OPTIMIZER = FLAGS.optimizer
MOMENTUM = FLAGS.momentum
DECAY_RATE = FLAGS.decay_rate
DECAY_STEP = FLAGS.decay_step
MAX_EPOCH = FLAGS.epoch
DATA_DIR = FLAGS.train_data
IS_SHOW = FLAGS.is_visual

IS_TRANS = 'q_trans'
IS_KEY = 'False'
IS_STEP2 = 'False'

MODEL = importlib.import_module(FLAGS.model)  # import network module
LOG_DIR = os.path.join('../Logs', LOG_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % ('Models/'+FLAGS.model+'.py', LOG_DIR))
os.system('cp run.py %s' % LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())
            batch = tf.Variable(0, trainable=False)

            network = MODEL.DeformNet(is_training_pl, BATCH_SIZE, POINT_NUM, DENSE_WEIGHT, IS_TRANS, IS_KEY)

            if IS_KEY is True:
                shape_loss = network.shape_loss
                key_loss = network.key_points_loss
                category_loss = network.category_loss
                seg_loss = network.real_seg_loss

                total_loss = 0.4*shape_loss + 0.3*key_loss + 0.3*category_loss
            else:
                category_loss = network.category_loss
                shape_loss = network.shape_loss
                seg_loss = network.real_seg_loss
                fake_seg_loss = network.fake_seg_loss
                total_loss = 0.4*shape_loss + 0.3*category_loss + 0.3*seg_loss

            reg_loss = 0.00001*tf.losses.get_regularization_loss()

            tf.summary.scalar('loss/loss_total', total_loss, collections=['train'])
            tf.summary.scalar('loss/shape_loss', shape_loss, collections=['train'])
            tf.summary.scalar('loss/category_loss', category_loss, collections=['train'])
            tf.summary.scalar('loss/seg_loss', seg_loss, collections=['train'])
            tf.summary.scalar('loss/fake_seg_loss', fake_seg_loss, collections=['train'])

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate, collections=['train'])
            if OPTIMIZER =='momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER =='adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                train_op = optimizer.minimize(total_loss + reg_loss+0.5*fake_seg_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged_train = tf.summary.merge_all('train')
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # show all the trainable variables:
        # variable_names = [v.name for v in tf.trainable_variables()]
        # print(variable_names)

        ops = {'input_shapes': network.points,
               'category_labels': network.category_labels,
               'input_shape_labels': network.point_labels,
               'template_shapes': network.temp_points,
               'template_shape_labels': network.temp_point_labels,
               'morph_shapes': network.points_deform,
               'pred_category': network.category_pred,
               'real_seg_preds': network.real_seg_preds,
               'real_seg_loss': network.real_seg_loss,
               'fake_seg_preds': network.fake_seg_preds,
               'fake_seg_loss': network.fake_seg_loss,
               'total_loss': total_loss,
               'shape_loss': network.shape_loss,
               'category_loss': network.category_loss,
               'train_op': train_op,
               'extra_update_ops': extra_update_ops,
               'merged': merged_train,
               'step': batch,
               'is_training_pl': is_training_pl
               }
        # ?????? update the ops
        if IS_TRANS is not 'no_trans':
            ops['trans_matrix'] = network.transform_matrix

        if IS_KEY is True:
            ops['key_list_points'] = network.key_list_points
            ops['key_list_temp'] = network.key_list_temp
            ops['key_loss'] = network.key_loss

        saver.save(sess, os.path.join(LOG_DIR, "model_init.ckpt"))
        # saver.restore(sess, os.path.join(LOG_DIR, "model15.ckpt"))
        for epoch in range(MAX_EPOCH):

            log_string('**** EPOCH %03d ****' % (epoch))
            train_one_epoch(sess, ops, train_writer, batch)
            if (epoch+1) % 5 == 0:
                saver.save(sess, os.path.join(LOG_DIR, "model" + str(epoch + 1) + ".ckpt"))


def train_one_epoch(sess, ops, train_writer, batch):
    total_seen = 0
    loss_sum = 0
    shape_list = sp.get_file_list(DATA_DIR, '.h5')
    # Shuffle train files
    train_file_ids = np.arange(0, len(shape_list))
    np.random.shuffle(train_file_ids)
    total_batch = len(shape_list) // BATCH_SIZE

    for fn in range(0, total_batch):
        start_id = fn * BATCH_SIZE
        Data = []
        Seg_Label = []
        Categroy_Label = []
        Temp = []
        for sn in range(BATCH_SIZE):
            shape_temp = os.path.join(DATA_DIR, shape_list[train_file_ids[start_id+sn]])
            f = h5py.File(shape_temp)
            points = f['points'][:]
            seg_labels = f['seg_labels'][:]
            category_labels = f['shape_labels'].value-1
            templates = f['shape_temp'][:]
            f.close()
            points, seg_labels, _ = pf.shuffle_data(points, seg_labels)
            Data.append(points)
            Seg_Label.append(seg_labels)
            Categroy_Label.append(category_labels)
            Temp.append(templates)

        Data = np.array(Data)
        Seg_Label = np.array(Seg_Label)
        Categroy_Label = np.array(Categroy_Label)
        Temp = np.array(Temp)

        rotated_data = pf.rotate_point_cloud(Data)
        jittered_data = pf.jitter_point_cloud(rotated_data)
        batch_val = sess.run(batch)

        if (IS_SHOW == 'True') & ((batch_val+1) % 5000 == 1):
            mlab.figure('points', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.points3d(10 * Data[0, :, 0], 10 * Data[0, :, 1], 10 * Data[0, :, 2], Seg_Label[0, :],
                          scale_factor=0.2, scale_mode='vector')
            mlab.show()
            mlab.figure('temp', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.points3d(10 * Temp[0, :, 0], 10 * Temp[0, :, 1], 10 * Temp[0, :, 2], color=(0.7, 0.7, 0.7),
                          scale_factor=0.2, scale_mode='vector')


            mlab.show()

        feed_dict = {ops['input_shapes']: jittered_data,
                     ops['category_labels']: Categroy_Label,
                     ops['input_shape_labels']: Seg_Label,
                     ops['template_shapes']: Temp,
                     ops['is_training_pl']: 'True'
                     }

        summary, step, _, loss_val, morhp_shapes, real_seg, fake_seg = sess.run([ops['merged'],
                                                        ops['step'],
                                                        ops['train_op'],
                                                        ops['total_loss'],
                                                        ops['morph_shapes'],
                                                        ops['real_seg_preds'],
                                                        ops['fake_seg_preds']],
                                                        feed_dict=feed_dict)


        real_seg_labels = np.argmax(real_seg, axis=-1)
        fake_seg_labels = np.argmax(fake_seg, axis=-1)

        if (IS_SHOW == 'True') & ((batch_val+1) % 5000 == 1):
            mlab.figure('real_seg', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.points3d(10 * Data[0, :, 0], 10 * Data[0, :, 1], 10 * Data[0, :, 2], real_seg_labels[0, :],
                          scale_factor=0.2, scale_mode='vector')
            mlab.figure('fake_seg', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.points3d(10 * morhp_shapes[0, :, 0], 10 * morhp_shapes[0, :, 1], 10 * morhp_shapes[0, :, 2],
                          fake_seg_labels[0, :], scale_factor=0.2, scale_mode='vector')
            mlab.show()
            mlab.figure('morph_shape', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.points3d(10 * morhp_shapes[0, :, 0], 10 * morhp_shapes[0, :, 1], 10 * morhp_shapes[0, :, 2], color=(0.7, 0.7, 0.7),
                          scale_factor=0.2, scale_mode='vector')
            mlab.show()
            # show the correspondence between the temptation and the morphing shape
            c = Temp[0, 0, :]
            sp.vis_correspondance(Temp[0, ...], morhp_shapes[0, ...], Temp[0, ...], morhp_shapes[0, ...], c,
                                  if_gt=False, if_save=False, scale_factor=0.03)

            direct = morhp_shapes[0, ...]-Temp[0, ...]

            mlab.figure('move', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.quiver3d(Temp[0, ::20, 0], Temp[0, ::20, 1], Temp[0, ::20, 2],
                          direct[::20, 0], direct[::20, 1], direct[::20, 2], scale_factor=1)
            mlab.points3d( morhp_shapes[0, :, 0], morhp_shapes[0, :, 1], morhp_shapes[0, :, 2], color=(0.7, 0.7, 0.7),
                          scale_factor=0.02, scale_mode='vector')
            mlab.show()

        train_writer.add_summary(summary, step)
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        log_string('batch loss: %f' % (loss_val))

    log_string('mean loss: %f' % (loss_sum*BATCH_SIZE / float(total_seen)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()


