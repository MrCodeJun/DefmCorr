import numpy as np
import trimesh
from mayavi import mlab
import open3d
import sample_points as sp
import os
import h5py
import random
import sample_points


def sphere_normal_point(point_num, max_v, min_v):
    p = np.zeros([point_num, 3])
    for i in range(point_num):
        x = random.uniform(min_v, max_v)  # 服从标准正态分布的随机数
        y = random.uniform(min_v, max_v)
        z = random.uniform(min_v, max_v)
        r = (x * x + y * y + z * z) ** (1 / 2)
        p[i, :] = [x/r, y/r, z/r]
    return p


def generate_template_shape_v0(point_num):
    # this method just give fix number points distributed on a [-1,1] sphere:
    max_v = 1
    min_v = -1
    s_points = sphere_normal_point(point_num, max_v, min_v)
    return s_points


# give the file dir and save file
IS_VISUAL = False
MESH_DIR_ROOT = '/Users/jinwei/Match/Data/Models'
V0_DATA_DIR = '/Users/jinwei/Match/Data/shapenetcore_partanno_segmentation_benchmark_v0'

SAVE_H5_DIR = '/Users/jinwei/Match/Data/benchmark_v0_deform'
if not os.path.exists(SAVE_H5_DIR):
    os.mkdir(SAVE_H5_DIR)

category_file = os.path.join(V0_DATA_DIR, 'synsetoffset2category.txt')
category = {}
f = open(category_file, 'r')
lines = f.read().strip('\n').split('\n')
for line in lines:
    c = line.split('\t')
    category[c[0]] = c[1]
print(category)

count = 0
sample_num = 2048
for key, value in sorted(category.items()):
    count = count+1
    v0_dir = os.path.join(os.path.join(V0_DATA_DIR, value), 'points')
    mesh_dir = os.path.join(MESH_DIR_ROOT, key)
    files_list = sp.get_file_list(v0_dir, '.pts')

    # generate the template for shapes belong to the same category
    temp_points = generate_template_shape_v0(sample_num)
    for file in files_list:
        # give the mesh, points and the seg file path:
        mesh_file = os.path.join(mesh_dir, file[:len(file)-3]+'off')
        label_file = os.path.join(os.path.join(os.path.join(V0_DATA_DIR, value),
                                               'points_label'), file[:len(file)-3] + 'seg')
        point_file = os.path.join(v0_dir, file)

        # read the .pts files (point_file) and normalize the points to [-1,1]:
        # the points have same elevation direction y-axis and x-axis is given as the face direction
        points = np.loadtxt(point_file)
        max_p = np.max(points, axis=0)
        min_p = np.min(points, axis=0)
        max_val = (np.max(np.abs(points)))
        o_p = 0.5*(max_p + min_p)
        points = (points-o_p)/max_val
        new_points, new_id = sample_points.graipher(points, sample_num)
        # read .seg files get the seg labels of the points
        seg_labels = np.loadtxt(label_file)
        new_seg_labels = seg_labels[new_id]
        # give the shape labels(10)
        shape_labels = np.asarray(count)

        if IS_VISUAL is True:
            mlab.figure('points', fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.points3d(10 * new_points[:, 0], 10 * new_points[:, 1], 10 * new_points[:, 2], new_seg_labels,
                          scale_factor=0.1, scale_mode='vector')

            mlab.points3d(10 * temp_points[:, 0], 10 * temp_points[:, 1], 10 * temp_points[:, 2], color=(0, 0, 1),
                          scale_factor=0.1, scale_mode='vector')
            mlab.show()

        # save to a .h5 file for each shape
        current_name = os.path.join(SAVE_H5_DIR, file[:len(file) - 4] + '_' + value + '.h5')
        f = h5py.File(current_name, "w")
        f.create_dataset("points", new_points.shape, dtype='f', data=new_points)
        f.create_dataset("seg_labels", new_seg_labels.shape, dtype='int', data=new_seg_labels)
        f.create_dataset("shape_labels", shape_labels.shape, dtype='int', data=shape_labels)
        f.create_dataset("shape_temp", temp_points.shape, dtype='f', data=temp_points)
        f.close()




