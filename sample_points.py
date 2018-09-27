import os
import numpy as np
import trimesh
from scipy import spatial
import time
import sklearn.neighbors
import sklearn.metrics
import open3d
from mayavi import mlab
import h5py

def chamfer_distance(points1, points2):
    num_points = points1.shape[0]
    tree1 = sklearn.neighbors.KDTree(points1, leaf_size=num_points + 1)
    tree2 = sklearn.neighbors.KDTree(points2, leaf_size=num_points + 1)
    distances1, _ = tree1.query(points2)
    distances2, _ = tree2.query(points1)
    av_dist1 = np.mean(distances1)
    av_dist2 = np.mean(distances2)
    dist = av_dist2 + av_dist1
    return dist


def get_corrs_shapes(corrs_file, mesh_dir, shape_num=20000):
    """
    corrs_file : a file such as
               1a04e3eab45ca15dd86060f189eb133___corr___1c2e9dedbcf511e616a077c4c0fc1181.pts

    mesh_dir: the *off files

    sampling_num:  points num

    output  2 mesh and 2 the corrs points sets
    """
    if not os.path.exists(corrs_file):
        print("File is not found")
    else:
        key_word = '___corr___'
        path, file = os.path.split(corrs_file)
        [part1, part2] = file.split(key_word)
        part2 = part2[:len(part2) - 4]
        shape_file1 = os.path.join(mesh_dir, part1)
        shape_file1 = shape_file1 + '.off'  # the mesh is *.off
        shape_file2 = os.path.join(mesh_dir, part2)
        shape_file2 = shape_file2 + '.off'  # the mesh is *.off

        if os.path.exists(shape_file1) and os.path.exists(shape_file2):
            mesh1 = trimesh.load_mesh(shape_file1)
            mesh2 = trimesh.load_mesh(shape_file2)

            # uniform sampling points 20000 points
            points1, face_ids1, normals1 = mesh2points(mesh1, shape_num)
            points2, face_ids2, normals2 = mesh2points(mesh2, shape_num)
            # pcd1 = open3d.PointCloud()
            # for i in range(len(face_ids1)):
            #     pcd1.points.append(points1[i])
            #     pcd1.normals.append(normals1[i])
            # pcd2 = open3d.PointCloud()
            # for i in range(len(face_ids2)):
            #     pcd2.points.append(points2[i])
            #     pcd2.normals.append(normals2[i])
            return points1, points2, face_ids1, face_ids2, normals1, normals2
        else:
            print("The mesh files are not exists")


def mesh2points(mesh, count):
    """
    given a 3d mesh and return the uniform samples points on the surface
    input: a 3d mesh
    the number of the sample points
    output: points of the 3d mesh, id, normals
    """
    # of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    normals = mesh.face_normals
    new_normals = normals[face_index, :]

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index, new_normals


def get_corres_points(corrs_file):
    """given the corrs file and return two corres point sets
        input: a corres file
        output:two points sets
    """
    try:
        fp = open(corrs_file, 'r')
    except FileNotFoundError:
        print("The file is not found.")
    else:
        corres_array = np.loadtxt(fp)
        points1 = corres_array[:, 0:3]
        points2 = corres_array[:, 3:6]
        return points1, points2


def generate_corres_qurey_points(corres_file, mesh_dir, is_fpfh, shape_num, query_num, input_num, voxel_size):
    cp1, cp2 = get_corres_points(corres_file)

    # get_corrs_shapes
    p1, p2, face1, face2, normals1, normals2 = get_corrs_shapes(corres_file, mesh_dir, shape_num)
    max_p1 = np.max(p1, axis=0)
    min_p1 = np.min(p1, axis=0)
    center1 = 0.5 * (max_p1 + min_p1)
    p1 = p1 - center1

    max_p2 = np.max(p2, axis=0)
    min_p2 = np.min(p2, axis=0)
    center2 = 0.5 * (max_p2 + min_p2)
    p2 = p2 - center2
    max1 = (np.max(np.abs(p1)))
    max2 = (np.max(np.abs(p2)))

    cp1 = (cp1 - center1) / max1
    cp2 = (cp2 - center2) / max2
    p1 = p1 / max1
    p2 = p2 / max2

    # farthest sampling: query points
    fcp1, f_id = graipher(cp1, query_num // 2)
    fcp2, f_id = graipher(cp2, query_num, f_id)
    fcp1 = cp1[f_id]
    ps1, id1 = graipher(p1, input_num)
    ps2, id2 = graipher(p2, input_num)

    if is_fpfh == True:
        f1 = local_geomtric_fpfh(p1, face1, normals1, voxel_size=voxel_size)
        f1 = f1.data
        f2 = local_geomtric_fpfh(p2, face2, normals2, voxel_size=voxel_size)
        f2 = f2.data
        f1 = f1[:, id1]
        f1 = np.transpose(f1,[1,0])
        f2 = f2[:, id2]
        f2 = np.transpose(f2,[1,0])
    else:
        f1 = []
        f2 = []

    data1 = {'data': ps1,
             'center': fcp1,
             'feature': f1
             }
    data2 = {'data': ps2,
             'center': fcp2,
             'feature': f2
             }
    return data1, data2


def local_geomtric_fpfh(points, face_index, normals, voxel_size=0.02):
    pcd_points = open3d.PointCloud()
    for i in range(len(face_index)):
        pcd_points.normals.append(normals[i])
        pcd_points.points.append(points[i])
    # radius_normal = voxel_size * 2
    # open3d.estimate_normals(pcd_points, open3d.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    points_fpfh = open3d.compute_fpfh_feature(pcd_points, open3d.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return points_fpfh


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def graipher(pts, K, init_points_id=[]):
    """
    sample some farthest points on the 3d points
    pts: the points
    K: the sampling points
    nit_points: given the first point
    output: K sampling points
    """
    farthest_pts = np.zeros((K, 3))
    farthest_ind = np.arange(K)
    if not any(init_points_id):
        farthest_ind[0] = np.random.randint(len(pts))
        farthest_pts[0] = pts[farthest_ind[0]]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(K):
            farthest_ind[i] = np.argmax(distances)
            farthest_pts[i] = pts[farthest_ind[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    else:
        l = len(init_points_id)
        farthest_ind[0:l] = init_points_id
        farthest_pts[0:l] = pts[init_points_id]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(l):
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
        for i in range(l, K):
            farthest_ind[i] = np.argmax(distances)
            farthest_pts[i] = pts[farthest_ind[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, farthest_ind


def search_near_points(input_points, query_points, method, r=0.3):
    """
    query the poinytss for neighbors within a radius r
    for r = 0.3 with 1024 query points and 25000 shape points
    sk_kdtree : about 0.2s
    sk_balltree: about 0.06s
    sp_kdtree: too slow maybe 200s

    return the index（an array of objects）
    """

    if method == 'sk_kdtree':
        tree = sklearn.neighbors.KDTree(input_points, leaf_size=2)
        return tree.query_radius(query_points, r)

    if method == 'sk_balltree':
        tree = sklearn.neighbors.BallTree(input_points, leaf_size=2)
        return tree.query_radius(query_points, r)

    if method == 'sp_kdtree':
        x = input_points[:, 0]
        y = input_points[:, 1]
        z = input_points[:, 2]
        data = list(zip(x.ravel(), y.ravel(), z.ravel()))
        tree = spatial.KDTree(data)
        query_points_x = query_points[:, 0]
        query_points_y = query_points[:, 1]
        query_points_z = query_points[:, 2]
        query = list(zip(query_points_x.ravel(), query_points_y.ravel(), query_points_z.ravel()))
        return tree.query_ball_point(query, r)


def get_negative_query_points(query_points1, query_points2, num, r=0.5):
    """  obtain the negative query indices from the given query points  """
    dist1 = sklearn.metrics.pairwise.euclidean_distances(query_points1, query_points1)
    dist2 = sklearn.metrics.pairwise.euclidean_distances(query_points2, query_points2)
    I, J = ((dist1 > r) & (dist2 > r)).nonzero()
    if len(I) <= num:
        return I, J
    else:
        ind = np.random.choice(len(I), num)
        return I[ind], J[ind]


def normalize_part_points(prts, center_point, radius, num=512):
    """points move to center"""
    l = prts.shape[0]
    if l <= num:
        temp = np.zeros((num, 3))
        temp[0:l, :] = prts
        norm_points = temp
    else:
        ind = np.random.choice(l, num)
        norm_points = prts[ind, :]
    norm_points = (norm_points - center_point) / radius
    return norm_points


def get_file_list(file_dir, key):
    file_list = []
    f_list = os.listdir(file_dir)
    for file in f_list:
        if os.path.splitext(file)[1] == key:
            file_list.append(file)
    return np.array(file_list)


def read_bhcp_corrs(file):
    try:
        f = open(file, 'r')
    except FileNotFoundError:
        print("the file can not be found")
    else:
        idp = 0
        corres_id = []
        corres_points = []
        lines = f.readlines()
        for line in lines:
            content = line.strip('\n')
            content = content.split(" ")
            if content[0] == 'VALID':
                pass
            elif content[0] == str(-1):
                idp = idp + 1
            else:
                corres_id.append(idp)
                p = content[len(content) - 3:len(content)]
                p = content[len(content) - 3:len(content)]
                p = list(map(eval, p))
                corres_points.append(p)
                idp = idp + 1
    return corres_points, corres_id


def generate_train_samples(corres_file, mesh_dir, radius, train_sampling_num, N, neg_dist, prts_num):
    # get_corres_points
    start = time.time()
    cp1, cp2 = get_corres_points(corres_file)
    # get_corrs_shapes
    p1, p2, m1, m2 = get_corrs_shapes(corres_file, mesh_dir, N)
    # shape to [-1 1]
    p1_con = np.concatenate([p1, cp1], axis=0)
    p2_con = np.concatenate([p2, cp2], axis=0)
    max1 = (np.max(np.abs(p1_con)))
    max2 = (np.max(np.abs(p2_con)))
    cp1 = cp1 / max1
    cp2 = cp2 / max2
    p1_con = p1_con / max1
    p2_con = p2_con / max2
    # farthest sampling
    fcp1, f_id = graipher(cp1, train_sampling_num // 2)
    fcp2, f_id = graipher(cp2, train_sampling_num, f_id)
    fcp1 = cp1[f_id]
    # neighbor points search for query points ball query
    out1 = search_near_points(p1_con, fcp1, method='sk_balltree', r=radius)
    out2 = search_near_points(p2_con, fcp2, method='sk_balltree', r=radius)
    # negative points query
    I, J = get_negative_query_points(fcp1, fcp2, train_sampling_num, neg_dist)
    print(len(I))
    # generate training points set
    train_data1 = []
    train_data2 = []
    train_label = []
    for i in range(train_sampling_num):
        part1 = p1_con[out1[i]]
        center1 = fcp1[i]
        train_data1.append(normalize_part_points(part1, center1, radius, prts_num))
        part2 = p2_con[out2[i]]
        center2 = fcp2[i]
        train_data2.append(normalize_part_points(part2, center2, radius, prts_num))
        train_label.append(1)
        if i <= len(I):
            part1_n = p1_con[out1[I[i]]]
            center1_n = fcp1[I[i]]
            train_data1.append(normalize_part_points(part1_n, center1_n, radius, prts_num))
            part2_n = p2_con[out2[J[i]]]
            center2_n = fcp2[J[i]]
            train_data2.append(normalize_part_points(part2_n, center2_n, radius, prts_num))
            train_label.append(0)
        else:
            pass
    train_data1 = np.array(train_data1)
    train_data2 = np.array(train_data2)
    train_label = np.array(train_label)
    return train_data1, train_data2, train_label


def generate_corres_points_samples(corres_file, mesh_dir, radius, train_sampling_num, N, neg_dist, prts_num):
    cp1, cp2 = get_corres_points(corres_file)
    # get_corrs_shapes
    p1, p2, m1, m2 = get_corrs_shapes(corres_file, mesh_dir, N)
    # shape to [-1 1]
    p1_con = np.concatenate([p1, cp1], axis=0)
    p2_con = np.concatenate([p2, cp2], axis=0)
    max1 = (np.max(np.abs(p1_con)))
    max2 = (np.max(np.abs(p2_con)))
    cp1 = cp1 / max1
    cp2 = cp2 / max2
    p1_con = p1_con / max1
    p2_con = p2_con / max2
    # farthest sampling
    fcp1, f_id = graipher(cp1, train_sampling_num // 2)
    fcp2, f_id = graipher(cp2, train_sampling_num, f_id)
    fcp1 = cp1[f_id]
    # neighbor points search for query points ball query
    out1 = search_near_points(p1_con, fcp1, method='sk_balltree', r=radius)
    out2 = search_near_points(p2_con, fcp2, method='sk_balltree', r=radius)

    points_data1 = []
    points_data2 = []
    center_data1 = []
    center_data2 = []

    for i in range(train_sampling_num):
        part1 = p1_con[out1[i]]
        center1 = fcp1[i]
        points_data1.append(normalize_part_points(part1, center1, radius, prts_num))
        center_data1.append(center1)

        part2 = p2_con[out2[i]]
        center2 = fcp2[i]
        points_data2.append(normalize_part_points(part2, center2, radius, prts_num))
        center_data2.append(center2)

    points_data1 = np.array(points_data1)
    center_data1 = np.array(center_data1)

    points_data2 = np.array(points_data2)
    center_data2 = np.array(center_data2)
    data1 = {'data': points_data1,
             'center': center_data1,
             'feature': []
             }
    data2 = {'data': points_data2,
             'center': center_data2,
             'feature': []
             }
    return data1, data2


def shuffle_data(data, labels):
    """ Shuffle data and labels.
		Input:
		  data: B,N,... numpy array
		  label: B,... numpy array
		Return:
		  shuffled data, label and shuffle indices
	"""
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
		rotation is per shape based along up direction
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    if batch_data.ndim == 2:
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data
        rotated_data = np.dot(shape_pc, rotation_matrix)
    else:
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, jittered batch of point clouds
	"""
    N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def calculate_saliency_map(data):
    """show the saliency points
	data：a dictionary include ：
	'data'：M*N*3
	'center'：M*3
	'feature'：M*128（featuredim=128）
	"""
    features = data['feature']
    dist = sklearn.metrics.pairwise.euclidean_distances(features, features)
    dist_mean = np.mean(dist, axis=1)
    return dist_mean


def generate_train_samples_multiscale(corres_file, mesh_dir, radius, train_sampling_num, N, neg_dist, prts_num):
    pass


def generate_validation_query_points(mesh_file, corres_file, is_fpfh, shape_num, query_num, input_num, voxel_size):
    """
	using bhcp dataset:
	mesh_file ='/Users/jinwei/Match/Data/CorrsTmplt/Chair/XXXXXXX.off
	corres_file = '/Users/jinwei/Match/Data/CorrsTmplt/gt/chair_gt/XXXXXXX.txt'
	radius = 0.4
	sampling_num = 256
	N = 25000
	parts_num = 512
	output: the point prts and the key points index
	"""
    current_mesh = trimesh.load_mesh(mesh_file)
    current_points_t, faceid, normals_t = mesh2points(current_mesh, shape_num)
    current_points = np.ones(current_points_t.shape)
    normals = np.ones(normals_t.shape)
    current_points[:, 0] = current_points_t[:, 0]
    current_points[:, 2] = current_points_t[:, 1]
    current_points[:, 1] = current_points_t[:, 2]
    normals[:, 0] = normals_t[:, 0]
    normals[:, 2] = normals_t[:, 1]
    normals[:, 1] = normals_t[:, 2]

    max_p = np.max(current_points, axis=0)
    min_p = np.min(current_points, axis=0)
    center = 0.5 * (min_p + max_p)

    # center = np.mean(current_points,axis=0)
    current_points = current_points - center
    max_value = (np.max(np.abs(current_points)))
    current_points = current_points / max_value

    ps1, id1 = graipher(current_points, input_num)
    if is_fpfh == True:
        f1 = local_geomtric_fpfh(current_points, faceid, normals, voxel_size=voxel_size)
        f1 = f1.data
        f1 = np.transpose(f1, [1, 0])
        f1 = f1[id1, :]
    else:
        f1 = []
    p_t, ind = read_bhcp_corrs(corres_file)
    p_t = np.array(p_t)
    p = np.ones(p_t.shape)
    p[:, 0] = p_t[:, 0]
    p[:, 2] = p_t[:, 1]
    p[:, 1] = p_t[:, 2]
    p = (p - center) / max_value
    farthest_pts, farthest_pts_id = graipher(current_points, query_num - p.shape[0])
    farthest_pts = np.concatenate([p, farthest_pts], axis=0)

    return ps1, f1, ind, farthest_pts


def generate_validation_query_points_h5(mesh_file_h5, is_fpfh, shape_num, query_num, input_num, voxel_size):
    """
	using bhcp dataset:
	mesh_file ='/Users/jinwei/Match/Data/CorrsTmplt/Chair/XXXXXXX.off
	corres_file = '/Users/jinwei/Match/Data/CorrsTmplt/gt/chair_gt/XXXXXXX.txt'
	radius = 0.4
	sampling_num = 256
	N = 25000
	parts_num = 512
	output: the point prts and the key points index
	"""

    fp = h5py.File(mesh_file_h5,'r')
    current_points = fp['points']
    normals = fp['normals']
    p = fp['ket_points']
    ind = fp['key_index']
    faceid = np.ones(normals.shape[0])

    max_p = np.max(current_points, axis=0)
    min_p = np.min(current_points, axis=0)
    center = 0.5 * (min_p + max_p)

    # center = np.mean(current_points,axis=0)
    current_points = current_points - center
    max_value = (np.max(np.abs(current_points)))
    current_points = current_points / max_value

    ps1, id1 = graipher(current_points, input_num)
    if is_fpfh == True:
        f1 = local_geomtric_fpfh(current_points, faceid, normals, voxel_size=voxel_size)
        f1 = f1.data
        f1 = np.transpose(f1, [1, 0])
        f1 = f1[id1, :]
    else:
        f1 = []
    p = (p - center) / max_value
    farthest_pts, farthest_pts_id = graipher(current_points, query_num - p.shape[0])
    farthest_pts = np.concatenate([p, farthest_pts], axis=0)

    return ps1, f1, ind, farthest_pts


def generate_validation_query_points_h5_capture(mesh_file_h5, is_fpfh, shape_num, query_num, input_num, voxel_size):
    """
	using bhcp dataset:
	mesh_file ='/Users/jinwei/Match/Data/CorrsTmplt/Chair/XXXXXXX.off
	corres_file = '/Users/jinwei/Match/Data/CorrsTmplt/gt/chair_gt/XXXXXXX.txt'
	radius = 0.4
	sampling_num = 256
	N = 25000
	parts_num = 512
	output: the point prts and the key points index
	"""

    fp = h5py.File(mesh_file_h5,'r')
    current_points = fp['points']
    normals = fp['normals']
    # p = fp['ket_points']
    # ind = fp['key_index']
    faceid = np.ones(normals.shape[0])

    max_p = np.max(current_points, axis=0)
    min_p = np.min(current_points, axis=0)
    center = 0.5 * (min_p + max_p)

    # center = np.mean(current_points,axis=0)
    current_points = current_points - center
    max_value = (np.max(np.abs(current_points)))
    current_points = current_points / max_value

    ps1, id1 = graipher(current_points, input_num)
    if is_fpfh == True:
        f1 = local_geomtric_fpfh(current_points, faceid, normals, voxel_size=voxel_size)
        f1 = f1.data
        f1 = np.transpose(f1, [1, 0])
        f1 = f1[id1, :]
    else:
        f1 = []
    # p = (p - center) / max_value
    farthest_pts, farthest_pts_id = graipher(current_points, query_num)
    # farthest_pts = np.concatenate([p, farthest_pts], axis=0)

    return ps1, f1, farthest_pts


def generate_validation_samples(mesh_file, corres_file, radius, sampling_num, N, parts_num):
    """
	using bhcp dataset:
	mesh_file ='/Users/jinwei/Match/Data/CorrsTmplt/Chair/XXXXXXX.off
	corres_file = '/Users/jinwei/Match/Data/CorrsTmplt/gt/chair_gt/XXXXXXX.txt'
	radius = 0.4
	sampling_num = 256
	N = 25000
	parts_num = 512
	output: the point prts and the key points index
	"""
    current_mesh = trimesh.load_mesh(mesh_file)
    current_points, _ = mesh2points(current_mesh, N)
    center = np.mean(current_points, axis=0)
    current_points = current_points - center
    max_value = (np.max(np.abs(current_points)))
    current_points = current_points / max_value
    p, ind = read_bhcp_corrs(corres_file)
    p = (p - center) / max_value
    current_points = np.concatenate([p, current_points], axis=0)
    farthest_pts, farthest_pts_id = graipher(current_points, sampling_num, np.arange(p.shape[0]))
    out = search_near_points(current_points, farthest_pts, 'sk_balltree', radius)
    train_data = []
    for i in range(sampling_num):
        part = current_points[out[i]]
        center = farthest_pts[i]
        train_data.append(normalize_part_points(part, center, radius, parts_num))
    train_data = np.array(train_data)
    return train_data, ind, farthest_pts


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    lists = [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value
    isexist_list = []
    for ind, v in enumerate(lists):
        if v != None:
            isexist_list.append(v)
    return lists, isexist_list


def vis_correspondance(current_data1, current_data2, feature1, feature2, c, if_gt=False, if_save=False, save_name=None, scale_factor=0.08):
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    distance = sklearn.metrics.pairwise.pairwise_distances(feature2, feature1, metric='euclidean')
    dd = np.argmin(distance, axis=1)

    d = np.sqrt(np.sum(np.square(current_data1-c), axis=-1))
    d = (d-np.min(d))/(np.max(d)-np.min(d))

    mlab.figure('color1',fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.points3d(current_data1[:, 0], current_data1[:, 1], current_data1[:, 2], d, scale_factor=scale_factor,
                  scale_mode='vector')
    mlab.figure('color2',fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.points3d(current_data2[:, 0], current_data2[:, 1], current_data2[:, 2], d[dd], scale_factor=scale_factor,
                  scale_mode='vector')
    #mlab.points3d(c[0], c[1], c[2], scale_factor=0.1, color=(1, 0, 0),
    #             scale_mode='vector')
    if if_save==True:
        mlab.savefig(save_name)
        mlab.show()
        mlab.close(all=True)
    else:
        mlab.show()

    if if_gt == True:
        mlab.figure('color_gt')
        mlab.points3d(current_data1[:, 0], current_data1[:, 1], current_data1[:, 2], d, scale_factor=scale_factor,
                      scale_mode='vector')
        mlab.points3d(current_data2[:, 0]+2, current_data2[:, 1], current_data2[:, 2], d, scale_factor=scale_factor,
                      scale_mode='vector')
        mlab.show()



def compute_CMC_curve(feature_lists, qurey_id_lists):
    """
		Cumulative Match Characteristic (CMC) measure:
		inputs : a list of features dim: N*[M,D]
		N shapes ,each shape has M points and each point has D dim features
		qurey id: a list: len(qurey_id_lists)== N each list has some query id

		output : show the curve figure and save the figure
	"""
    l = len(feature_lists)
    for i in range(l):
        for j in range(l):
            if i != j:
                print("compute the {0}th and %{1}th shape".format(str(i), str(j)))
                _, current_query_id = ismember(qurey_id_lists[i], qurey_id_lists[j])
                _, gt_id = ismember(qurey_id_lists[j], qurey_id_lists[i])
                print(current_query_id)
                print(gt_id)
                current_query_points = feature_lists[i][current_query_id]
                dist = sklearn.metrics.pairwise.euclidean_distances(current_query_points, feature_lists[j])

# file_name = '/Users/jinwei/Match/Data/Models/Airplane/1a04e3eab45ca15dd86060f189eb133.off'
# mesh = trimesh.load_mesh(file_name)
# N = 20000
# sample_num = 3096
# points, face_index, normals = mesh2points(mesh, N)
# f = local_geomtric_fpfh(points, face_index, normals, 0.1)
# ind = np.arange(0,N)
# np.random.shuffle(ind)
# ind = ind[:sample_num]
# points_s = points[ind]
# f_s = f[:,ind]
# mlab.figure()
# mlab.points3d(points[:, 0], points[:, 1], points[:, 2],f[10,:],scale_factor=.001)
# # mlab.quiver3d(points[::10, 0], points[::10, 1], points[::10, 2],
# #               normals[::10, 0], normals[::10, 1], normals[::10, 2],
# #               scale_factor=.05)
# mlab.show()


# file_name = '/Users/jinwei/Match/Data/Models/Airplane/1a04e3eab45ca15dd86060f189eb133.off'
# points, normals = convert_mesh_to_pcd(file_name,10000)
# mlab.figure('show the 3d mesh with normals')
# mlab.quiver3d(points[:,0], points[:,1], points[:,2],
#			  normals[:,0] , normals[:,1], normals[:,2],
#			  line_width=0.03, scale_factor=0.01)
# mlab.show()

# """
# ############### a demo test the functions above ###################

# SHAPE_NAME = 'Chair'
# CORRS_DIR = '/Users/jinwei/Match/Data/CorrsTmplt/'+ SHAPE_NAME
# N = 25000
# radius = 0.4
# Sample_N = 256
# prts_num = 512
# corrs_dir = os.path.join('/Users/jinwei/Match/Data/CorrsTmplt/','gt/'+ SHAPE_NAME.lower() +'_gt')
# key_point_lists = get_file_list(corrs_dir,'.txt')

# for file in key_point_lists:
# 	mesh_file = CORRS_DIR+'/'+file[:len(file)-3]+'off'
# 	print(mesh_file)
# 	data,id = generate_validation_samples(mesh_file,corrs_dir+'/'+file,radius,256,N,prts_num)
# 	print(data.shape)

# ################temp para for test the functions#############################

# file_path = '/Users/jinwei/Match/Data/Corr/Chair'
# mesh_dir = "/Users/jinwei/Match/Data/Models/Chair/"
# fig_save_dir = './save_fig'
# radius = 0.4
# train_sampling_num = 1024
# N = 25000
# neg_dist = 0.8
# prts_num = 512
# ###################### generate the training sampling ######################
# file_list = get_file_list(file_path, '.pts')
# for file_name in file_list:
# 	file_name_temp = os.path.join(file_path,file_name)
# 	node1 = time.time()
# 	train_data1,train_data2,train_label = generate_train_samples(
# 										corres_file = file_name_temp, 
# 										mesh_dir = mesh_dir, 
# 										radius = 0.4, 
# 										train_sampling_num = 1024 ,
# 						 				N =30000, 
# 						 				neg_dist =0.8, 
# 						 				prts_num =512)
# 	node2 = time.time()
# 	print("-------————————generate training points set /s——————---------")
# 	print(node2-node1)
# 	print("---------------------------train label------------------------")
# 	print(train_label.shape)
# 	print("---------------------------train data------------------------")
# 	print(train_data1.shape)


# ##1##
# start = time.time()
# cp1,cp2 = get_corres_points(file_name_temp)
# node1 = time.time()
# print("------------get_corres_points /s-------------")
# print(node1-start)

# ##2##
# node2 = time.time()
# p1,p2,m1,m2 = get_corrs_shapes(file_name_temp,mesh_dir,sampling_num = N)
# node3 = time.time()
# print("-------——————get_corrs_shapes /s——————---------")
# print(node3-node2)

# p1_con = np.concatenate([p1,cp1],axis=0)
# p2_con = np.concatenate([p2,cp2],axis=0)
# max1 = (np.max(np.abs(p1_con)))
# max2 = (np.max(np.abs(p2_con)))
# cp1 = cp1/max1
# cp2 = cp2/max2

# ##3##
# node4 = time.time()
# fcp1,f_id = graipher(cp1, train_sampling_num//2)
# fcp2,f_id = graipher(cp2, train_sampling_num,f_id)
# fcp1 = cp1[f_id]
# node5 = time.time()
# print("-------——————farthest sampling /s——————---------")
# print(node5-node4)

# p1 = p1/max1
# p2 = p2/max2
# ##4##
# node6 = time.time()
# out1 = search_near_points(p1, fcp1, method = 'sk_balltree', r = radius)
# out2 = search_near_points(p2, fcp2, method = 'sk_balltree', r = radius)
# node7 = time.time()
# print("-------————————ball query /s——————---------")
# print(node7-node6)

# ##4 ##
# node8 = time.time()
# I,J = get_negative_query_points(fcp1, fcp2, train_sampling_num, neg_dist)
# node9 = time.time()
# print("-------————————negative points query /s——————---------")
# print(node7-node6)

# ##4 ##
# node10 = time.time()
# train_data1= []
# train_data2 = []
# train_label = []
# index = 0
# for i in range(train_sampling_num):
# 	part1 = p1[out1[i]]
# 	center1 = fcp1[i] 
# 	train_data1.append(normalize_part_points(part1,center1,prts_num))
# 	part2 = p2[out2[i]]
# 	center2 = fcp2[i]  
# 	train_data2.append(normalize_part_points(part2,center2,prts_num))
# 	train_label.append(1)
# 	part1_n = p1[out1[I[i]]]
# 	center1_n = fcp1[I[i]]
# 	train_data1.append(normalize_part_points(part1_n,center1_n,prts_num))
# 	part2_n = p2[out2[J[i]]]
# 	center2_n = fcp2[J[i]]
# 	train_data2.append(normalize_part_points(part2_n,center2_n,prts_num))
# 	train_label.append(0)
# train_data1 = np.array(train_data1)
# train_data2 = np.array(train_data2)
# train_label= np.array(train_label)

# node11 = time.time()
# print("-------————————generate training points set /s——————---------")
# print(node11-node10)
# print("---------------------------train label------------------------")
# print(train_label.shape)
# print("---------------------------train data------------------------")
# print(train_data1.shape)

# ##   ???? don't know how to olny save the pics but not display the figure
# ##    on the PC screen     ?????

# ################# vis & save pics ########################
# for i in range(0,fcp1.shape[0],64):
# 	part1 = p1[out1[i]]  
# 	part2 = p2[out2[i]]
# 	part1_n = p1[out1[I[i]]]
# 	part2_n = p2[out2[J[i]]]
# 	mlab.figure("The part1 points: " + str(i) +"part")
# 	mlab.points3d(part1[:,0],part1[:,1],part1[:,2],scale_factor=.005)
# 	mlab.points3d(fcp1[i,0],fcp1[i,1],fcp1[i,2],scale_factor=.05, color = (1, 0, 0))
# 	current_save_path = os.path.join(fig_save_dir, str(i)+'_part1'+'.png')
# 	mlab.savefig(current_save_path)
# 	mlab.close(all=True)
# 	mlab.figure("The part2 points"  + str(i) + "part")
# 	mlab.points3d(part2[:,0],part2[:,1],part2[:,2],scale_factor=.005)
# 	mlab.points3d(fcp2[i,0],fcp2[i,1],fcp2[i,2],scale_factor=.05, color = (1, 0, 0))
# 	current_save_path = os.path.join(fig_save_dir, str(i)+'_part2'+'.png')
# 	mlab.savefig(current_save_path)
# 	mlab.close(all=True)

# 	mlab.figure("The part1_n points: " + str(i) +"part")
# 	mlab.points3d(part1_n[:,0],part1_n[:,1],part1_n[:,2],scale_factor=.005)
# 	mlab.points3d(fcp1[I[i],0],fcp1[I[i],1],fcp1[I[i],2],scale_factor=.05, color = (1, 0, 0))
# 	current_save_path = os.path.join(fig_save_dir, 'negative_'+str(i)+'_part1'+'.png')
# 	mlab.savefig(current_save_path)
# 	mlab.close(all=True)
# 	mlab.figure("The part1_n points: " + str(i) +"part")
# 	mlab.points3d(part2_n[:,0],part2_n[:,1],part2_n[:,2],scale_factor=.005)
# 	mlab.points3d(fcp2[J[i],0],fcp2[J[i],1],fcp2[J[i],2],scale_factor=.05, color = (1, 0, 0))
# 	current_save_path = os.path.join(fig_save_dir, 'negative_'+str(i)+'_part2'+'.png')
# 	mlab.savefig(current_save_path)
# 	mlab.close(all=True)

# ####################  vis #########################
# ##            ##
# part1 = p1[out1[10]]
# part2 = p2[out2[10]]
# mlab.figure("The part1 points")
# mlab.points3d(part1[:,0],part1[:,1],part1[:,2],scale_factor=.005)
# mlab.points3d(fcp1[10,0],fcp1[10,1],fcp1[10,2],scale_factor=.05, color = (1, 0, 0))
# mlab.figure("The part2 points")
# mlab.points3d(part2[:,0],part2[:,1],part2[:,2],scale_factor=.005)
# mlab.points3d(fcp2[10,0],fcp2[10,1],fcp2[10,2],scale_factor=.05, color = (1, 0, 0))

# v1 = m1.vertices   # array:(N_v,3)
# f1 = m1.faces      # array:(F_v,3)
# v2 = m2.vertices   # array:(N_v,3)
# f2 = m2.faces      # array:(F_v,3)

# mlab.figure("The first farthest sampling points")
# mlab.points3d(fcp1[:,0],fcp1[:,1],fcp1[:,2],scale_factor=.005)
# mlab.figure("The second farthest sampling points")
# mlab.points3d(fcp2[:,0],fcp2[:,1],fcp2[:,2],scale_factor=.005)

# mlab.figure("The first corres points")
# mlab.points3d(cp1[:,0],cp1[:,1],cp1[:,2],scale_factor=.005)
# mlab.figure("The second corres points")
# mlab.points3d(cp2[:,0],cp2[:,1],cp2[:,2],scale_factor=.005)

# mlab.figure("The first 3d mesh for correspondence")
# mlab.triangular_mesh(v1[:,0],v1[:,1],v1[:,2],f1,colormap='YlGnBu')
# mlab.figure("The second 3d mesh for correspondence")
# mlab.triangular_mesh(v2[:,0],v2[:,1],v2[:,2],f2,colormap='YlGnBu')
# mlab.figure("The first 3d mesh to points")
# mlab.points3d(p1[:,0],p1[:,1],p1[:,2],scale_factor=.005)
# mlab.figure("The second 3d mesh to points")
# mlab.points3d(p2[:,0],p2[:,1],p2[:,2],scale_factor=.005)


# mlab.figure("The first farthest sampling points + points")
# mlab.points3d(p1_con[:,0],p1_con[:,1],p1_con[:,2],scale_factor=.005)
# mlab.figure("The second farthest sampling points + points")
# mlab.points3d(p2_con[:,0],p2_con[:,1],p2_con[:,2],scale_factor=.005)

# mlab.show()
# """
