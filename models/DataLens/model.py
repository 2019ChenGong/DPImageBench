from __future__ import division
import os
import time
import math
import tensorflow as tf
import numpy as np
from models.DataLens.ops import *
from models.DataLens.utils import *
from models.DataLens.rdp_utils import *
from models.DataLens.pate_core import *
from keras.utils import np_utils
# import pandas as pd
from models.DataLens.dp_pca import ComputeDPPrincipalProjection
from sklearn.random_projection import GaussianRandomProjection
from collections import defaultdict
from tqdm import tqdm


def partition_dataset(data, labels, nb_teachers, teacher_id):
    """
    Simple partitioning algorithm that returns the right portion of the data
    needed by a given teacher out of a certain nb of teachers
    :param data: input data to be partitioned
    :param labels: output data to be partitioned
    :param nb_teachers: number of teachers in the ensemble (affects size of each
                       partition)
    :param teacher_id: id of partition to retrieve
    :return:
    """

    # Sanity check
    assert (int(teacher_id) < int(nb_teachers))

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    # Compute start, end indices of partition
    start = teacher_id * batch_len
    end = (teacher_id + 1) * batch_len

    # Slice partition off
    partition_data = data[start:end]
    if labels is not None:
        partition_labels = labels[start:end]
    else:
        partition_labels = None

    return partition_data, partition_labels

def evenly_partition_dataset(data, labels, nb_teachers):
    """
    Simple partitioning algorithm that returns the right portion of the data
    needed by a given teacher out of a certain nb of teachers
    :param data: input data to be partitioned
    :param labels: output data to be partitioned
    :param nb_teachers: number of teachers in the ensemble (affects size of each
                       partition)
    :param teacher_id: id of partition to retrieve
    :return:
    """

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    nclasses = len(labels[0])
    print("Start Index Selection")
    data_sel = [data[labels[:, j] == 1] for j in range(nclasses)]
    print("End Index Selection")
    i = 0
    data_sel_id = [0] * len(labels[0])
    partition_data = []
    partition_labels = []

    while True:
        partition_data.append(data_sel[i][data_sel_id[i]].unsqueeze(0))
        partition_labels.append(np_utils.to_categorical(i, nclasses))

        if len(partition_data) == batch_len:
            partition_data = torch.cat(partition_data)
            partition_labels = np.asarray(partition_labels)
            yield partition_data, torch.from_numpy(partition_labels).float()
            partition_data = []
            partition_labels = []

        data_sel_id[i] += 1
        if data_sel_id[i] == len(data_sel[i]):
            data_sel_id[i] = 0
        i = (i + 1) % nclasses

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

class DCGAN(object):
    def __init__(self, sess, image_size=32,
                 y_dim=10, z_dim=100, gf_dim=64, df_dim=32,
                 gfc_dim=1024, dfc_dim=256, dataset_name='default',
                 batch_teachers=10, teachers_batch=2,
                 orders=None,
                 pca=False, pca_dim=5, random_proj=False, wgan=False,
                 wgan_scale=10, config=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
          batch_teachers:  Number of teacher models in one batch. Default 10.
          teachers_batch:  Batches of training teacher models. Default 1.
        """
        self.config = config
        self.wgan = wgan
        self.wgan_scale = wgan_scale

        self.pca = pca
        self.pca_dim = pca_dim
        self.random_proj = random_proj

        self.dp_eps_list = []
        self.rdp_eps_list = []
        self.rdp_order_list = []
        self.dp_eps_list_dept = []
        self.rdp_eps_list_dept = []
        self.rdp_order_list_dept = []
        self.dataset = dataset_name
        self.batch_teachers = batch_teachers
        self.teachers_batch = teachers_batch
        self.overall_teachers = batch_teachers * teachers_batch

        self.sess = sess

        self.input_height = image_size
        self.input_width = image_size
        self.output_height = image_size
        self.output_width = image_size

        self.z_dim = z_dim
        self.y_dim = y_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name

        if orders is not None:
            self.orders = np.asarray(orders)
        else:
            self.orders = np.hstack([1.1, np.arange(2, config.orders)])

        self.rdp_counter = np.zeros(self.orders.shape)
        self.rdp_counter_dept = np.zeros(self.orders.shape)


    ## nonprivate aggregation
    def aggregate_topk(self, output_list, topk, alpha=1e-3):
        flatten_grad = np.asarray([arr.flatten() for arr in output_list])
        flatten_grad_pos = np.array(flatten_grad).clip(0)
        flatten_grad_neg = np.array(flatten_grad).clip(max=0)
        concat_grad = np.hstack((flatten_grad_pos, flatten_grad_neg))
        aggregate_grad = np.abs(np.sum(concat_grad, axis=0))

        if self.config.save_vote and self.epoch_change:
            self.epoch_change = False
            # def save_sign_weighted_votes():
            #     save_dir = os.path.join(config.checkpoint_dir, 'sign_weighted_votes.pkl')
            #     import joblib
            #     if os.path.exists(save_dir):
            #         votes = joblib.load(save_dir)
            #         votes = np.vstack((votes, aggregate_grad))
            #         joblib.dump(votes, save_dir)
            #         print(aggregate_grad.shape)
            #     else:
            #         joblib.dump(aggregate_grad, save_dir)

            # def save_sign_equal_votes():
            #     save_dir = os.path.join(config.checkpoint_dir, 'sign_equal_votes.pkl')
            #     voted_arr = np.sum(convert2topk(np.abs(concat_grad), topk), axis=0)
            #     import joblib
            #     if os.path.exists(save_dir):
            #         votes = joblib.load(save_dir)
            #         votes = np.vstack((votes, voted_arr))
            #         print(voted_arr.shape)
            #         joblib.dump(votes, save_dir)
            #     else:
            #         joblib.dump(voted_arr, save_dir)

            # def save_unsign_equal_votes():
            #     save_dir = os.path.join(config.checkpoint_dir, 'unsign_equal_votes.pkl')
            #     voted_arr = np.sum(convert2topk(np.abs(flatten_grad), topk), axis=0)
            #     import joblib
            #     if os.path.exists(save_dir):
            #         votes = joblib.load(save_dir)
            #         votes = np.vstack((votes, voted_arr))
            #         print(voted_arr.shape)

            #         joblib.dump(votes, save_dir)
            #     else:
            #         joblib.dump(voted_arr, save_dir)

            # def save_unsign_weighted_votes():
            #     save_dir = os.path.join(config.checkpoint_dir, 'unsign_weighted_votes.pkl')
            #     voted_arr = np.sum(np.abs(flatten_grad), axis=0)
            #     import joblib
            #     if os.path.exists(save_dir):
            #         votes = joblib.load(save_dir)
            #         votes = np.vstack((votes, voted_arr))
            #         print(voted_arr.shape)
            #         joblib.dump(votes, save_dir)
            #     else:
            #         joblib.dump(voted_arr, save_dir)

            # save_sign_equal_votes()
            # save_sign_weighted_votes()
            # save_unsign_equal_votes()
            # save_unsign_weighted_votes()

        topk_ind = np.argpartition(aggregate_grad, -topk)[-topk:]
        pos_ind = topk_ind[topk_ind < flatten_grad[0].shape]
        neg_ind = topk_ind[topk_ind >= flatten_grad[0].shape]
        neg_ind -= flatten_grad[0].shape
        sign_grad = np.zeros_like(flatten_grad[0])
        sign_grad[pos_ind] = 1
        sign_grad[neg_ind] = -1

        return alpha * sign_grad.reshape(output_list[0].shape)


    def aggregate_results(self, output_list, config, thresh=None, epoch=None):
        if self.pca:
            res, rdp_budget = gradient_voting_rdp(
                output_list,
                config.step_size,
                config.sigma,
                config.sigma_thresh,
                self.orders,
                pca_mat=self.pca_components,
                thresh=thresh
            )
        elif config.mean_kernel:
            from skimage.measure import block_reduce
            from skimage.transform import resize
            arr = np.asarray(output_list)
            mean_arr = block_reduce(arr, block_size=(1, 8, 8, 1), func=np.mean)
            res, rdp_budget = gradient_voting_rdp(
                mean_arr,
                config.step_size,
                config.sigma,
                config.sigma_thresh,
                self.orders,
                thresh=thresh
            )
            res = resize(res, (16, 16, 3), clip=False)
        elif config.signsgd:
            # res = self.aggregate_topk(output_list, topk=self.config.topk, alpha=self.config.learning_rate)
            # rdp_budget = 0
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = signsgd_aggregate(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
        elif config.signsgd_nothresh:
            # res = self.aggregate_topk(output_list, topk=self.config.topk, alpha=self.config.learning_rate)
            # rdp_budget = 0
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = signsgd_aggregate_no_thresh(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
        elif config.sketchsgd:
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = sketchtopk_aggregate(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
        elif config.klevelsgd:
            # res = self.aggregate_topk(output_list, topk=self.config.topk, alpha=self.config.learning_rate)
            # rdp_budget = 0
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget = k_level_sgd_aggregate(output_list, config.sigma, self.orders, config.klevel, config.thresh,
                                                alpha=config.learning_rate, b=b)
        elif config.signsgd_dept:
            b = config.max_grad if config.max_grad > 0 else None
            res, rdp_budget, dept_rdp_budget = signsgd_aggregate_dept(output_list, config.sigma, self.orders, config.topk, config.thresh,
                                                alpha=config.learning_rate, stochastic=config.stochastic, b=b)
            return res, rdp_budget, dept_rdp_budget
        elif self.random_proj:
            orig_dim = 1
            for dd in self.image_dims:
                orig_dim = orig_dim * dd

            if epoch is not None:
                proj_dim = min(epoch + 1, self.pca_dim)
            else:
                proj_dim = self.pca_dim

            n_data = output_list[0].shape[0]
            if config.proj_mat > 1:
                proj_dim_ = proj_dim // config.proj_mat
                n_data_ = n_data // config.proj_mat
                orig_dim_ = orig_dim // config.proj_mat
                print("n_data:", n_data)
                print("orig_dim:", orig_dim)
                transformers = [GaussianRandomProjection(n_components=proj_dim_) for _ in range(config.proj_mat)]
                for transformer in transformers:
                    transformer.fit(np.zeros([n_data_, orig_dim_]))
                    print(transformer.components_.shape)
                proj_matrices = [np.transpose(transformer.components_) for transformer in transformers]
                res, rdp_budget = gradient_voting_rdp_multiproj(
                    output_list,
                    config.step_size,
                    config.sigma,
                    config.sigma_thresh,
                    self.orders,
                    pca_mats=proj_matrices,
                    thresh=thresh
                )
            else:
                transformer = GaussianRandomProjection(n_components=proj_dim)
                transformer.fit(np.zeros([n_data, orig_dim]))  # only the shape of output_list[0] is used
                proj_matrix = np.transpose(transformer.components_)

                # proj_matrix = np.random.normal(loc=np.zeros([orig_dim, proj_dim]), scale=1/float(proj_dim), size=[orig_dim, proj_dim])
                res, rdp_budget = gradient_voting_rdp(
                    output_list,
                    config.step_size,
                    config.sigma,
                    config.sigma_thresh,
                    self.orders,
                    pca_mat=proj_matrix,
                    thresh=thresh
                )
        else:
            res, rdp_budget = gradient_voting_rdp(output_list, config.step_size, config.sigma, config.sigma_thresh,
                                                  self.orders, thresh=thresh)
        return res, rdp_budget

    def non_private_aggregation(self, output_list, config):
        # TODO update nonprivate aggregation
        sum_arr = np.zeros(output_list[0].shape)
        for arr in output_list:
            sum_arr += arr
        return sum_arr / len(output_list)


    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.input_height, self.input_width, self.c_dim], name='real_images')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.image_dims = image_dims

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        if 'slt' in self.dataset_name or 'cifar' in self.dataset_name:
            self.G_sum = image_summary("G", self.G, max_outputs=10)

        self.updated_img = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='updated_img')
        self.g_loss = tf.reduce_sum(tf.square(self.updated_img - self.G))

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.teachers_list = []
        for i in range(self.batch_teachers):
            with tf.variable_scope("teacher%d" % i) as scope:
                D, D_logits = self.discriminator(inputs, self.y)

                scope.reuse_variables()
                D_, D_logits_ = self.discriminator(self.G, self.y)

                if self.wgan:
                    # Use WassersteinGAN loss with gradient penalty. Reference: https://github.com/jiamings/wgan/blob/master/wgan_v2.py
                    # Calculate interpolation of real and fake image
                    if 'mnist' in self.dataset_name:
                        alpha = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
                        alpha = tf.tile(alpha, tf.constant([1, self.input_height, self.input_width, self.c_dim]))
                    else:
                        alpha = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
                        alpha = tf.tile(alpha, tf.constant([1, self.input_size]))

                    x_hat = tf.math.multiply(alpha, inputs) + tf.math.multiply((1 - alpha), self.G)
                    _, d_hat = self.discriminator(x_hat, self.y)

                    # Calculate gradient penalty for wgan
                    ddx = tf.gradients(d_hat, x_hat)[0]
                    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
                    ddx = tf.reduce_mean(tf.square(ddx - 1.0) ** 2 * self.wgan_scale)

            if self.wgan:
                teacher = {
                    'd_loss': tf.reduce_mean(D_logits_) - tf.reduce_mean(D_logits) + ddx,
                    'g_loss': -tf.reduce_mean(D_logits_),
                }
            else:
                teacher = {
                    'd_loss': tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D))) + \
                              tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_))),
                    'g_loss': tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_))),
                }

            teacher.update({
                'd_loss_sum': scalar_summary("d_loss_%d" % i, teacher['d_loss']),
                'g_loss_sum': scalar_summary("g_loss_%d" % i, teacher['g_loss']),
            })

            # calculate the change in the images that would minimize generator loss
            teacher['img_grads'] = -tf.gradients(teacher['g_loss'], self.G)[0]

            if 'slt' in self.dataset_name:
                teacher['img_grads_sum'] = image_summary("img_grads", teacher['img_grads'], max_outputs=10)

            self.teachers_list.append(teacher)

        t_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        add_save = [g for g in g_list if "moving_mean" in g.name]
        add_save += [g for g in g_list if "moving_variance" in g.name]

        self.save_vars = t_vars + add_save

        self.d_vars = []
        for i in range(self.batch_teachers):
            self.d_vars.append([var for var in t_vars if 'teacher%d' % i in var.name])
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_save_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_save_vars = [var for var in t_vars if 'd_' in var.name]
        # print(self.d_save_vars)
        print(self.save_vars)
        # self.d_save_vars = {'k': v for k, v in zip(self.d_save_vars, self.d_save_vars)}
        self.saver = tf.train.Saver(max_to_keep=5, var_list=self.save_vars)
        self.saver_g = tf.train.Saver(max_to_keep=5, var_list=self.g_save_vars)
        self.saver_d = tf.train.Saver(max_to_keep=self.teachers_batch, var_list=self.d_save_vars)

    def get_random_labels(self, batch_size):
        # print(self.y_dim)
        y_vec = np.zeros((batch_size, self.y_dim), dtype=np.float)
        y = np.random.randint(0, self.y_dim, batch_size)

        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return y_vec

    def train_together(self, sensitive_data, config):
        # Load the dataset, ignore test data for now

        print("Dataset load finished!")
        # Load the dataset, ignore test data for now
        self.batch_size = config.batch_size
        data_X, data_y = sensitive_data
        self.c_dim = data_X.shape[-1]
        self.grayscale = (self.c_dim == 1)
        self.input_height = self.input_width = data_X.shape[1]
        self.output_height = self.output_width = data_X.shape[2]
        train_data_list = []
        train_label_list = []

        # if non_private:
        #     for i in range(self.overall_teachers):
        #         partition_data, partition_labels = partition_dataset(self.data_X, self.data_y, 1, i)
        #         self.train_data_list.append(partition_data)
        #         self.train_label_list.append(partition_labels)
        # else:

        self.save_dict = defaultdict(lambda: False)
        # stats = []
        if config.shuffle:
            gen = evenly_partition_dataset(data_X, data_y, self.overall_teachers)
            for i in tqdm(range(self.overall_teachers)):
                partition_data, partition_labels = next(gen)
                train_data_list.append(partition_data)
                train_label_list.append(partition_labels)
                # stats.append(np.average(partition_labels, axis=0))
                # print(stats[-1])
        else:
            for i in tqdm(range(self.overall_teachers)):
                partition_data, partition_labels = partition_dataset(data_X, data_y, self.overall_teachers, i)
                train_data_list.append(partition_data)
                train_label_list.append(partition_labels)

        self.train_size = len(train_data_list[0])

        if self.train_size < self.batch_size:
            self.batch_size = self.train_size
            print('adjusted batch size:', self.batch_size)
            # raise Exception("[!] Entire dataset size (%d) is less than the configured batch_size (%d) " % (
            # self.train_size, self.batch_size))
        self.build_model()

        print("Training teacher models and student model together...")

        if not config.non_private:
            assert len(train_data_list) == self.overall_teachers
        else:
            print(str(len(train_data_list)))

        if self.pca:
            data = data_X.reshape([data_X.shape[0], -1])
            self.pca_components, rdp_budget = ComputeDPPrincipalProjection(
                data,
                self.pca_dim,
                self.orders,
                config.pca_sigma,
            )
            self.rdp_counter += rdp_budget

        d_optim_list = []

        for i in range(self.batch_teachers):
            d_optim_list.append(tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(
                self.teachers_list[i]['d_loss'], var_list=self.d_vars[i]))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss,
                                                                                            var_list=self.g_vars)

        if not config.pretrain:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
        else:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            self.load_pretrain(config.checkpoint_dir)

        if 'slt' in self.dataset_name:
            self.g_sum = merge_summary([self.z_sum, self.G_sum, self.g_loss_sum])
        else:
            self.g_sum = merge_summary([self.z_sum, self.g_loss_sum])

        self.d_sum_list = []

        for i in range(self.batch_teachers):
            teacher = self.teachers_list[i]
            if 'slt' in self.dataset_name:
                self.d_sum_list.append(
                    merge_summary([teacher['d_loss_sum'], teacher['g_loss_sum'], teacher['img_grads_sum']]))
            else:
                self.d_sum_list.append(merge_summary([teacher['d_loss_sum'], teacher['g_loss_sum']]))

        self.writer = SummaryWriter(os.path.join(config.checkpoint_dir, "logs"), self.sess.graph)

        counter = 0
        start_time = time.time()

        self.save_d(config.checkpoint_dir, 0, -1)
        for epoch in range(config.epoch):
            self.epoch_change = True
            self.epoch = epoch
            print("----------------epoch: %d --------------------" % epoch)
            print("-------------------train-teachers----------------")
            batch_idxs = int(self.train_size // self.batch_size)
            # The idex of each batch
            print("Train %d idxs" % batch_idxs)
            for idx in range(0, batch_idxs):

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                errD = 0
                # train teacher models in batches, teachers_batch: how many batches of teacher
                for batch_num in range(self.teachers_batch):
                    if self.teachers_batch > 1:
                        could_load, checkpoint_counter = self.load_d(config.checkpoint_dir, epoch=epoch,
                                                                     batch_num=batch_num)
                        if could_load:
                            counter = checkpoint_counter
                            print("load sucess_this_epoch")
                        else:
                            print('fail_1')
                            could_load, checkpoint_counter = self.load_d(config.checkpoint_dir, epoch=epoch - 1,
                                                                         batch_num=batch_num)
                            if could_load:
                                counter = checkpoint_counter
                                print("load sucess_previous_epoch")
                            else:
                                print('fail_2')
                                could_load, checkpoint_counter = self.load_d(config.checkpoint_dir, epoch=0,
                                                                             batch_num=-1)

                    # train each teacher in this batch, batch_teachers: how many teacher in a batch
                    for teacher_id in range(self.batch_teachers):
                        # print("Training teacher model %d" % teacher_id)
                        # data_X = self.data_X if config.non_private else self.train_data_list[teacher_id+batch_num*self.batch_teachers]
                        X = train_data_list[teacher_id + batch_num * self.batch_teachers]

                        batch_idx = range(idx * self.batch_size, (idx + 1) * self.batch_size)
                        batch_images = X[batch_idx]

                        for k in range(config.d_step if epoch > 0 or config.pretrain_teacher == 0 else config.pretrain_teacher):
                            if self.y is not None:
                                # data_y = self.data_y if config.non_private else self.train_label_list[teacher_id+batch_num*self.batch_teachers]
                                y = train_label_list[teacher_id + batch_num * self.batch_teachers]
                                # print(data_y.shape)
                                batch_labels = y[batch_idx]

                                _, summary_str = self.sess.run([d_optim_list[teacher_id], self.d_sum_list[teacher_id]],
                                                               feed_dict={
                                                                   self.inputs: batch_images,
                                                                   self.z: batch_z,
                                                                   self.y: batch_labels,
                                                               })

                                self.writer.add_summary(summary_str, epoch)

                                err = self.teachers_list[teacher_id]['d_loss'].eval({
                                    self.z: batch_z,
                                    self.inputs: batch_images,
                                    self.y: batch_labels,
                                })
                                # print(str(batch_num*self.batch_teachers + teacher_id) + "loss:"+str(err))
                                errD += err
                            else:
                                _, summary_str = self.sess.run([d_optim_list[teacher_id], self.d_sum_list[teacher_id]],
                                                               feed_dict={
                                                                   self.inputs: batch_images,
                                                                   self.z: batch_z,
                                                               })

                                self.writer.add_summary(summary_str, epoch)

                                err = self.teachers_list[teacher_id]['d_loss'].eval({
                                    self.z: batch_z,
                                    self.inputs: batch_images,
                                })
                                # print(str(batch_num * self.batch_teachers + teacher_id) + "d_loss:" + str(err))
                                errD += err

                    self.save_d(config.checkpoint_dir, epoch, batch_num)

                # print("------------------train-generator-------------------")
                for k in range(config.g_step):
                    errG = 0
                    img_grads_list = []
                    if self.y is not None:
                        batch_labels = self.get_random_labels(self.batch_size)
                        for batch_num in range(self.teachers_batch):
                            if self.teachers_batch > 1:
                                could_load, checkpoint_counter = self.load_d(config.checkpoint_dir, epoch=epoch,
                                                                             batch_num=batch_num)
                                if could_load:
                                    counter = checkpoint_counter
                                    print("load sucess")
                                else:
                                    print('fail')

                            for teacher_id in range(self.batch_teachers):
                                img_grads = self.sess.run(self.teachers_list[teacher_id]['img_grads'],
                                                          feed_dict={
                                                              self.z: batch_z,
                                                              self.y: batch_labels,
                                                          })
                                img_grads_list.append(img_grads)

                        old_img = self.sess.run(self.G, feed_dict={self.z: batch_z, self.y: batch_labels})

                    else:
                        for batch_num in range(self.teachers_batch):
                            if self.teachers_batch > 1:
                                could_load, checkpoint_counter = self.load_d(config.checkpoint_dir, epoch=epoch,
                                                                             batch_num=batch_num)
                                if could_load:
                                    counter = checkpoint_counter
                                    print("load sucess")
                                else:
                                    print('fail')

                            for teacher_id in range(self.batch_teachers):
                                img_grads = self.sess.run(self.teachers_list[teacher_id]['img_grads'],
                                                          feed_dict={
                                                              self.z: batch_z,
                                                          })
                                img_grads_list.append(img_grads)

                        old_img = self.sess.run(self.G, feed_dict={self.z: batch_z})

                    img_grads_agg_list = []
                    for j in range(self.batch_size):
                        thresh = config.thresh

                        if config.non_private:
                            img_grads_agg_tmp = self.non_private_aggregation([grads[j] for grads in img_grads_list],
                                                                             config)
                            rdp_budget = 0
                        elif config.increasing_dim:
                            img_grads_agg_tmp, rdp_budget = self.aggregate_results(
                                [grads[j] for grads in img_grads_list], config, thresh=thresh, epoch=epoch)
                        elif config.signsgd_dept:
                            img_grads_agg_tmp, rdp_budget, rdp_budget_dept = self.aggregate_results(
                                [grads[j] for grads in img_grads_list], config, thresh=thresh)
                            self.rdp_counter_dept += rdp_budget_dept
                        else:
                            img_grads_agg_tmp, rdp_budget = self.aggregate_results(
                                [grads[j] for grads in img_grads_list], config, thresh=thresh)

                        img_grads_agg_list.append(img_grads_agg_tmp)
                        self.rdp_counter += rdp_budget

                    img_grads_agg = np.asarray(img_grads_agg_list)
                    updated_img = old_img + img_grads_agg

                    if config.non_private:
                        eps = 0
                        order = 0
                    else:
                        # calculate privacy budget and break if exceeds threshold
                        eps, order = compute_eps_from_delta(self.orders, self.rdp_counter, config.dp_delta)
                        if config.signsgd_dept:
                            eps_dept, order_dept = compute_eps_from_delta(self.orders, self.rdp_counter_dept, config.dp_delta)

                        if eps > config.epsilon:
                            print("New budget (eps = %.2f) exceeds threshold of %.2f. Early break (eps = %.2f)." % (
                                eps, config.epsilon, self.dp_eps_list[-1]))

                            # save privacy budget
                            self.save(config.checkpoint_dir, counter)
                            np.savetxt(config.checkpoint_dir + "/dp_eps.txt", np.asarray(self.dp_eps_list), delimiter=",")
                            np.savetxt(config.checkpoint_dir + "/rdp_eps.txt", np.asarray(self.rdp_eps_list),
                                       delimiter=",")
                            np.savetxt(config.checkpoint_dir + "/rdp_order.txt", np.asarray(self.rdp_order_list),
                                       delimiter=",")
                            if config.signsgd_dept:
                                np.savetxt(config.checkpoint_dir + "/dept_dp_eps.txt", np.asarray(self.dp_eps_list_dept),
                                           delimiter=",")
                                np.savetxt(config.checkpoint_dir + "/dept_rdp_eps.txt", np.asarray(self.rdp_eps_list_dept),
                                           delimiter=",")
                                np.savetxt(config.checkpoint_dir + "/dept_rdp_order.txt", np.asarray(self.rdp_order_list_dept),
                                           delimiter=",")

                            gen_batch = 100000 // self.batch_size + 1
                            data = self.gen_data(gen_batch)
                            data = data[:100000]
                            import joblib
                            interval = 100000 // 10
                            for slice in range(10):
                                joblib.dump(data[slice * interval: (slice+1) * interval], config.checkpoint_dir + '/eps-%.2f.data' % self.dp_eps_list[-1] + f'-{slice}.pkl')
                            sys.exit()

                    self.dp_eps_list.append(eps)
                    self.rdp_order_list.append(order)
                    self.rdp_eps_list.append(self.rdp_counter)
                    if config.signsgd_dept:
                        self.dp_eps_list_dept.append(eps_dept)
                        self.rdp_order_list_dept.append(order_dept)
                        self.rdp_eps_list_dept.append(self.rdp_counter_dept)

                    # Update G network
                    if self.y is not None:
                        _, summary_str, errG2 = self.sess.run([g_optim, self.g_sum, self.g_loss],
                                                              feed_dict={
                                                                  self.z: batch_z,
                                                                  self.updated_img: updated_img,
                                                                  self.y: batch_labels,
                                                              })
                        self.writer.add_summary(summary_str, epoch)

                        errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.updated_img: updated_img,
                            self.y: batch_labels,
                        })
                    else:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                       feed_dict={
                                                           self.z: batch_z,
                                                           self.updated_img: updated_img,
                                                       })
                        self.writer.add_summary(summary_str, epoch)

                        errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.updated_img: updated_img,
                        })

                counter += 1
                print(
                    "Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, g_loss_before: %.8f, dp_eps: %.8f, rdp_order: %d" \
                    % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errD, errG, errG2, eps, order))
            # filename = 'epoch'+str(epoch)+'_errD'+str(errD)+'_errG'+str(errG)+'_teachers'+str(self.batch_teachers)+'f.csv'
            # if epoch % 4 == 0:
            print('----------------------generate sample----------------------')
            # data = self.gen_data(500)
            # output_dir = os.path.join(config.checkpoint_dir, self.sample_dir)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # filename = 'private.data_epoch_' + str(epoch) + '.pkl'
            # outfile = os.path.join(output_dir, filename)
            # mkdir(output_dir)
            # with open(outfile,'wb') as f:
            #     pickle.dump(data, f)

            filename = 'epoch' + str(epoch) + '_errD' + str(errD) + '_errG' + str(errG) + '_teachers' + str(
                self.batch_teachers) + 'f.csv'

            # save each epoch
            self.save(config.checkpoint_dir, counter)
            np.savetxt(config.checkpoint_dir + "/dp_eps.txt", np.asarray(self.dp_eps_list), delimiter=",")
            np.savetxt(config.checkpoint_dir + "/rdp_order.txt", np.asarray(self.rdp_order_list), delimiter=",")
            np.savetxt(config.checkpoint_dir + "/rdp_eps.txt", np.asarray(self.rdp_eps_list), delimiter=",")
            if config.signsgd_dept:
                np.savetxt(config.checkpoint_dir + "/dept_dp_eps.txt", np.asarray(self.dp_eps_list_dept),
                           delimiter=",")
                np.savetxt(config.checkpoint_dir + "/dept_rdp_eps.txt", np.asarray(self.rdp_eps_list_dept),
                           delimiter=",")
                np.savetxt(config.checkpoint_dir + "/dept_rdp_order.txt", np.asarray(self.rdp_order_list_dept),
                           delimiter=",")

            if config.save_epoch:
                floor_eps = math.floor(eps * 10) / 10.0
                if not self.save_dict[floor_eps]:
                    # get a checkpoint of low eps
                    self.save_dict[floor_eps] = True
                    from shutil import copytree
                    src_dir = os.path.join(config.checkpoint_dir, self.model_dir)
                    dst_dir = os.path.join(config.checkpoint_dir, str(floor_eps))
                    copytree(src_dir, dst_dir)

        #
        # save after training
        self.save(config.checkpoint_dir, counter)
        np.savetxt(config.checkpoint_dir + "/dp_eps.txt", np.asarray(self.dp_eps_list), delimiter=",")
        np.savetxt(config.checkpoint_dir + "/rdp_eps.txt", np.asarray(self.rdp_eps_list), delimiter=",")
        np.savetxt(config.checkpoint_dir + "/rdp_order.txt", np.asarray(self.rdp_order_list), delimiter=",")
        if config.signsgd_dept:
            np.savetxt(config.checkpoint_dir + "/dept_dp_eps.txt", np.asarray(self.dp_eps_list_dept),
                       delimiter=",")
            np.savetxt(config.checkpoint_dir + "/dept_rdp_eps.txt", np.asarray(self.rdp_eps_list_dept),
                       delimiter=",")
            np.savetxt(config.checkpoint_dir + "/dept_rdp_order.txt", np.asarray(self.rdp_order_list_dept),
                       delimiter=",")

        return self.dp_eps_list[-1], config.dp_delta

    def discriminator(self, image, y):
        if self.config.simple_gan:
            return self.simple_discriminator(image, y)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        if self.wgan:
            h1 = lrelu(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'))
        else:
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

        h1 = tf.reshape(h1, [self.batch_size, -1])
        h1 = concat([h1, y], 1)

        if self.wgan:
            h2 = lrelu(linear(h1, self.dfc_dim, 'd_h2_lin'))
        else:
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

    def simple_discriminator(self, image, y):
        # yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        image = tf.reshape(image, (-1, 20))
        x = concat([image, y], 1)

        h0 = tf.nn.relu(linear(x, 100, 'd_h0_lin'))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(linear(h0, 50, 'd_h1_lin'))
        h1 = concat([h1, y], 1)

        h2 = linear(h1, 1, 'd_h2_lin')

        return tf.nn.sigmoid(h2), h2

    def simple_generator(self, z, y):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            h0 = tf.nn.relu(linear(z, 200, 'g_h0_lin'))
            h0 = concat([h0, y], 1)

            h1 = tf.nn.relu(linear(h0, 50, 'g_h1_lin'))
            h1 = concat([h1, y], 1)

            h2 = linear(h1, 20, 'g_h2_lin')
            return tf.reshape(h2, (-1, 2, 2, 5))

    def generator(self, z, y):
        if self.config.simple_gan:
            return self.simple_generator(z, y)
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h / 2), max(int(s_h / 4), 1)
            s_w2, s_w4 = int(s_w / 2), max(int(s_w / 4), 1)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            if self.wgan:
                h0 = tf.nn.relu(linear(z, self.gfc_dim, 'g_h0_lin'))
            else:
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = concat([h0, y], 1)

            if self.wgan:
                h1 = tf.nn.relu(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'))
            else:
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            if self.wgan:
                h2 = tf.nn.relu(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'))
            else:
                h2 = tf.nn.relu(
                    self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            # if 'ae' in self.config.dataset:
            #     return deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')

            if self.config.tanh:
                return (1 + tf.nn.tanh(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))) / 2.
            else:
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def gen_data(self, n_batch, label=None):
        x_list = []
        y_list = []
        for i in range(n_batch):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            if self.y is not None:
                if label is None:
                    batch_labels = self.get_random_labels(self.batch_size)
                else:
                    batch_labels = np.zeros((self.batch_size, self.y_dim), dtype=np.float)
                    batch_labels[:, label] = 1.0
                outputs = self.sess.run(self.G,
                                        feed_dict={
                                            self.z: batch_z,
                                            self.y: batch_labels,
                                        })
                outputsX = outputs.reshape([self.batch_size, -1])
                outputsy = np.argmax(batch_labels, axis=1)
            else:
                outputs = self.sess.run(self.G,
                                        feed_dict={
                                            self.z: batch_z,
                                        })
                outputsX = outputs.reshape([self.batch_size, -1])
                outputsy = None

            x_list.append(outputsX)
            y_list.append(outputsy)

        x_list = np.vstack(x_list)
        if self.y is not None:
            y_list = np.concatenate(y_list)
        else:
            y_list = None
        return x_list, y_list

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def print_tensors_in_checkpoint(self, checkpoint_dir, ckpt_name):
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        import os
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
        print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

    def load_pretrain(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        print(checkpoint_dir)
        save_vars_dict = {x.name[:-2]: x for x in self.save_vars if x.name.startswith('generator')}
        pretrain_saver = tf.train.Saver(max_to_keep=5, var_list=save_vars_dict)
        print(self.dataset_name)
        if 'cifar' in self.dataset_name or 'cinic' in self.dataset_name:
            ckpt_name = 'DCGAN.model-100'
        elif 'mnist' in self.dataset_name:
            ckpt_name = 'CIFAR.model-250'
        elif 'celebA' in self.dataset_name:
            ckpt_name = 'CIFAR.model-99'
        pretrain_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        import re
        if self.config.load_d:
            for i in range(self.batch_teachers):
                print('loading teacher {}'.format(i))
                save_vars_dict = {re.sub(r'teacher[0-9]+', 'teacher0', x.name[:-2]): x for x in self.save_vars if
                                  x.name.startswith('teacher{}/'.format(i))}
                pretrain_saver = tf.train.Saver(max_to_keep=5, var_list=save_vars_dict)
                pretrain_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

        # save_vars_dict = {x.name: x for x in self.save_vars}
        # print(save_vars_dict.keys())
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        # current_scope = tf.contrib.framework.get_name_scope()
        # with tf.variable_scope(current_scope, reuse=True):
        #     biases = tf.get_variable("teacher0/d_h0_conv/biases")
        #     biases2 = tf.get_variable("teacher12/d_h0_conv/biases")
        #     biases3 = tf.get_variable("generator/g_h0_lin/Matrix")
        #     biases = tf.Print(biases, [biases, biases2, biases3])
        #     self.sess.run(biases)
        return True, counter

    def load(self, checkpoint_dir, ckpt_name):
        import re
        print(" [*] Reading checkpoints...")
        print(checkpoint_dir)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter


    def load_d(self, checkpoint_dir, batch_num, epoch):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        model_name = "DCGAN_batch_" + str(batch_num) + "_epoch-" + str(epoch)

        ckpt = os.path.join(checkpoint_dir, model_name)
        print(ckpt + ".meta")
        if os.path.isfile(ckpt + ".meta"):
            # model_name = "DCGAN_batch_" + str(batch_num) + "_epoch_" + str(epoch)
            # print(model_name)
            self.saver_d.restore(self.sess, ckpt)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", model_name)).group(0))
            print(" [*] Success to read {}".format(model_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        model_name = "CIFAR.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def save_d(self, checkpoint_dir, step, teacher_batch):
        model_name = "DCGAN_batch_" + str(teacher_batch) + "_epoch"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver_d.save(self.sess,
                          os.path.join(checkpoint_dir, model_name),
                          global_step=step)
        print("-------------save-dis----------------------")

    def save_g(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver_g.save(self.sess,
                          os.path.join(checkpoint_dir, model_name),
                          global_step=step)
