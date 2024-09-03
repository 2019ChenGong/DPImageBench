import numpy as np
import sys

from models.G_PATE.pate_core import *
from numpy import linalg as LA

EPS = sys.float_info.epsilon


# Algorithm 1 in 'Scalable Private Learning with PATE'
def gnmax_thresh_aggregator(counts, thresh_cnt, sigma_thresh, sigma, orders):
    log_pr_answered = compute_logpr_answered(thresh_cnt, sigma_thresh, counts)
    rdp_budget = compute_rdp_threshold(log_pr_answered, sigma_thresh, orders)
    # print("Threshold budget:" + str(rdp_budget))

    if np.random.normal(np.max(counts), sigma_thresh) >= thresh_cnt:
        logq = compute_logq_gaussian(counts, sigma)
        res = np.argmax(np.random.normal(counts, sigma))
        g_rdp_budget = rdp_gaussian(logq, sigma, orders)
        rdp_budget += g_rdp_budget
    else:
        # do not return result if teacher models do not agree
        res = -1

    return res, rdp_budget


def gnmax_aggregator(counts, sigma, orders):
    logq = compute_logq_gaussian(counts, sigma)
    dir_index = np.argmax(np.random.normal(counts, sigma))
    rdp_budget = rdp_gaussian(logq, sigma, orders)
    return dir_index, rdp_budget


def rdp_percentile(arr_list, q, orders, vmin, vmax, lmbd, axis=0):
    arr_length = len(arr_list)
    arr_size = arr_list[0].size
    input_shape = arr_list[0].shape
    arr_reshaped = np.vstack([arr.reshape([1, arr_size]) for arr in arr_list])

    arr_ordered = np.sort(arr_reshaped, axis=0)
    arr_ordered = arr_ordered.clip(min=vmin, max=vmax)

    arr_ordered_new = np.vstack([np.ones([1, arr_size]) * vmin, arr_ordered, np.ones([1, arr_size]) * vmax])
    arr_ordered_new[np.abs(arr_ordered_new) < sys.float_info.epsilon] = 0

    n_teachers, n_feature = arr_reshaped.shape
    arr_prob = np.zeros([n_teachers + 1, n_feature])

    for i in range(arr_length + 1):
        diff = arr_ordered_new[i + 1, :] - arr_ordered_new[i, :]
        diff = diff.clip(min=0)
        arr_prob[i] = diff * np.exp(-0.5 / lmbd * abs(i - q / 100 * arr_length))
        # arr_prob[i] = np.exp(np.log(diff) - 0.5/lmbd * abs(i - q/100 * arr_length))

    # arr_prob = normalize(arr_prob, norm='l1', axis=0)

    if np.min(arr_prob) < 0:
        print(arr_prob)
        exit()

    low = np.zeros([1, arr_size])
    high = np.zeros([1, arr_size])

    for i in range(arr_size):
        prob = arr_prob[:, i] / np.sum(arr_prob[:, i])
        rindex = np.random.choice(arr_length + 1, p=prob)
        # print(rindex)

        low[0, i] = arr_ordered_new[rindex, i]
        high[0, i] = arr_ordered_new[rindex + 1, i]

    output_q = np.random.uniform(low=low, high=high, size=[1, arr_size])
    output_q = output_q.reshape(input_shape)

    rdp_budget = arr_size * np.multiply(
        1 / (orders - 1),
        np.log(
            np.multiply(np.divide(orders, 2 * orders - 1), np.exp((orders - 1) / lmbd)) \
            + np.multiply(np.divide(orders - 1, 2 * orders - 1), np.exp(-orders / lmbd))
        )
    )

    return output_q, rdp_budget


def rdp_winsorized_mean(arr_list, step_size, sigma_mean, sigma_percentile, orders, pca_mat=None):
    vmin = -step_size
    vmax = step_size

    flatten_arr = np.asarray([arr.flatten() for arr in arr_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    q25, q25_budget = rdp_percentile(flatten_arr, 25, orders, vmin=vmin, vmax=vmax, lmbd=sigma_percentile)
    q75, q75_budget = rdp_percentile(flatten_arr, 75, orders, vmin=vmin, vmax=vmax, lmbd=sigma_percentile)

    arr_mean = np.mean(flatten_arr.clip(min=q25, max=q75), axis=0)

    arr_mean[np.sign(q75) != np.sign(q25)] = 0

    # when 75 percentile is smaller, update the model with the average of 75 and 25 percentile
    # quantile_mean = (q75 + q25) / 2
    arr_mean[q75 < q25] = 0

    update_index = np.nonzero(np.logical_and(np.sign(q75) == np.sign(q25), q75 > q25))
    q_range = q75 - q25

    sensitivity = LA.norm(q_range[update_index] / len(arr_list))

    gaussian_noise, mean_budget = gaussian_rdp(arr_mean[update_index], sensitivity, orders, sigma_mean)
    arr_mean[update_index] += gaussian_noise
    arr_mean[update_index] = arr_mean[update_index].clip(min=q25[update_index], max=q75[update_index])

    # for testing only
    # update_ratio = gaussian_noise.size / arr_mean.size
    # print("Update ratio: %.8f, norm: %.8f" % (update_ratio, sensitivity))

    rdp_budget = q25_budget + q75_budget + mean_budget

    if pca_mat is not None:
        # project res direction back to original axis
        arr_mean = np.matmul(arr_mean, np.transpose(pca_mat))

    return arr_mean.reshape(arr_list[0].shape), rdp_budget


def gradient_voting_nonprivate(output_list, step_size, nbins=10):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    flatten_arr = flatten_arr.clip(min=-step_size, max=step_size)

    bins = np.arange(-step_size, step_size, (step_size * 2 / nbins))
    bins = np.hstack([bins, step_size])
    result = np.zeros([1, n_features])

    for i in range(n_features):
        votes_arr, _ = np.histogram(flatten_arr[:, i], bins)
        res_idx = np.argmax(votes_arr)
        result[:, i] = (bins[res_idx] + bins[res_idx + 1]) / 2

    return result.reshape(output_list[0].shape)


def gradient_voting_rdp(output_list, step_size, sigma, sigma_thresh, orders, pca_mat=None, nbins=10, thresh=0.9):
    import time
    st = time.time()
    n = len(output_list)
    use_gpu = False  # turn it on if you are running a huge matrix and the bottleneck lies on CPU matmul
    if use_gpu:
        # have to use torch==1.2.0 and torchvision==0.4.0 to run tensorflow-gpu==1.4.0
        import torch
        flatten_arr = torch.tensor([arr.flatten() for arr in output_list], device='cuda:0')
    else:
        flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        if use_gpu:
            pca_mat_tensor = torch.from_numpy(pca_mat).float().to('cuda:0')
            flatten_arr = torch.matmul(flatten_arr, pca_mat_tensor)
            flatten_arr = flatten_arr.cpu().numpy()
        else:
            flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    flatten_arr = flatten_arr.clip(min=-step_size, max=step_size)

    bins = np.arange(-step_size, step_size, (step_size * 2 / nbins))
    bins = np.hstack([bins, step_size])
    result = np.zeros([1, n_features])

    rdp_budget = 0
    skipped_cnt = 0
    for i in range(n_features):
        votes_arr, _ = np.histogram(flatten_arr[:, i], bins)
        print(votes_arr)
        res_idx, cur_budget = gnmax_thresh_aggregator(votes_arr, thresh * n_teachers, sigma_thresh, sigma, orders)
        rdp_budget += cur_budget
        if res_idx < 0:
            skipped_cnt += 1
        else:
            result[:, i] = (bins[res_idx] + bins[res_idx + 1]) / 2
    print("Skipped %d feaatures out of %d" % (skipped_cnt, n_features))


    if pca_mat is not None:
        # project res direction back to original axis
        result = np.matmul(result, np.transpose(pca_mat))
    return result.reshape(output_list[0].shape), rdp_budget


def gradient_voting_rdp_multiproj(output_list, step_size, sigma, sigma_thresh, orders, pca_mats=None, nbins=10, thresh=0.9):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape
    print("flatten arr shape", flatten_arr.shape)

    if pca_mats is not None:
        # project to principal components
        split_flatten_arr = np.split(flatten_arr, len(pca_mats), axis=1)
        reduced_flatten_arr = []
        for pca_mat, arr in zip(pca_mats, split_flatten_arr):
            print("arr shape", arr.shape)
            print("pca shape", pca_mat.shape)
            arr = np.matmul(arr, pca_mat)
            reduced_flatten_arr.append(arr)
        flatten_arr = np.concatenate(reduced_flatten_arr, axis=1)
        n_features = flatten_arr.shape[1]

    flatten_arr = flatten_arr.clip(min=-step_size, max=step_size)

    bins = np.arange(-step_size, step_size, (step_size * 2 / nbins))
    bins = np.hstack([bins, step_size])
    result = np.zeros([1, n_features])

    rdp_budget = 0
    skipped_cnt = 0
    for i in range(n_features):
        votes_arr, _ = np.histogram(flatten_arr[:, i], bins)
        print(votes_arr)
        res_idx, cur_budget = gnmax_thresh_aggregator(votes_arr, thresh * n_teachers, sigma_thresh, sigma, orders)
        rdp_budget += cur_budget
        if res_idx < 0:
            skipped_cnt += 1
        else:
            result[:, i] = (bins[res_idx] + bins[res_idx + 1]) / 2

    print("Skipped %d feaatures out of %d" % (skipped_cnt, n_features))

    if pca_mat is not None:
        # project res direction back to original axis
        split_results = np.split(result, len(pca_mats), axis=1)
        final_results = []
        for split_result, pca_mat in zip(split_results, pca_mats):
            final_results.append(np.matmul(split_result, np.transpose(pca_mat)))
        final_results = np.concatenate(final_results, axis=1)
    return final_results.reshape(output_list[0].shape), rdp_budget


def gradient_sign_rdp(output_list, step_size, sigma, sigma_thresh, orders, pca_mat=None, thresh=0.9):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    # first line for positive votes, second line for negative votes
    votes_arr = np.zeros([2, n_features])
    votes_sign = np.sign(flatten_arr)
    # counts for positive votes
    votes_arr[0, :] = np.sum(votes_sign[votes_sign > 0], axis=0)
    # counts for negative votes 
    votes_arr[1, :] = -np.sum(votes_sign[votes_sign < 0], axis=0)

    res_dir = np.zeros([1, n_features])

    rdp_budget = 0
    skipped_cnt = 0
    for i in range(n_features):
        dir_index, cur_budget = gnmax_thresh_aggregator(votes_arr[:, i], thresh * n_teachers, sigma_thresh, sigma,
                                                        orders)
        if dir_index == 0:
            res_dir[0, i] = step_size
        elif dir_index == 1:
            res_dir[0, i] = -step_size
        else:
            skipped_cnt += 1
        rdp_budget += cur_budget

    print("Skipped %d feaatures out of %d" % (skipped_cnt, n_features))

    if pca_mat is not None:
        # project res direction back to original axis
        res_dir = np.matmul(res_dir, np.transpose(pca_mat))

    return res_dir.reshape(output_list[0].shape), rdp_budget


def gradient_rdp(output_list, step_size, sigma, orders, pca_mat=None, thresh=None, sigma_thresh=1):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    # first half votes for positive direction, second half votes for negative direction
    votes_arr = np.zeros([n_teachers, n_features * 2])
    max_index = np.argmax(np.abs(flatten_arr), axis=1)

    for i in range(n_teachers):
        if flatten_arr[i, max_index[i]] > 0:
            votes_arr[i, max_index[i]] = 1
        else:
            votes_arr[i, max_index[i] + n_features] = 1

    votes_count = np.sum(votes_arr, axis=0)

    if thresh is None:
        dir_index, rdp_budget = gnmax_aggregator(votes_count, sigma, orders)
    else:
        dir_index, rdp_budget = gnmax_thresh_aggregator(votes_count, thresh * n_teachers, sigma_thresh, sigma, orders)

    max_votes = np.max(votes_count)
    selected_votes = votes_count[dir_index]
    # print("Max cnt: %d, selected cnt: %d" % (max_votes, selected_votes))

    res_dir = np.zeros([1, n_features])

    if dir_index < n_features and dir_index >= 0:
        res_dir[0, dir_index] = step_size
    elif dir_index >= n_features:
        res_dir[0, dir_index - n_features] = -step_size
    else:
        print("Teachers don't agree. Skip...")

    if pca_mat is not None:
        # project res direction back to original axis
        res_dir = np.matmul(res_dir, np.transpose(pca_mat))

    return res_dir.reshape(output_list[0].shape), rdp_budget


def gaussian_rdp(arr, sensitivity, orders, sigma):
    gaussian_noise = np.random.normal(loc=np.zeros(arr.shape), scale=sigma * sensitivity, size=arr.shape)

    # Table 2 @ https://arxiv.org/pdf/1702.07476.pdf
    rdp_budget = [o / ((2 * sigma) ** 2) for o in orders]

    return gaussian_noise, rdp_budget
