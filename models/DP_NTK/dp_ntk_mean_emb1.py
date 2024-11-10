import random
import torch as pt
# from autodp import privacy_calibrator


def calc_mean_emb1(model_ntk, sensitive_dataloader, n_classes, noise_factor, device, cond=True):

    """ initialize the variables"""

    mean_v_samp = pt.Tensor([]).to(device)
    for p in model_ntk.parameters():
        mean_v_samp = pt.cat((mean_v_samp, p.flatten()))
    d = len(mean_v_samp)
    mean_emb1 = pt.zeros((d, n_classes), device=device)
    print('Feature Length:', d)

    n_data = 0
    for data, labels in sensitive_dataloader:
        if len(labels.shape) == 2:
            data = data.to(pt.float32) / 255.
            labels = pt.argmax(labels, dim=1)
        if cond:
            labels = labels % n_classes
        else:
            labels = pt.zeros_like(labels)
        data, y_train = data.to(device), labels.to(device)
        for i in range(data.shape[0]):
            """ manually set the weight if needed """
            # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[y_train[i],:][None,:])

            mean_v_samp = pt.Tensor([]).to(device)  # sample mean vector init
            f_x = model_ntk(data[i])

            """ get NTK features """
            f_idx_grad = pt.autograd.grad(f_x, model_ntk.parameters(),
                                          grad_outputs=f_x.data.new(f_x.shape).fill_(1))
            for g in f_idx_grad:
                mean_v_samp = pt.cat((mean_v_samp, g.flatten()))
            # mean_v_samp = mean_v_samp[:-1]

            """ normalize the sample mean vector """
            mean_emb1[:, y_train[i]] += mean_v_samp / pt.linalg.vector_norm(mean_v_samp)
        n_data += data.shape[0]
        break

    """ average by class count """
    mean_emb1 = pt.div(mean_emb1, n_data)

    """ adding DP noise to sensitive data """
    noise = pt.randn_like(mean_emb1)
    noise = noise * noise_factor * 2 / n_data
    noise_mean_emb1 = mean_emb1 + noise

    return noise_mean_emb1
