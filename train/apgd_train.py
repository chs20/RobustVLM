
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# from fra31/robust-finetuning

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 = eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    # u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2* (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    # print(s[0])

    # print(c5.shape, c2)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
             x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


# criterion_dict = {
#     'ce': lambda x, y: F.cross_entropy(x, y, reduction='none'),
#     'dlr': dlr_loss, 'dlr-targeted': dlr_loss_targeted
# }


def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()


def apgd_train(model, x, y, norm, eps, n_iter=10, use_rs=False, loss_fn=None,
               verbose=False, is_train=True, initial_stepsize=None):
    assert not model.training
    norm = norm.replace('linf', 'Linf').replace('l2', 'L2')
    device = x.device
    ndims = len(x.shape) - 1

    if not use_rs:
        x_adv = x.clone()
    else:
        raise NotImplemented
        if norm == 'Linf':
            t = torch.rand_like(x)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)

    # set loss
    # criterion_indiv = criterion_dict[loss]


    # set params
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old = n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1.

    if initial_stepsize:
        alpha = initial_stepsize / eps

    step_size = alpha * eps * torch.ones(
        [x.shape[0], *[1] * ndims],
        device=device
        )
    counter3 = 0

    x_adv.requires_grad_()
    # grad = torch.zeros_like(x)
    # for _ in range(self.eot_iter)
    # with torch.enable_grad()
    logits = model(x_adv, output_normalize=True)
    loss_indiv = loss_fn(logits, y)
    loss = loss_indiv.sum()
    # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    # grad /= float(self.eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()

    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()

    for i in range(n_iter):
        ### gradient step
        if True:  # with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach().mean()

            a = 0.75 if i > 0 else 1.0

            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv_1,
                            x - eps
                            ), x + eps
                        ), 0.0, 1.0
                    )
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                            x - eps
                        ), x + eps
                    ), 0.0, 1.0
                )

            elif norm == 'L2':
                x_adv_1 = x_adv + step_size * grad / (L2_norm(
                    grad,
                    keepdim=True
                    ) + 1e-12)
                x_adv_1 = torch.clamp(
                    x + (x_adv_1 - x) / (L2_norm(
                        x_adv_1 - x,
                        keepdim=True
                        ) + 1e-12) * torch.min(
                        eps * torch.ones_like(x),
                        L2_norm(x_adv_1 - x, keepdim=True)
                        ), 0.0, 1.0
                    )
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(
                    x + (x_adv_1 - x) / (L2_norm(
                        x_adv_1 - x,
                        keepdim=True
                        ) + 1e-12) * torch.min(
                        eps * torch.ones_like(x),
                        L2_norm(x_adv_1 - x, keepdim=True)
                        ), 0.0, 1.0
                    )

            elif norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                            -1, 1, 1, 1
                        ) + 1e-10)

                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, eps)
                x_adv_1 = x + delta_u + delta_p

            elif norm == 'L0':
                L1normgrad = grad / (grad.abs().view(grad.shape[0], -1).sum(
                    dim=-1, keepdim=True
                ) + 1e-12).view(
                    grad.shape[0], *[1] * (
                            len(grad.shape) - 1)
                    )
                x_adv_1 = x_adv + step_size * L1normgrad * n_fts
                x_adv_1 = L0_projection(x_adv_1, x, eps)
                # TODO: add momentum

            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        # grad = torch.zeros_like(x)
        # for _ in range(self.eot_iter)
        # with torch.enable_grad()
        logits = model(x_adv, output_normalize=True)
        loss_indiv = loss_fn(logits, y)
        loss = loss_indiv.sum()

        # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        # grad /= float(self.eot_iter)
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()

        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        if verbose:
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts
            ) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean()
            )
            print(
                'iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats
                )
            )
            # print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))

        ### check step size
        if True:  # with torch.no_grad()
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1

            if counter3 == k:
                if norm in ['Linf', 'L2']:
                    fl_oscillation = check_oscillation(
                        loss_steps, i, k,
                        loss_best, k3=thr_decr
                        )
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(
                        fl_oscillation,
                        fl_reduce_no_impr
                        )
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    counter3 = 0
                    k = max(k - size_decr, n_iter_min)

                elif norm == 'L1':
                    # adjust sparsity
                    sp_curr = L0_norm(x_best - x)
                    fl_redtopk = (sp_curr / sp_old) < .95
                    topk = sp_curr / n_fts / 1.5
                    step_size[fl_redtopk] = alpha * eps
                    step_size[~fl_redtopk] /= adasp_redstep
                    step_size.clamp_(alpha * eps / adasp_minstep, alpha * eps)
                    sp_old = sp_curr.clone()

                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

    #return x_best, acc, loss_best, x_best_adv
    return x_best_adv




