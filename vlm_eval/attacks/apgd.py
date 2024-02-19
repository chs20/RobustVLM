import torch
import math


class APGD:
    def __init__(self, model, norm, eps, mask_out='context', initial_stepsize=None, decrease_every=None, decrease_every_max=None, random_init=False):
        # model returns loss sum over batch
        # thus currently only works with batch size 1
        # initial_stepsize: in terms of eps. called alpha in apgd
        # decrease_every: potentially decrease stepsize every x fraction of total iterations. default: 0.22
        self.model = model
        self.norm = norm
        self.eps = eps
        self.initial_stepsize = initial_stepsize
        self.decrease_every = decrease_every
        self.decrease_every_max = decrease_every_max
        self.random_init = random_init
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None

    def perturb(self, data_clean, iterations, pert_init=None, verbose=False):
        mask = self._set_mask(data_clean)
        data_adv, _, _ = apgd(
            self.model, data_clean, norm=self.norm, eps=self.eps, n_iter=iterations,
            use_rs=self.random_init, mask=mask, alpha=self.initial_stepsize,
            n_iter_2=self.decrease_every, n_iter_min=self.decrease_every_max, pert_init=pert_init,
            verbose=verbose
            )

        return data_adv

    def _set_mask(self, data):
        mask = torch.ones_like(data)
        if self.mask_out == 'context':
            mask[:, :-1, ...] = 0
        elif self.mask_out == 'query':
            mask[:, -1, ...] = 0
        elif isinstance(self.mask_out, int):
            mask[:, self.mask_out, ...] = 0
        elif self.mask_out is None:
            pass
        else:
            raise NotImplementedError(f'Unknown mask_out: {self.mask_out}')
        return mask

    def __str__(self):
        return 'APGD'


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

    inu = 2 * (indbs < u.shape[1]).float() - 1
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


def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)


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

def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()


def apgd(model, x, norm, eps, n_iter=10, use_rs=False, mask=None, alpha=None, n_iter_2=None,
               n_iter_min=None, pert_init=None, verbose=False, is_train=True):
    # from https://github.com/fra31/robust-finetuning
    assert x.shape[0] == 1  # only support batch size 1 for now
    norm = norm.replace('l', 'L')
    device = x.device
    ndims = len(x.shape) - 1

    if not use_rs:
        x_adv = x.clone()
    else:
        if norm == 'Linf':
            t = torch.zeros_like(x).uniform_(-eps, eps).detach()
            x_adv = x + t
        elif norm == 'L2':
            t = torch.randn(x.shape).to(device).detach()
            x_adv = x + eps * torch.ones_like(x).detach() * t / (L2_norm(t, keepdim=True) + 1e-12)
    if pert_init is not None:
        assert not use_rs
        assert pert_init.shape == x.shape, f'pert_init.shape: {pert_init.shape}, x.shape: {x.shape}'
        x_adv = x + pert_init

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)

    # set params
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2_frac = 0.22 if n_iter_2 is None else n_iter_2
        n_iter_min_frac = 0.06 if n_iter_min is None else n_iter_min
        n_iter_2 = max(int(n_iter_2_frac * n_iter), 1)
        n_iter_min = max(int(n_iter_min_frac * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2. if alpha is None else alpha
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old = n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1. if alpha is None else alpha

    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims],
                                         device=device)
    counter3 = 0

    x_adv.requires_grad_()
    # grad = torch.zeros_like(x)
    # for _ in range(self.eot_iter)
    with torch.enable_grad():
        loss_indiv = model(x_adv)#.unsqueeze(0)
        loss = loss_indiv.sum()
    # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    if mask is not None:
        grad *= mask
    # grad /= float(self.eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv = loss_indiv.detach()
    loss = loss.detach()

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
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                                                          x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            elif norm == 'L2':
                x_adv_1 = x_adv + step_size * grad / (L2_norm(grad,
                                                              keepdim=True) + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                                                                   keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                                                                                                      L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                                                                   keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                                                                                                      L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

            elif norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                            -1, 1, 1, 1) + 1e-10)

                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, eps)
                x_adv_1 = x + delta_u + delta_p

            elif norm == 'L0':
                L1normgrad = grad / (grad.abs().view(grad.shape[0], -1).sum(
                    dim=-1, keepdim=True) + 1e-12).view(grad.shape[0], *[1] * (
                        len(grad.shape) - 1))
                x_adv_1 = x_adv + step_size * L1normgrad * n_fts
                x_adv_1 = L0_projection(x_adv_1, x, eps)
                # TODO: add momentum

            x_adv = x_adv_1.to(dtype=x_adv.dtype) + 0.

        ### get gradient
        x_adv.requires_grad_()
        # grad = torch.zeros_like(x)
        # for _ in range(self.eot_iter)
        with torch.enable_grad():
            loss_indiv = model(x_adv)#.unsqueeze(0)
            loss = loss_indiv.sum()

        # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            if mask is not None:
                grad *= mask
        # grad /= float(self.eot_iter)
        x_adv.detach_()
        loss_indiv = loss_indiv.detach()
        loss = loss.detach()

        x_best_adv = x_adv + 0.
        if verbose and (i % max(n_iter // 10, 1) == 0 or i == n_iter - 1):
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean())
            print('iteration: {} - best loss: {:.6f} curr loss {:.6f} {}'.format(
                i, loss_best.sum(), loss_curr, str_stats))
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
                    fl_oscillation = check_oscillation(loss_steps, i, k,
                                                       loss_best, k3=thr_decr)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                                               fl_reduce_no_impr)
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

    return x_best, loss_best, x_best_adv


