import torch
from vlm_eval.attacks.utils import project_perturbation, normalize_grad


class PGD:
    """
    Minimize or maximize given loss
    """

    def __init__(self, forward, norm, eps, mode='min', mask_out='context', image_space=True):
        self.model = forward

        self.norm = norm
        self.eps = eps
        self.momentum = 0.9

        self.mode = mode
        self.mask_out = mask_out
        self.image_space = image_space

    def perturb(self, data_clean, iterations, stepsize, perturbation=None, verbose=False, return_loss=False):
        if self.image_space:
            # make sure data is in image space
            assert torch.max(data_clean) < 1. + 1e-6 and torch.min(data_clean) > -1e-6 # todo

        if perturbation is None:
            perturbation = torch.zeros_like(data_clean, requires_grad=True)
        mask = self._set_mask(data_clean)
        velocity = torch.zeros_like(data_clean)
        for i in range(iterations):
            perturbation.requires_grad_()
            with torch.enable_grad():
                loss = self.model(data_clean + perturbation)
                # print 10 times in total and last iteration
                if verbose and (i % (iterations // 10 + 1) == 0 or i == iterations - 1):
                    print(f'[iteration] {i} [loss] {loss.item()}')

            with torch.no_grad():
                gradient = torch.autograd.grad(loss, perturbation)[0]
                gradient = mask * gradient
                if gradient.isnan().any():  #
                    print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                    gradient[gradient.isnan()] = 0.
                # normalize
                gradient = normalize_grad(gradient, p=self.norm)
                # momentum
                velocity = self.momentum * velocity + gradient
                velocity = normalize_grad(velocity, p=self.norm)
                # update
                if self.mode == 'min':
                    perturbation = perturbation - stepsize * velocity
                elif self.mode == 'max':
                    perturbation = perturbation + stepsize * velocity
                else:
                    raise ValueError(f'Unknown mode: {self.mode}')
                # project
                perturbation = project_perturbation(perturbation, self.eps, self.norm)
                if self.image_space:
                    perturbation = torch.clamp(
                        data_clean + perturbation, 0, 1
                    ) - data_clean  # clamp to image space
                    assert torch.max(data_clean + perturbation) < 1. + 1e-6 and torch.min(
                        data_clean + perturbation
                    ) > -1e-6
                assert not perturbation.isnan().any()

                # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
        # todo return best perturbation
        # problem is that model currently does not output expanded loss
        if return_loss:
            return data_clean + perturbation.detach(), loss
        else:
            return data_clean + perturbation.detach()

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
