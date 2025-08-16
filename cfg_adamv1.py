import os
import torch
import torch.distributed as dist


class CFGAdamV1(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        base_grad_alpha=0.8,
    ):
        defaults = dict(
            lr=lr,
            orig_lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            base_grad_alpha=base_grad_alpha,
        )
        super().__init__(params, defaults)

    def load_negative_gradient(self):
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                state['negative_grad'] = g.clone()
                del p.grad

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(g)
                    state['exp_avg_sq'] = torch.zeros_like(g)

                state['step'] += 1

                # update momentum with the base grad
                state['exp_avg'].lerp_(g, 1 - group['beta1'])

                # compute cfg grad
                cfg_grad = state['exp_avg'] - state['negative_grad']
                pre_norm_update = state['exp_avg'] * group['base_grad_alpha'] + cfg_grad * (1 - group['base_grad_alpha'])
                # pre_norm_update = state['exp_avg']

                # update variance
                state['exp_avg_sq'].lerp_(pre_norm_update.square(), 1 - group['beta2'])

                # the update
                update = pre_norm_update / (group['eps'] + state['exp_avg_sq'].sqrt())

                # bias correction
                bias_correction1 = 1 - group['beta1'] ** state['step']
                bias_correction2 = 1 - group['beta2'] ** state['step']
                scale = bias_correction1 / bias_correction2**0.5

                # apply weight decay and update
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # regular update
                p.data.add_(update, alpha=-(group['lr']) / scale)
