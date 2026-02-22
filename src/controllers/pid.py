import torch

class DistributedPID:
    def __init__(self, N, Kp, Ki, Kd, I_max, alpha=0.9, device='cpu'):
        self.Kp = torch.full((N,), Kp, device=device)
        self.Ki = torch.full((N,), Ki, device=device)
        self.Kd = torch.full((N,), Kd, device=device)
        self.I_max = I_max
        self.alpha = alpha
        self.I = torch.zeros(N, device=device)
        self.e_prev = torch.zeros(N, device=device)
        self.edot_f = torch.zeros(N, device=device)
        self.device = device

    def reset(self):
        self.I.zero_()
        self.e_prev.zero_()
        self.edot_f.zero_()

    def compute(self, x, v, r, dt):
          e = r - x
          self.I = torch.clamp(self.I + e * dt, -self.I_max, self.I_max)
          edot = (e - self.e_prev) / dt
          self.edot_f = self.alpha * self.edot_f + (1 - self.alpha) * edot
          u_pid = self.Kp * e + self.Ki * self.I + self.Kd * self.edot_f
          self.e_prev = e.clone()
          return u_pid, e, self.I