import torch

class DistributedPID:
    def __init__(self, N, Kp, Ki, Kd, I_max, alpha=0.9, device='cpu', u_sat=None):
        self.Kp = torch.full((N,), Kp, device=device)
        self.Ki = torch.full((N,), Ki, device=device)
        self.Kd = torch.full((N,), Kd, device=device)
        self.I_max = I_max
        self.u_sat = u_sat
        if u_sat is not None and u_sat <= 0:
            raise ValueError("u_sat must be positive or null")
        self.alpha = alpha
        self.I = torch.zeros(N, device=device)
        self.e_prev = torch.zeros(N, device=device)
        self.edot_f = torch.zeros(N, device=device)
        self.device = device

    def reset(self):
        self.I = torch.zeros_like(self.I)
        self.e_prev = torch.zeros_like(self.e_prev)
        self.edot_f = torch.zeros_like(self.edot_f)

    def compute(self, x, v, r, dt):
          e = r - x
          self.I = torch.clamp(self.I + e * dt, -self.I_max, self.I_max)
          edot = (e - self.e_prev) / dt
          self.edot_f = self.alpha * self.edot_f + (1 - self.alpha) * edot
          u_pid = self.Kp * e + self.Ki * self.I + self.Kd * self.edot_f
          self.e_prev = e.clone()
          return u_pid, e, self.I

    def limit(self, u):
        return u if self.u_sat is None else u.clamp(-self.u_sat, self.u_sat)
