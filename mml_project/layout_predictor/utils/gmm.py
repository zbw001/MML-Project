import math
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class GMM2D:
    EPS = 1e-6
    SQRTEPS = math.sqrt(EPS)
    def __init__(self,
                 num_components: int,
                 params: torch.Tensor,
                 temperature: float = None,
                 greedy: bool = False):
        self.num_components = num_components
        assert params.shape[-1] == self.num_components * 6

        self._init_params(params)

        self.temperature = temperature
        self.greedy = greedy

    def copy(self):
        return GMM2D(
            num_components=self.num_components,
            params=self.params,
            temperature=self.temperature,
            greedy=self.greedy
        )

    def __getitem__(self, index):
        return GMM2D(
            num_components=self.num_components,
            params=self.params[index],
            temperature=self.temperature,
            greedy=self.greedy
        )

    def _init_params(self, params): 
        # params: (..., num_components * 6)
        self.params = params
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(params, self.num_components, dim=-1)
        # pi: (..., num_components), u_x: (..., num_components), ...
        self.pi = torch.softmax(pi, dim=-1)
        self.u_x, self.u_y = u_x, u_y
        self.sigma_x, self.sigma_y = torch.exp(sigma_x), torch.exp(sigma_y)
        self.rho_xy = torch.tanh(rho_xy).clamp(min=-0.95, max=0.95) # avoid singular covariance
        self.covariance_matrix = self._covariance_matrix()
    
    def _covariance_matrix(self):
        cov00 = self.sigma_x * self.sigma_x
        cov01 = self.rho_xy * self.sigma_x * self.sigma_y
        cov10 = self.rho_xy * self.sigma_x * self.sigma_y
        cov11 = self.sigma_y * self.sigma_y
        cov = torch.stack(
            [
                torch.stack([cov00, cov01], dim=-1),
                torch.stack([cov10, cov11], dim=-1)
            ], dim=-2
        )
        det = cov00 * cov11 - cov01 * cov10
        cov[det < self.EPS] += torch.tensor([[[self.SQRTEPS, 0.], [0., self.SQRTEPS]]], device=det.device)
        return cov

    @torch.no_grad()
    def sample(self):
        adjusted_pi = self.pi
        if self.temperature is not None:
            adjusted_pi = torch.log(adjusted_pi) / self.temperature
            adjusted_pi -= torch.max(adjusted_pi, dim=-1, keepdim=True)[0]
            adjusted_pi = torch.exp(adjusted_pi)
            adjusted_pi /= torch.sum(adjusted_pi, dim=-1, keepdim=True)
            
        try:
            pi_idx = torch.multinomial(adjusted_pi, 1)
        except:
            pi_idx = adjusted_pi.argmax(-1).unsqueeze(-1)
            
        u_x = torch.gather(self.u_x, dim=-1, index=pi_idx)[..., 0]
        u_y = torch.gather(self.u_y, dim=-1, index=pi_idx)[..., 0]

        if self.greedy:
            return torch.stack([u_x, u_y], dim=-1)
        
        cov = torch.gather(self.covariance_matrix, dim=-3, index=pi_idx)
        dist = MultivariateNormal(loc=torch.stack([u_x, u_y], dim=-1), covariance_matrix=cov)
        return dist.sample()

    def log_prob(self, input: torch.Tensor):
        '''
        Log loss proposed in sketch-RNN and Obj-GAN
        '''
        assert input.shape == self.u_x.shape[:-1] + (2,)
        x, y = input[..., 0], input[..., 1]
        z_x = ((x - self.u_x) / self.sigma_x) ** 2
        z_y = ((y - self.u_y) / self.sigma_y) ** 2
        z_xy = (x - self.u_x) * (y - self.u_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        a = - z / (2 * (1 - self.rho_xy ** 2))
        exp = torch.exp(a)

        # avoid 0 in denominator
        norm = torch.clamp(
            2 * math.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2), 
            min = 1e-5
        )
        raw_pdf = self.pi * exp / norm

        # avoid log(0)
        raw_pdf = torch.log(torch.sum(raw_pdf, dim=-1) + 1e-5)
        return raw_pdf

    