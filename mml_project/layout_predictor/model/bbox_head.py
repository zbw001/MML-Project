import torch
import torch.nn as nn
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from .Decoder import CustomTransformerDecoder
from .Inference_PDFDecoder import greedy_PDF
import random

class GMM2D(nn.Module):
    GMM_PARAM_NUM = 6
    def __init__(self, 
                 hidden_size: int,
                 num_componenets: int, 
                 temperature: float,
                 X_Softmax=False, 
                 greedy: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.aug_size = 64
        self.num_componenets = num_componenets
        self.X_Sfotmax = X_Softmax
        self.greedy = greedy
        self.temperature = temperature

        self.fc = nn.Linear(self.hidden_size, self.num_componenets * self.GMM_PARAM_NUM)

    def forward(self, x):
        batch_size = x.shape[0]
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = self.get_gmm_params(self.fc(x))

        sample_xy = self.sample_box(pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy, 
                                    temp=self.xy_temperature,
                                    greedy=self.greedy).reshape(batch_size, -1, 2)
        # calculate pdf score for refine Encoder
        sample_x = sample_xy[:,:,0]
        sample_y = sample_xy[:,:,1]
        sample_x = sample_x.unsqueeze(2).repeat(1,1, self.gmm_comp_num)
        sample_y = sample_y.unsqueeze(2).repeat(1,1, self.gmm_comp_num)

        if self.X_Sfotmax:
            xy_pdf = self.batch_pdf(pi_xy, sample_x, sample_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, self.gmm_comp_num)
        else:
            xy_pdf = None
            
        return None, sample_xy, None, xy_gmm, xy_pdf

        
    def batch_pdf(self, pi_xy, x, y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num):
        '''
        Log loss proposed in sketch-RNN and Obj-GAN
        '''
        # all inputs have the same shape: (batch, gmm_comp_num)
        u_x = u_x.reshape(batch_size, -1, gmm_comp_num).cuda()
        u_y = u_y.reshape(batch_size, -1, gmm_comp_num).cuda()
        sigma_x = sigma_x.reshape(batch_size, -1, gmm_comp_num).cuda()
        sigma_y = sigma_y.reshape(batch_size, -1, gmm_comp_num).cuda()
        pi_xy = pi_xy.reshape(batch_size, -1, gmm_comp_num).cuda()
        rho_xy = rho_xy.reshape(batch_size, -1, gmm_comp_num).cuda()

        z_x = ((x-u_x)/sigma_x)**2
        z_y = ((y-u_y)/sigma_y)**2
        z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
        z = z_x + z_y - 2*rho_xy*z_xy
        a = -z/(2*(1-rho_xy**2))
        exp = torch.exp(a)

        # avoid 0 in denominator
        norm = torch.clamp(2*math.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-5)
        raw_pdf = pi_xy*exp/norm
        # avoid log(0)
        raw_pdf = torch.sum(raw_pdf, dim=2)
        return raw_pdf.detach()
    
    def get_gmm_params(self, gmm_params):
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(gmm_params, self.num_componenets, dim=2)

        pi = nn.Softmax(dim=-1)(pi).reshape(-1, self.num_componenets).detach().cpu()
        u_x = u_x.reshape(-1, self.num_componenets).detach().cpu()
        u_y = u_y.reshape(-1, self.num_componenets).detach().cpu()
        sigma_x = torch.exp(sigma_x).reshape(-1, self.num_componenets).detach().cpu()
        sigma_y = torch.exp(sigma_y).reshape(-1, self.num_componenets).detach().cpu()
        # Clamp to avoid singular covariance
        rho_xy = torch.tanh(rho_xy).clamp(min=-0.95, max=0.95).reshape(-1, self.gmm_comp_num).detach().cpu()

        return (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)

    def sample_box(self, pi, u_x, u_y, sigma_x, sigma_y, rho_xy, temp = None, greedy=False):
        temperature = temp
        
        def adjust_temp(pi_pdf):
            pi_pdf = torch.log(pi_pdf)/temperature
            pi_pdf -= torch.max(pi_pdf)
            pi_pdf = torch.exp(pi_pdf)
            pi_pdf /= torch.sum(pi_pdf)
            return pi_pdf

        # get mixture indice:
        if temp is not None:
            pi = adjust_temp(pi)
            
        try:
            pi_idx = torch.multinomial(pi, 1)
        except:
            pi_idx = pi.argmax(1).unsqueeze(-1)
            
        # get mixture params:
        u_x = torch.gather(u_x, dim=1, index=pi_idx)
        u_y = torch.gather(u_y, dim=1, index=pi_idx)
#         if temp is not None:
#             sigma_x= torch.gather(sigma_x*temperature, dim=1, index=pi_idx)
#             sigma_y = torch.gather(sigma_y*temperature, dim=1, index=pi_idx)
#         else:
        sigma_x= torch.gather(sigma_x, dim=1, index=pi_idx)
        sigma_y = torch.gather(sigma_y, dim=1, index=pi_idx)
        rho_xy = torch.gather(rho_xy, dim=1, index=pi_idx)
        xy = self.sample_bivariate_normal(u_x, u_y, sigma_x, sigma_y, rho_xy, 
            temperature, greedy=greedy)
        return xy

    def sample_bivariate_normal(self, u_x, u_y, sigma_x, sigma_y, rho_xy, 
        temperature, greedy=False):
        # inputs must be floats
        if greedy:
            xy = torch.cat((u_x, u_y), dim=-1).cuda()
            return xy
        mean = torch.cat((u_x, u_y), dim=1)
        sigma_x *= math.sqrt(temperature)
        sigma_y *= math.sqrt(temperature)
        cov = torch.zeros((u_x.size(0), 2, 2)).cuda().detach().cpu()
        cov[:, 0, 0] = sigma_x.flatten() * sigma_x.flatten()
        cov[:, 0, 1] = rho_xy.flatten() * sigma_x.flatten() * sigma_y.flatten()
        cov[:, 1, 0] = rho_xy.flatten() * sigma_x.flatten() * sigma_y.flatten()
        cov[:, 1, 1] = sigma_y.flatten() * sigma_y.flatten()
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        singular_idx = (det == 0).nonzero()
        for idx in singular_idx:
            cov[idx] *= 0.
            cov[idx, 0, 0] += 1.
            cov[idx, 1, 1] += 1.
        m = MultivariateNormal(loc=mean, covariance_matrix=cov)
        x = m.sample()
        return x.cuda()
    
class PDFDecoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden_size=256, num_layers=2, attn_heads=2, dropout=0.1, cfg=None):
        
        super(PDFDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.global_feature = cfg['MODEL']['DECODER']['GLOBAL_FEATURE']
        self.aug_size = 64
        if self.global_feature:
            self.output_Layer = nn.Linear(hidden_size, hidden_size)
        else:
            self.output_Layer = nn.Linear(hidden_size+self.aug_size, hidden_size)
        self.decoder = CustomTransformerDecoder(hidden_size=hidden_size, 
                                          hidden_bb_size=self.aug_size,
                                          ff_size=hidden_size*4,
                                          num_layers=num_layers,num_heads=attn_heads, 
                                          dropout=dropout, emb_dropout=dropout)
        self.box_predictor = GMM_head(hidden_size, X_Softmax=cfg['MODEL']['REFINE']['X_Softmax'],
                                        greedy=cfg['MODEL']['DECODER']['GREEDY'],
                                        cfg=cfg)

    def random_sample(self, output_box, pred_box, sample_num):
        length = torch.arange(output_box.size(1))
        index_value = random.sample(list(enumerate(length)), sample_num)
        index = torch.Tensor(index_value)[:,0].long()
        mask = torch.zeros(output_box.size(),dtype=torch.bool)
        mask[:, index] = 1
        output_box[mask] = pred_box[mask]
        return output_box
        
    def forward(self, output_context, encoder_output, src_mask, trg_mask, epoch=0, is_train=True, global_mask=None):
        box_predictor_input = self.output_Layer(encoder_output)
        _, sample_xy, _, xy_gmm, xy_pdf = self.box_predictor(box_predictor_input)

        return box_predictor_input, None, sample_xy, None, xy_gmm, xy_pdf

class Coord2DHead(nn.Module):
    def __init__(self, hidden_size, dropout, cfg=None):
        super().__init__()

        self.cfg = cfg
        self.Decoder = PDFDecoder(hidden_size=hidden_size, num_layers=2, 
                                             attn_heads=2, dropout=dropout, cfg=cfg)

    def forward(self, epoch, encoder_output, mask, trg_mask, global_mask):
        # Decoder Forward
        _, _, coarse_xy, _, coarse_xy_gmm, _ = self.Decoder(encoder_output[:,:-1,:], encoder_output, mask, trg_mask, epoch, global_mask=global_mask)
        return coarse_xy, coarse_xy_gmm, None, None
    
    def inference(self, encoder_output, mask, trg_mask, global_mask):
        # Decoder Module
        decoder_output, coarse_wh, coarse_xy,  coarse_wh_gmm, coarse_xy_gmm, xy_pdf_score = greedy_PDF(encoder_hidden=None, encoder_output=encoder_output, \
            src_mask=mask, bos_index=self.bos_index, eos_index=self.eos_index, \
            decoder=self.Decoder, max_output_length=128, global_mask=global_mask)

        return coarse_xy, coarse_xy_gmm, None, None