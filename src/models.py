import torch
import torch.nn as nn
from modules.XLNet import XLNetForSequenceClassification


class DEMSAModel(nn.Module):
    def __init__(self, args):
        super(DEMSAModel, self).__init__()
        self.text_model = XLNetForSequenceClassification.from_pretrained(args.model, num_labels=1)
        audio_in, video_in = args.AUDIO_DIM, args.VISION_DIM  
        self.audio_model = AuViSubNet(audio_in, H_A_DIM, Output_A_DIM, num_layers=1, dropout=args.dropout)
        self.video_model = AuViSubNet(video_in, H_V_DIM, Output_V_DIM, num_layers=1, dropout=args.dropout)

        self.t_diffusion = MultiModalFeatureDenoiser(hidden_T, H_T, L_T,  Output_T_DIM)
        self.v_diffusion = MultiModalFeatureDenoiser(H_V_DIM,  H_V, L_V,  Output_A_DIM)
        self.a_diffusion = MultiModalFeatureDenoiser(H_A_DIM,  H_A, L_A,  Output_V_DIM)

        self.t_Reconstruction = ReconstructionNetwork(hidden_T,H_T, L_T, Output_T_DIM)
        self.v_Reconstruction = ReconstructionNetwork(H_V_DIM, H_V, L_V, Output_A_DIM)
        self.a_Reconstruction = ReconstructionNetwork(H_A_DIM, H_A, L_A, Output_V_DIM)


    def forward(self, input_ids, video, audio, segment_ids, input_mask):
        text = self.text_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[:, 0, :]
        audio = self.audio_model(audio)
        video = self.video_model(video)
        fusion_h = torch.cat([text, audio, video], dim=-1)

        denoised_visual = self.v_diffusion(video)
        denoised_audio = self.a_diffusion(audio)
        denoised_text = self.t_diffusion(text)

        denoised_fusion = torch.cat([denoised_text, denoised_audio, denoised_visual], dim=-1)
        return fusion_h, denoised_fusion


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(AuViSubNet, self).__init__()
        self.rnn = nn.GRU(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(x)
        h = self.dropout(final_states.squeeze())
        y_1 = self.linear_1(h)
        return y_1


class ReconstructionNetwork(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dim_feedforward):
        super(ReconstructionNetwork, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(feature_dim, num_heads, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        transformed = self.transformer_encoder(x)
        return self.linear_out(transformed)


class TransformerDenoiser(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dim_feedforward):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(feature_dim, num_heads, dim_feedforward, dropout=0.1,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        transformed = self.transformer_encoder(x)
        return self.linear_out(transformed)


class MultiModalFeatureDenoiser(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dim_feedforward, n_timestep=1000):
        super().__init__()
        self.denoiser = TransformerDenoiser(feature_dim, num_heads, num_layers, dim_feedforward)
        self.n_timestep = n_timestep
        self.betas = torch.linspace(0.0001, 0.02, n_timestep)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - self.betas, axis=0))

    def forward(self, features):
        noisy_features = self.ddim_noise(features)
        denoised_visual = self.denoiser(noisy_features)
        return denoised_visual

    def ddim_noise(self, features):
        # DDIM加噪过程
        noisy_features = features.clone()
        for time_step in range(1, self.n_timestep):
            alpha = self.alphas_cumprod[time_step]
            noise = torch.randn_like(features) * torch.sqrt(1.0 - alpha)
            noisy_features = torch.sqrt(alpha) * noisy_features + noise
        return noisy_features
