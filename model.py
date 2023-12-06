import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, dim, F):
        super().__init__()
        
        self.F = F
        self.norm = nn.LayerNorm(dim)
        
    def forward(self,x):
        
        x = x + self.norm(self.F(x))
        
        return x


class Adapter(nn.Module): 
    def __init__(self, dim, factor, dropout):
        super().__init__()
        
        hidden_dim = int(dim * factor)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            )
        
    def forward(self, x):
        
        x = self.adapter(x)
        
        return x


class Block(nn.Module): 
    def __init__(self, input_size, num_classes, num_layers, time):
        super().__init__()
        
        self.time = time
        self.adapter_space = nn.Sequential(*
            [Residual(input_size[1], Adapter(input_size[1], factor=0.25, dropout=0.2))] * num_layers,
            )
        
        if self.time:
            self.adapter_time = nn.Sequential(*
                [Residual(input_size[0], Adapter(input_size[0], factor=0.25, dropout=0.2))] * num_layers,
                nn.AdaptiveAvgPool1d(1)
                )
            
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size[1], num_classes),
            )
        
    def forward(self, x):
        
        x = self.adapter_space(x)
        
        if self.time:
            x = x.permute(0, 2, 1)
            x = self.adapter_time(x)
        
        x = self.fc(x)
        
        return x


class Feature_Fusion_Network(nn.Module):
    def __init__(self, t_in, a_in, v_in, num_classes):
        super().__init__()
        
        self.textnet = Block(t_in, num_classes, num_layers=2, time=True)
        self.audionet = Block(a_in, num_classes, num_layers=2, time=True)
        self.visionnet = Block(v_in, num_classes, num_layers=2, time=False)
        
    def forward(self, text, audio, vision):
        
        x = self.textnet(text) + self.audionet(audio) + self.visionnet(vision)

        return x
