import torch
from torch import nn


class Encoder(nn.Module):
    """MusicVAE Encoder"""
        
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=True):
        """Initialize class
     
        Parameters
        ----------
        input_size : dim of input sequence
        hidden_size : LSTM hidden size
        latent_dim : dim of latent z
        num_layers : the number of LSTM layers
        bidirectional : True or False
        """
            
        super(Encoder, self).__init__()
        
        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1
            
        self.hidden_size = hidden_size
        self.num_hidden = num_directions * num_layers
        self.final_size = self.num_hidden * hidden_size
        
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional)
        
        self.mu = nn.Linear(self.final_size, latent_dim)
        self.std = nn.Linear(self.final_size, latent_dim)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        
    def encode(self, x):
        """
        Parameters
        ----------
        x : input sequecne (batch, seq, feat)
        
        Returns
        -------
        z : latent z (batch, latent_dim)
        mu : mu (batch, latent_dim)
        std : std (batch, latent_dim)
        """
        
        x, (h, c) = self.lstm(x)
        h = h.transpose(0, 1).reshape(-1, self.final_size)
        
        mu = self.norm(self.mu(h))
        std = nn.Softplus()(self.std(h))
        
        # reparam
        z = self.reparameterize(mu, std)
        
        return z, mu, std
    
    def reparameterize(self, mu, std):
        """
        Parameters
        ----------
        mu : mu (batch, latent_dim)
        std : std (batch, latent_dim)
        
        Returns
        -------
        z : reparam latent z (batch, latent_dim)
        """
            
        eps = torch.randn_like(std)

        return mu + (eps * std)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : input sequence (batch, seq, feat)
        
        Returns
        -------
        z : reparam latent z (batch, latent_dim)
        mu : mu (batch, latent_dim)
        std : std (batch, latent_dim)
        """
            
        z, mu, std = self.encode(x)
        
        return z, mu, std
    

class Decoder(nn.Module):
    """MusicVAE Decoder"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=False):
        """Initialize class
     
        Parameters
        ----------
        input_size : dim of input sequence
        hidden_size : dim of LSTM hidden size
        output_size : dim of output sequence
        num_layers : the number of LSTM layers
        bidirectional : True or False
        """
            
        super(Decoder, self).__init__()
        
        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden = num_directions * num_layers

        self.logits= nn.Linear(hidden_size, output_size)
        self.decoder = nn.LSTM(batch_first=True,
                               input_size=input_size+output_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        
    def forward(self, x, h, c, temp=1):
        """
        Parameters
        ----------
        x : input sequence (batch, 1, feat)
        h : LSTM state (num_hidden, batch, hidden_size)
        c : LSTM cell (num_hidden, batch, hidden_size)
        temp : temperature of softmax
        
        Returns
        -------
        out : predicted label (batch, 1, output_size)
        prob: predicted prob (batch, 1, output_size)
        h : LSTM next state
        c : LSTM next cell
        """
        
        x, (h, c) = self.decoder(x, (h, c))
        logits = self.logits(x) / temp
        prob = nn.Softmax(dim=2)(logits)
        out = torch.argmax(prob, 2)
                
        return out, prob, h, c

    
class Conductor(nn.Module):
    """MusicVAE Conductor"""
    
    def __init__(self, input_size, hidden_size, device, num_layers=2, bidirectional=False, bar=4):
        """Initialize class
     
        Parameters
        ----------
        input_size : dim of input sequence
        hidden_size : dim of LSTM hidden size
        output_size : dim of output sequence
        num_layers : the number of LSTM layers
        bidirectional : True or False
        bar : the number of units in bar
        """
        
        super(Conductor, self).__init__()

        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1

        self.bar = bar
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_directions * num_layers
        
        self.norm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.conductor = nn.LSTM(batch_first=True,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional)
        
    def init_hidden(self, batch_size, z):
        h0 = z.repeat(self.num_hidden, 1, 1)
        c0 = z.repeat(self.num_hidden, 1, 1)

        return h0, c0
    
    def forward(self, z):
        """
        Parameters
        ----------
        z : latent z (batch, input_size)
        
        Returns
        -------
        feat : conductor feat (batch, bar_seq, hidden_size)
        """
            
        batch_size = z.shape[0]
        
        z = self.norm(z) # it is different from paper
        h, c = self.init_hidden(batch_size, z)
        z = z.unsqueeze(1)
        
        # initialize
        feat = torch.zeros(batch_size, self.bar, self.hidden_size, device=self.device)
        
        # conductor
        z_input = z
        for i in range(self.bar):
            z_input, (h, c) = self.conductor(z_input, (h, c))
            feat[:, i, :] = z_input.squeeze()
            z_input = z
            
        feat = self.linear(feat)
            
        return feat
    

class Hierarchical_Decoder(nn.Module):
    """MusicVAE Hierarchical Decoder"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=False):
        """Initialize class
     
        Parameters
        ----------
        input_size : dim of input sequence
        hidden_size : dim of LSTM hidden size
        output_size : dim of output sequence
        num_layers : the number of LSTM layers
        bidirectional : True or False
        """
            
        super(Hierarchical_Decoder, self).__init__()
        
        if bidirectional == True:
            num_directions = 2
        else:
            num_directions = 1
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden = num_directions * num_layers
        
        self.logits= nn.Linear(hidden_size, output_size)
        self.decoder = nn.LSTM(batch_first=True,
                               input_size=input_size+output_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional)
        
    def forward(self, x, h, c, z, temp=1):
        """
        Parameters
        ----------
        x : input sequence (batch, 1, feat)
        h : LSTM state (num_hidden, batch, hidden_size)
        c : LSTM cell (num_hidden, batch, hidden_size)
        z : concat feature
        temp : temperature of softmax
        
        Returns
        -------
        out : predicted label (batch, 1, output_size)
        prob: predicted prob (batch, 1, output_size)
        h : LSTM next state
        c : LSTM next cell
        """
            
        x = torch.cat((x, z.unsqueeze(1)), 2)
        
        x, (h, c) = self.decoder(x, (h, c))
        logits = self.logits(x) / temp
        prob = nn.Softmax(dim=2)(logits)
        out = torch.argmax(prob, 2)
                
        return out, prob, h, c