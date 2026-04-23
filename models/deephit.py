import torch
import torch.nn as nn
import torchtuples as tt

class SimpleMLP(torch.nn.Module):
    """Simple network structure for competing risks.
    """
    def __init__(self, in_features, num_nodes, num_risks, out_features, batch_norm=True,
                 dropout=None):
        super().__init__()
        self.num_risks = num_risks
        self.mlp = tt.practical.MLPVanilla(
            in_features, num_nodes, num_risks * out_features,
            batch_norm, dropout,
        )
        
    def forward(self, input):
        out = self.mlp(input)
        return out.view(out.size(0), self.num_risks, -1)
    
class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out