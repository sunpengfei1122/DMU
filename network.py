import torch
import torch.nn as nn
import torch.nn.functional as F

class DMU(nn.Module):
    """
    RNN with learnable delay lines per timestep (tau=1).
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list[int] = [128，128],
        delay_dim: int = 20,      
        output_dim: int = 10,
        bidirectional: bool = False,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.delay_dim = delay_dim      #  delay line length n
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.use_cuda = use_cuda

        self.num_layers = len(self.hidden_dims)

        # Per-layer modules
        self.input_projections = nn.ModuleList()
        self.recurrent_projections = nn.ModuleList()
        self.delay_input_projections = nn.ModuleList()
        self.delay_recurrent_projections = nn.ModuleList()
        self.delay_line_gates = nn.ModuleList()
        self.activations = nn.ModuleList()

        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            # Project input to hidden
            self.input_projections.append(nn.Linear(current_dim, hidden_dim))
            # Recurrent hidden→hidden
            self.recurrent_projections.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            # Project input to delay state
            self.delay_input_projections.append(nn.Linear(current_dim, self.delay_dim))
            # Recurrent delay→delay
            self.delay_recurrent_projections.append(nn.Linear(self.delay_dim, self.delay_dim, bias=False))
            # Gate to distribute hidden over delay_size slots
            self.delay_line_gates.append(nn.Linear(1, self.delay_dim))
            # Activation function
            self.activations.append(nn.Tanh())

            current_dim = hidden_dim * (2 if self.bidirectional else 1)

        # Final classification layer
        self.classifier = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time_steps, input_dim)
        returns: (batch, time_steps, output_dim)
        """
        batch_size, time_steps, _ = x.size()

        for layer_idx in range(self.num_layers):
            hidden_dim = self.hidden_dims[layer_idx]

            # Initialize states
            h_t = torch.zeros(batch_size, hidden_dim, device=x.device)
            d_t = torch.zeros(batch_size, self.delay_dim, device=x.device)

            # Precompute feed-forward projections over all timesteps
            proj_input = self.input_projections[layer_idx](x)              # (B, T, H)
            proj_delay_input = self.delay_input_projections[layer_idx](x)  # (B, T, D)

            # Buffer to accumulate delayed contributions
            delay_buffer = torch.zeros(batch_size, hidden_dim, time_steps + self.delay_dim, device=x.device)

            outputs = []
            for t in range(time_steps):
                # recurrent updates
                a_t = proj_input[:, t] + self.recurrent_projections[layer_idx](h_t)
                d_update = proj_delay_input[:, t] + self.delay_recurrent_projections[layer_idx](d_t)

                # activations
                h_t = self.activations[layer_idx](a_t)
                d_t = self.activations[layer_idx](d_update)

                # compute softmax gate weights over delay slots
                gate_weights = F.softmax(d_t, dim=-1)  # (B, D)
                gate_weights = gate_weights.unsqueeze(1).expand(-1, hidden_dim, -1)  # (B, H, D)

                # scatter hidden state into delay_buffer
                delay_buffer[:, :, t : t + self.delay_dim] += h_t.unsqueeze(-1) * gate_weights

                # add delayed contribution back to h_t
                h_t = h_t + delay_buffer[:, :, t]

                outputs.append(h_t)

            # stack and prepare x for next layer
            x = torch.stack(outputs, dim=1)  # (B, T, H)

        # final classification per timestep
        logits = self.classifier(x)  # (B, T, output_dim)
        return logits
