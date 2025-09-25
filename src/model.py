import torch
import torch.nn as nn
import numpy as np

class LevelizedModel(nn.Module):
    """
    levelized DAG：
    - inputs: 8 (4-bit * 2)
    - layers: list of nodes per layer
    - each node has two fanins (w0,w1) parameters (scores)
    """
    def __init__(self, n_inputs=8, layers_config=[8, 8, 4]):
        super().__init__()
        self.n_inputs = n_inputs
        # build levelized structure
        # level 0: primary inputs (0..n_inputs-1)
        # levels 1..L: internal nodes
        self.layers = []
        self.candidates = [] # for each node store list of candidate indices (global index)
        self.param_w0 = nn.ParameterList()
        self.param_w1 = nn.ParameterList()
        idx_counter = n_inputs
        prev_level_nodes = list(range(n_inputs))
        for Lsize in layers_config:
            this_level = list(range(idx_counter, idx_counter+Lsize))
            # For level i, parameter vector length = 2 * M_i where M_i is total nodes in previous levels
            M_i = len(prev_level_nodes)  # number of nodes available for connection
            param_length = 2 * M_i  # according to paper: length 2M_i
            
            # candidates for each node = all nodes in prev levels
            for _ in range(Lsize):
                # Store all available previous nodes as candidates
                self.candidates.append(prev_level_nodes.copy())
                # Create parameters with correct length (2 * M_i)
                self.param_w0.append(nn.Parameter(torch.randn(param_length)))
                self.param_w1.append(nn.Parameter(torch.randn(param_length)))
            self.layers.append(this_level)
            idx_counter += Lsize
            prev_level_nodes = prev_level_nodes + this_level
        # outputs: connect last layer nodes to outputs (we map outputs from some nodes)
        # For 4x4 multiplier we expect 8-bit product; ensure we have 8 output nodes
        if len(self.layers[-1]) >= 8:
            self.output_nodes = self.layers[-1][:8]  # Take first 8 nodes from last layer
        else:
            self.output_nodes = self.layers[-1]  # Take all nodes if less than 8

    def forward(self, input_bits):
        # input_bits: tensor shape [B, n_inputs], values {0,1}
        if input_bits.shape[1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {input_bits.shape[1]}")
        
        B = input_bits.shape[0]
        device = input_bits.device
        node_vals = {}
        # primary inputs indexed 0..n_inputs-1
        for i in range(self.n_inputs):
            node_vals[i] = input_bits[:, i].float() * 2.0 - 1.0 # map 0->-1,1->+1
        softmax_b = []
        argmax_idx = []
        node_ab_values = {}  # Store ap0, ap1, bp0, bp1 for each node
        param_idx = 0
        # iterate layers
        for level_nodes in self.layers:
            for node in level_nodes:
                cand = self.candidates[param_idx]
                w0 = self.param_w0[param_idx]  # ωp,0 - full 2*M_i vector
                w1 = self.param_w1[param_idx]  # ωp,1 - full 2*M_i vector
                
                # According to paper: parameters have length 2*M_i, same as yi
                M_i = len(cand)
                
                # Collect candidate node values x = (x0, x1, ..., x_{M_i-1})
                candidate_values = []
                for cand_idx in cand:
                    candidate_values.append(node_vals[cand_idx])  # [B]
                x = torch.stack(candidate_values, dim=0)  # [M_i, B]
                
                # Construct yi according to paper definition:
                # yi = (yi_0, yi_1, ..., yi_{2M_i-1}) where:
                # yi_k = x_k for k < M_i (original values)
                # yi_k = -x_{k-M_i} for k >= M_i (negated values)
                yi = torch.cat([x, -x], dim=0)  # [2*M_i, B]
                
                # Apply paper formulas directly:
                # ap,0 = yi[argmax(ωp,0)] - select from yi using full ωp,0
                # bp,0 = yi · softmax(ωp,0) - weighted combination using full ωp,0
                # ap,1 = yi[argmax(ωp,1)] - select from yi using full ωp,1
                # bp,1 = yi · softmax(ωp,1) - weighted combination using full ωp,1
                
                # Compute softmax over full 2*M_i vectors
                softmax_w0 = torch.softmax(w0, dim=0)  # [2*M_i]
                softmax_w1 = torch.softmax(w1, dim=0)  # [2*M_i]
                
                # Compute argmax over full 2*M_i vectors
                argmax_w0 = int(torch.argmax(w0).item())
                argmax_w1 = int(torch.argmax(w1).item())
                
                # Apply formulas
                ap0 = yi[argmax_w0]  # [B] - selected from yi using ωp,0
                bp0 = torch.sum(yi * softmax_w0.unsqueeze(1), dim=0)  # [B] - weighted yi using ωp,0
                ap1 = yi[argmax_w1]  # [B] - selected from yi using ωp,1  
                bp1 = torch.sum(yi * softmax_w1.unsqueeze(1), dim=0)  # [B] - weighted yi using ωp,1
                
                # Store results
                softmax_b.append((softmax_w0, softmax_w1))
                argmax_idx.append((argmax_w0, argmax_w1))
                
                # Store ap0, ap1, bp0, bp1 values for this node
                node_ab_values[node] = {
                    'ap0': ap0,
                    'ap1': ap1,
                    'bp0': bp0,
                    'bp1': bp1
                }
                
                # Use ap0 and ap1 as the two fanins for the AND gate (according to paper)
                in0 = ap0  # First fanin: argmax selection from ωp,0
                in1 = ap1  # Second fanin: argmax selection from ωp,1
                
                # AND in ±1 domain: +1 if both +1 else -1
                out = torch.where((in0 > 0) & (in1 > 0), torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
                node_vals[node] = out
                
                # Update the stored values to include output
                node_ab_values[node]['out'] = out
                
                param_idx += 1
        # build outputs
        outputs = []
        for i, on in enumerate(self.output_nodes):
            if on in node_vals:
                outputs.append(node_vals[on].unsqueeze(1))
            else:
                # fallback: use zero output if node doesn't exist
                outputs.append(torch.zeros(B, 1, device=device))
        
        if outputs:
            outputs = torch.cat(outputs, dim=1)
        else:
            # fallback: return zero outputs
            outputs = torch.zeros(B, len(self.output_nodes), device=device)
            
        # map back to {0,1}
        outputs01 = (outputs > 0).long()
        return node_ab_values
    
    def export_selection_info(self):
        """Export the argmax indices for ap0 and ap1 per node, following paper specification"""
        res = []
        for i, (w0, w1) in enumerate(zip(self.param_w0, self.param_w1)):
            # Get argmax indices for ω₀ and ω₁ (these determine ap0 and ap1)
            ap0_idx = int(torch.argmax(w0).item())
            ap1_idx = int(torch.argmax(w1).item())
            
            # Note: bp0 and bp1 are computed via softmax, not argmax
            # They represent weighted combinations, not single candidate selections
            res.append((ap0_idx, ap1_idx))
        return res