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
    def __init__(self, n_inputs=8, layers_config=[8, 8, 4], seed=42):
        super().__init__()
        
        # Set random seed for reproducible initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
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
        num_layers = len(layers_config)
        
        for layer_idx, Lsize in enumerate(layers_config):
            this_level = list(range(idx_counter, idx_counter+Lsize))
            # For level i, parameter vector length = 2 * M_i where M_i is total nodes in previous levels
            M_i = len(prev_level_nodes)  # number of nodes available for connection
            param_length = 2 * M_i  # according to paper: length 2M_i
            
            is_last_layer = (layer_idx == num_layers - 1)
            
            # candidates for each node = all nodes in prev levels
            for _ in range(Lsize):
                # Store all available previous nodes as candidates
                self.candidates.append(prev_level_nodes.copy())
                # Create parameters with correct length (2 * M_i)
                self.param_w0.append(nn.Parameter(torch.randn(param_length)))
                
                # For the last layer, w1 is not used (only w0), so set it to zeros
                if is_last_layer:
                    w1_param = nn.Parameter(torch.zeros(param_length))
                    w1_param.requires_grad = False  # Freeze w1 for output layer
                    self.param_w1.append(w1_param)
                else:
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
        for layer_idx, level_nodes in enumerate(self.layers):
            is_last_layer = (layer_idx == len(self.layers) - 1)
            num_nodes_layer = len(level_nodes)

            # Candidates are identical for every node in the same layer
            cand = self.candidates[param_idx]
            M_i = len(cand)

            # Collect candidate node values once per layer
            x = torch.stack([node_vals[cand_idx] for cand_idx in cand], dim=0)  # [M_i, B]
            yi = torch.cat([x, -x], dim=0)  # [2*M_i, B]

            # Stack parameters for all nodes in this layer
            w0_layer = torch.stack([self.param_w0[param_idx + i] for i in range(num_nodes_layer)], dim=0)  # [P, 2*M_i]
            w1_layer = torch.stack([self.param_w1[param_idx + i] for i in range(num_nodes_layer)], dim=0)  # [P, 2*M_i]

            # Softmax over each node's weights
            softmax_w0 = torch.softmax(w0_layer, dim=1)  # [P, 2*M_i]
            softmax_w1 = torch.softmax(w1_layer, dim=1)  # [P, 2*M_i]

            # Argmax indices for selecting from yi
            argmax_w0 = torch.argmax(w0_layer, dim=1)  # [P]
            argmax_w1 = torch.argmax(w1_layer, dim=1)  # [P]

            # Apply formulas in batch
            ap0 = yi[argmax_w0]  # [P, B]
            ap1 = yi[argmax_w1]  # [P, B]
            yi_expanded = yi.unsqueeze(0)  # [1, 2*M_i, B]
            bp0 = torch.sum(yi_expanded * softmax_w0.unsqueeze(-1), dim=1)  # [P, B]
            bp1 = torch.sum(yi_expanded * softmax_w1.unsqueeze(-1), dim=1)  # [P, B]

            if is_last_layer:
                out_layer = ap0
            else:
                pos_mask = (ap0 > 0) & (ap1 > 0)
                out_layer = torch.where(pos_mask, torch.ones_like(ap0), -torch.ones_like(ap0))

            # Store per-node results back into dicts (keeps interface unchanged)
            for local_idx, node in enumerate(level_nodes):
                softmax_b.append((softmax_w0[local_idx], softmax_w1[local_idx]))
                argmax_idx.append((int(argmax_w0[local_idx].item()), int(argmax_w1[local_idx].item())))

                node_ab_values[node] = {
                    'ap0': ap0[local_idx],
                    'ap1': ap1[local_idx],
                    'bp0': bp0[local_idx],
                    'bp1': bp1[local_idx],
                    'out': out_layer[local_idx]
                }
                node_vals[node] = out_layer[local_idx]

            param_idx += num_nodes_layer
            # print("node_ab_values after layer", node_ab_values.keys())
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
    
