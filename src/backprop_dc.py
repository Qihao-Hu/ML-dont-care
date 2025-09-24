import torch
from model import LevelizedModel

def sign(x):
    """Sign function"""
    return torch.sign(x) if isinstance(x, torch.Tensor) else (1 if x > 0 else (-1 if x < 0 else 0))

def count_nodes_in_layers_before(model, node_p):
    """Count number of nodes in all layers before the layer containing node_p."""
    node_layer = None
    if node_p < model.n_inputs:
        node_layer = 0
    else:
        for layer_idx, layer_nodes in enumerate(model.layers):
            if node_p in layer_nodes:
                node_layer = layer_idx + 1
                break
    if node_layer is None:
        raise ValueError(f"Node {node_p} does not exist in the model.")
    total_nodes = 0
    if node_layer > 0:
        total_nodes += model.n_inputs
    if node_layer > 1:
        for layer_idx in range(node_layer - 1):
            total_nodes += len(model.layers[layer_idx])
    return total_nodes

def get_param_index(node, n_inputs):
    """Get parameter index for a given node."""
    return node - n_inputs

def set_m_values(m, p, q, w0_param, w1_param, offset, is_last_layer=False):
    """Set m matrix values for connections between p and q."""
    m[p, q, 0, 0] = w0_param[p]
    m[p, q, 0, 1] = w0_param[p + offset]
    if not is_last_layer:
        m[p, q, 1, 0] = w1_param[p]
        m[p, q, 1, 1] = w1_param[p + offset]

def update_backward_m_values(m, r, p, t_p, w0_param, w1_param, offset):
    """Update backward propagation m values for previous layers."""
    m[r, p, 0, 0] = t_p * w0_param[r]
    m[r, p, 0, 1] = t_p * w0_param[r + offset]
    m[r, p, 1, 0] = t_p * w1_param[r]
    m[r, p, 1, 1] = t_p * w1_param[r + offset]

def compute_f_values(g_p, ap0, ap1):
    """Compute f values based on gradient and ap values."""
    if (g_p < 0 or ap0 == 1) and (ap1 == -1):
        return 0, g_p
    elif (g_p > 0 or ap1 == 1) and (ap0 == -1):
        return g_p, 0
    else:
        return g_p, g_p

def compute_loss_term(f_val, bp_val):
    """Compute individual loss term."""
    return torch.abs(f_val) * (bp_val - sign(f_val)) ** 2

def backprop_dc(model, node_ab_values):
    # Initialize variables
    max_node_id = max(model.n_inputs - 1, max(max(layer) for layer in model.layers))
    m = torch.zeros(max_node_id + 1, max_node_id + 1, 2, 2)
    t = torch.zeros(max_node_id + 1)
    g = torch.zeros(max_node_id + 1)
    f = torch.zeros(max_node_id + 1, 2)
    loss = 0
    
    # Start backpropagation from the second-to-last layer
    for layer_idx in range(len(model.layers) - 2, -1, -1):
        layer_nodes = model.layers[layer_idx]
        is_last_layer = (layer_idx == len(model.layers) - 2)
        next_layer = model.layers[-1] if is_last_layer else model.layers[layer_idx + 1]
        
        for p in layer_nodes:
            # Reset accumulation values
            t[p] = 0
            g[p] = 0
            
            # Compute connections to next layer
            for q in next_layer:
                param_idx = get_param_index(q, model.n_inputs)
                offset = count_nodes_in_layers_before(model, q)
                
                # Set m values
                set_m_values(m, p, q, model.param_w0[param_idx], 
                           model.param_w1[param_idx], offset, is_last_layer)
                
                # Accumulate t and g values
                t[p] += m[p, q, 0, 0] + m[p, q, 0, 1]
                
                if is_last_layer:
                    g[p] += (m[p, q, 0, 0] / t[p]) * f[q, 0]
                else:
                    g[p] += (m[p, q, 0, 0] / t[p]) * f[q, 0] + (m[p, q, 1, 0] / t[p]) * (-f[q, 1])
            
            # Compute f values
            ap0_val = node_ab_values[p]['ap0'].item()
            ap1_val = node_ab_values[p]['ap1'].item()
            f[p, 0], f[p, 1] = compute_f_values(g[p].item(), ap0_val, ap1_val)
            
            # Update m values for previous layers
            param_idx = get_param_index(p, model.n_inputs)
            offset = count_nodes_in_layers_before(model, p)
            
            for prev_layer_idx in range(layer_idx):
                for r in model.layers[prev_layer_idx]:
                    update_backward_m_values(m, r, p, t[p], 
                                           model.param_w0[param_idx], 
                                           model.param_w1[param_idx], offset)
            
            # Accumulate loss
            bp0_val = node_ab_values[p]['bp0'].item()
            bp1_val = node_ab_values[p]['bp1'].item()
            loss += compute_loss_term(f[p, 0], bp0_val) + compute_loss_term(f[p, 1], bp1_val)
    
    print(f"Total loss: {loss}")

if __name__ == "__main__":
    model = LevelizedModel(n_inputs=8, layers_config=[4, 6, 8])

    # Test case: 3 * 5
    input_bits = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)
    node_ab_values = model(input_bits)
    
    backprop_dc(model, node_ab_values)