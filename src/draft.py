import torch
from model import LevelizedModel

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def count_nodes_in_layers_before(model, node_p):
    node_layer = None
    if node_p < model.n_inputs:
        node_layer = 0  
    else:
        for layer_idx, layer_nodes in enumerate(model.layers):
            if node_p in layer_nodes:
                node_layer = layer_idx + 1 
                break
    if node_layer is None:
        raise ValueError(f" Node {node_p} does not exist in the model.")
    total_nodes = 0
    if node_layer > 0:
        total_nodes += model.n_inputs  
    if node_layer > 1:
        for layer_idx in range(node_layer - 1): 
            total_nodes += len(model.layers[layer_idx])
    return total_nodes

def backprop_dc(model, node_ab_values, output_bits):
    max_node_id = model.n_inputs - 1
    for layer in model.layers:
        max_node_id = max(max_node_id, max(layer))
    m = torch.zeros(max_node_id + 1, max_node_id + 1, 2, 2)
    t = torch.zeros(max_node_id + 1)
    g = torch.zeros(max_node_id + 1)
    f = torch.zeros(max_node_id + 1, 2)
    loss = 0

    # ---- Initialize f for output layer using provided output_bits ----
    # output_bits expected shape: (num_outputs,) or (1, num_outputs)
    if output_bits.dim() == 2:
        if output_bits.size(0) != 1:
            raise ValueError("Batch size > 1 not supported here")
        output_bits_flat = output_bits[0]
    else:
        output_bits_flat = output_bits
    output_layer = model.layers[-1]
    assert len(output_layer) == output_bits_flat.numel(), \
        f"Length mismatch: output layer has {len(output_layer)} nodes but got {output_bits_flat.numel()} labels"
    # Map {0,1} -> {-1,1} by 2*x - 1, assign to both channels
    transformed = output_bits_flat * 2 - 1
    f[output_layer, 0] = transformed
    f[output_layer, 1] = transformed
    # ------------------------------------------------------------------

    for layer_idx in range(len(model.layers)-2, -1, -1):
        layer_nodes = model.layers[layer_idx]
        layer_num = layer_idx + 1
        # print(f"Layer{layer_num}: {layer_nodes}")

        if layer_num == len(model.layers) - 1:
            for p in layer_nodes:
                for q in model.layers[-1]:
                    # print(p,q)
                    # print(model.param_w0[q - model.n_inputs])
                    # print(count_nodes_in_layers_before(model, q))
                    m[p, q, 0, 0] = model.param_w0[q - model.n_inputs][p]
                    m[p, q, 0, 1] = model.param_w0[q - model.n_inputs][p + count_nodes_in_layers_before(model, q)]
                    # m[p, q, 1, 0] = model.param_w1[q - model.n_inputs][p]
                    # m[p, q, 1, 1] = model.param_w1[q - model.n_inputs][p + count_nodes_in_layers_before(model, q)]
                    
                    t[p] += m[p, q, 0, 0] + m[p, q, 0, 1]

                    g[p] += (m[p, q, 0, 0] / t[p]) * f[q, 0]


                if (g[p] < 0 or node_ab_values[p]['ap0'] == 1) and (node_ab_values[p]['ap1'] == -1):
                    f[p, 0] = 0
                elif (g[p] > 0 or node_ab_values[p]['ap1'] == 1) and (node_ab_values[p]['ap0'] == -1):
                    f[p, 1] = 0
                else:
                    f[p, 0] = g[p]
                    f[p, 1] = g[p]


                for prev_layer_idx in range(layer_idx):
                    for r in model.layers[prev_layer_idx]:
                        m[r, p, 0, 0] = t[p] * model.param_w0[p - model.n_inputs][r]
                        m[r, p, 0, 1] = t[p] * model.param_w0[p - model.n_inputs][r + count_nodes_in_layers_before(model, p)]
                        m[r, p, 1, 0] = t[p] * model.param_w1[p - model.n_inputs][r]
                        m[r, p, 1, 1] = t[p] * model.param_w1[p - model.n_inputs][r + count_nodes_in_layers_before(model, p)]
                
                
                loss += abs(f[p, 0]) * pow((node_ab_values[p]['bp0'] - sign(f[p, 0])), 2) + abs(f[p, 1]) * pow((node_ab_values[p]['bp1'] - sign(f[p, 1])), 2)
                        
        else:
            for p in layer_nodes:
                for q in model.layers[layer_idx + 1]:
                    m[p, q, 0, 0] = model.param_w0[q - model.n_inputs][p]
                    m[p, q, 0, 1] = model.param_w0[q - model.n_inputs][p + count_nodes_in_layers_before(model, q)]
                    m[p, q, 1, 0] = model.param_w1[q - model.n_inputs][p]
                    m[p, q, 1, 1] = model.param_w1[q - model.n_inputs][p + count_nodes_in_layers_before(model, q)]
                    
                    t[p] += m[p, q, 0, 0] + m[p, q, 0, 1]

                    g[p] += (m[p, q, 0, 0] / t[p]) * f[q, 0] + (m[p, q, 1, 0] / t[p]) * (- f[q, 1])

                if (g[p] < 0 or node_ab_values[p]['ap0'] == 1) and (node_ab_values[p]['ap1'] == -1):
                    f[p, 0] = 0
                elif (g[p] > 0 or node_ab_values[p]['ap1'] == 1) and (node_ab_values[p]['ap0'] == -1):
                    f[p, 1] = 0
                else:
                    f[p, 0] = g[p]
                    f[p, 1] = g[p]

                for prev_layer_idx in range(layer_idx):
                    for r in model.layers[prev_layer_idx]:
                        m[r, p, 0, 0] = t[p] * model.param_w0[p - model.n_inputs][r]
                        m[r, p, 0, 1] = t[p] * model.param_w0[p - model.n_inputs][r + count_nodes_in_layers_before(model, p)]
                        m[r, p, 1, 0] = t[p] * model.param_w1[p - model.n_inputs][r]
                        m[r, p, 1, 1] = t[p] * model.param_w1[p - model.n_inputs][r + count_nodes_in_layers_before(model, p)]

                loss += abs(f[p, 0]) * pow((node_ab_values[p]['bp0'] - sign(f[p, 0])), 2) + abs(f[p, 1]) * pow((node_ab_values[p]['bp1'] - sign(f[p, 1])), 2)


    print(loss)

if __name__ == "__main__":
    model = LevelizedModel(n_inputs=8, layers_config=[4, 6, 8])

    # Test case: 3 * 5
    input_bits = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)
    node_ab_values = model(input_bits)

    output_bits = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)  # 15 in binary
    
    backprop_dc(model, node_ab_values, output_bits)