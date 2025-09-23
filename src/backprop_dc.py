import torch
from model import LevelizedModel

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

def backprop_dc(model):
    max_node_id = model.n_inputs - 1
    for layer in model.layers:
        max_node_id = max(max_node_id, max(layer))
    m = torch.zeros(max_node_id + 1, max_node_id + 1, 2, 2)

    for layer_idx in range(len(model.layers)-2, -1, -1):
        layer_nodes = model.layers[layer_idx]
        layer_num = layer_idx + 1
        # print(f"Layer{layer_num}: {layer_nodes}")

        for p in layer_nodes:
            if layer_num == len(model.layers) - 1:
                for q in model.layers[-1]:
                    # print(p,q)
                    # print(model.param_w0[q - model.n_inputs])
                    # print(count_nodes_in_layers_before(model, q))
                    m[p, q, 0, 0] = model.param_w0[q - model.n_inputs][p]
                    m[p, q, 0, 1] = model.param_w0[q - model.n_inputs][p + count_nodes_in_layers_before(model, q)]
                    m[p, q, 1, 0] = model.param_w1[q - model.n_inputs][p]
                    m[p, q, 1, 1] = model.param_w1[q - model.n_inputs][p + count_nodes_in_layers_before(model, q)]
                    # print(m[p, q, 0, 0], m[p, q, 0, 1], m[p, q, 1, 0], m[p, q, 1, 1])
            else:
                pass
            



if __name__ == "__main__":
    model = LevelizedModel(n_inputs=8, layers_config=[4, 6, 8])

    # print(count_nodes_in_layers_before(model, 17))
    backprop_dc(model)