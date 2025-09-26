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

def backprop_dc(model, node_ab_values, output_bits, eps=1e-8):
    """
    Compute loss using don't-care backprop logic (batch-aware, autograd-friendly).
    - model: LevelizedModel
    - node_ab_values: dict mapping node_id -> {'ap0': Tensor[B], 'ap1': Tensor[B],
                                              'bp0': Tensor[B], 'bp1': Tensor[B], 'out': Tensor[B]}
    - output_bits: Tensor [B, num_outputs] with {0,1}
    Returns scalar loss (tensor) suitable for loss.backward()
    """
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    B = output_bits.size(0)
    n_inputs = model.n_inputs

    # find max node id
    max_node_id = n_inputs - 1
    for layer in model.layers:
        max_node_id = max(max_node_id, max(layer))

    # f: dict node -> Tensor shape [B, 2]
    # g: dict node -> Tensor shape [B]
    # t: dict node -> scalar Tensor (no batch)
    f = {}
    g = {}
    t = {}

    # initialize containers
    for nid in range(max_node_id + 1):
        # f default zeros [B,2]
        f[nid] = torch.zeros(B, 2, device=device)
        g[nid] = torch.zeros(B, device=device)
        t[nid] = torch.tensor(0.0, device=device)

    # --- initialize f for output layer using provided output_bits ---
    # output_bits: [B, num_outputs]
    if output_bits.dim() != 2:
        raise ValueError("output_bits should be [B, num_outputs]")
    output_layer = model.layers[-1]
    num_outputs = len(output_layer)
    assert output_bits.size(1) == num_outputs or output_bits.size(1) == len(model.output_nodes), \
        "output_bits width mismatch with model's output layer"

    # We assume output_bits correspond to output_layer order; user used output_nodes earlier but
    # here follow your prior code that set f for nodes in last layer.
    # Map {0,1} -> {-1,1}
    transformed = (output_bits[:, :num_outputs].float() * 2.0 - 1.0)  # [B, num_outputs]
    # assign into f for each q in output_layer
    for idx, q in enumerate(output_layer):
        val = transformed[:, idx]               # [B]
        f[q] = torch.stack([val, val], dim=1)   # [B,2]

    loss = torch.tensor(0.0, device=device)

    # traverse layers backwards (excluding the output layer which is set)
    for layer_idx in range(len(model.layers) - 2, -1, -1):
        layer_nodes = model.layers[layer_idx]
        layer_num = layer_idx + 1

        for p in layer_nodes:
            # t_p is scalar (accumulate weights for node p)
            t_p = torch.tensor(0.0, device=device)
            # g_p is per-batch
            g_p = torch.zeros(B, device=device)

            if layer_num == len(model.layers) - 1:
                # Special handling matching the non-comment backprop_dc for last-1 layer
                for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                    for q in model.layers[next_layer_idx]:
                        row_idx = q - n_inputs
                        w0_row = model.param_w0[row_idx]
                        w1_row = model.param_w1[row_idx]

                        idx0 = p
                        idx1 = p + count_nodes_in_layers_before(model, q)

                        m00 = w0_row[idx0]
                        m01 = w0_row[idx1]

                        # accumulate only m00+m01 for t_p
                        t_p = t_p + (m00 + m01)
                        t_p_safe = t_p if (t_p.abs() > eps).item() else (t_p + eps)

                        fq0 = f[q][:, 0]
                        # g[p] += (m00/t[p]) * f[q,0] + (m01/t[p]) * (- f[q,0])
                        g_p = g_p + (m00 / t_p_safe) * fq0 + (m01 / t_p_safe) * (-fq0)
            else:
                # General case matching the non-comment backprop_dc
                for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                    for q in model.layers[next_layer_idx]:
                        row_idx = q - n_inputs
                        w0_row = model.param_w0[row_idx]
                        w1_row = model.param_w1[row_idx]

                        idx0 = p
                        idx1 = p + count_nodes_in_layers_before(model, q)

                        m00 = w0_row[idx0]
                        m01 = w0_row[idx1]
                        m10 = w1_row[idx0]
                        m11 = w1_row[idx1]

                        # accumulate all four into t_p
                        t_p = t_p + (m00 + m01 + m10 + m11)
                        t_p_safe = t_p if (t_p.abs() > eps).item() else (t_p + eps)

                        fq0 = f[q][:, 0]
                        fq1 = f[q][:, 1]
                        # g[p] += (m00/t)*f[q,0] + (m10/t)*f[q,1] + (m01/t)*(- f[q,0]) + (m11/t)*(- f[q,1])
                        g_p = g_p + (m00 / t_p_safe) * fq0 + (m10 / t_p_safe) * fq1 \
                                   + (m01 / t_p_safe) * (-fq0) + (m11 / t_p_safe) * (-fq1)

            # now set f[p] according to your rules, but do it elementwise for batch
            # node_ab_values[p]['ap0'] and ['ap1'] are tensors [B]
            ap0 = node_ab_values[p]['ap0'].to(device)  # [B]
            ap1 = node_ab_values[p]['ap1'].to(device)  # [B]

            # elementwise masks (python bools per element avoided; keep tensors)
            mask0 = (g_p < 0) | (ap0 == 1)
            mask1 = (ap1 == -1)

            maskA = mask0 & mask1  # condition to set f[p,0]=0

            mask2 = (g_p > 0) | (ap1 == 1)
            mask3 = (ap0 == -1)
            maskB = mask2 & mask3  # condition to set f[p,1]=0

            # default both equal to g_p
            f_p0 = g_p.clone()
            f_p1 = g_p.clone()

            # set f_p0 to 0 where maskA true (elementwise), but use tensor ops (no in-place)
            f_p0 = torch.where(maskA, torch.zeros_like(f_p0), f_p0)

            # set f_p1 to 0 where maskB true
            f_p1 = torch.where(maskB, torch.zeros_like(f_p1), f_p1)

            # stack back into [B,2] and store
            f[p] = torch.stack([f_p0, f_p1], dim=1)
            t[p] = t_p
            g[p] = g_p

            # compute loss term (use tensor ops)
            bp0 = node_ab_values[p]['bp0'].to(device)  # [B]
            bp1 = node_ab_values[p]['bp1'].to(device)  # [B]

            # use torch.sign rather than python sign to remain in graph
            sign_f0 = torch.sign(f_p0)  # returns -1,0,1 per element
            sign_f1 = torch.sign(f_p1)

            term0 = torch.abs(f_p0) * ((bp0 - sign_f0) ** 2)  # [B]
            term1 = torch.abs(f_p1) * ((bp1 - sign_f1) ** 2)  # [B]

            # sum over batch and add to scalar loss
            loss = loss + term0.sum() + term1.sum()

    # final: average over batch size (optional; keep consistent with optimizer scale)
    loss = loss / float(B)
    return loss

if __name__ == "__main__":
    model = LevelizedModel(n_inputs=8, layers_config=[10, 10, 10, 10, 8])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    # # Test case: 3 * 5
    # input_bits = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)
    # node_ab_values = model(input_bits)

    # output_bits = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)  # 15 in binary
    
    # backprop_dc(model, node_ab_values, output_bits)
    for step in range(1000):
        input_bits = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)
        node_ab_values = model(input_bits)
        output_bits = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)

        # --- Calculate loss ---
        loss = backprop_dc(model, node_ab_values, output_bits)

        # --- Adam update ---
        optimizer.zero_grad()
        loss.backward()   # Automatically compute gradients for param_w0, param_w1
        optimizer.step()

        print(f"Step {step:03d} | Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        # Run forward again to get node_ab_values
        input_bits = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)
        node_ab_values = model(input_bits)

        # Get the outputs of the model's last layer
        outputs = []
        for q in model.layers[-1]:
            outputs.append(node_ab_values[q]["out"])  # Value of each output node
        outputs = torch.stack(outputs, dim=1)  # [1, num_outputs]

        # Convert continuous values to 0/1 (binarize)
        pred_bits = (outputs > 0).int()
        print("预测结果:", pred_bits.tolist())