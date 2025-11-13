import torch
import torch.nn.functional as F
import time
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

def backprop_dc(model, node_ab_values, output_bits, eps=1e-8, profile_time=False, return_internals=False):
    """
    Compute loss using don't-care backprop logic (batch-aware, autograd-friendly).
    
    Args:
        model: LevelizedModel instance
        node_ab_values: dict mapping node_id -> {'ap0': Tensor[B], 'ap1': Tensor[B],
                                                  'bp0': Tensor[B], 'bp1': Tensor[B], 'out': Tensor[B]}
        output_bits: Tensor [B, num_outputs] with {0,1}
        eps: Small epsilon for numerical stability
        profile_time: if True, return timing information
        return_internals: if True, return internal variables (m, t, f, g)
    
    Returns:
        If profile_time=False and return_internals=False: loss (scalar tensor)
        If profile_time=True: (loss, time_profile dict)
        If return_internals=True: (loss, {'m': dict, 't': dict, 'f': dict, 'g': dict})
        If both True: (loss, time_profile, internals)
    """
    if profile_time:
        time_profile = {
            'initialization': 0.0,
            'output_layer_loss': 0.0,
            'loop1_m_pq_calculation': 0.0,
            'loop2_g_p_calculation': 0.0,
            'loop3_f_p_calculation': 0.0,
            'loop4_m_rp_calculation': 0.0,
            'loss_accumulation': 0.0
        }
        start_init = time.time()
    
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
    # m: dict (node_r, node_p, i, j) -> Tensor (for storing m[r,p,i,j])
    f = {}
    g = {}
    t = {}
    m = {}

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

    # Map {0,1} -> {-1,1}
    transformed = (output_bits[:, :num_outputs].float() * 2.0 - 1.0)  # [B, num_outputs]
    # assign into f for each q in output_layer
    for idx, q in enumerate(output_layer):
        val = transformed[:, idx]               # [B]
        f[q] = torch.stack([val, torch.zeros_like(val)], dim=1)   # [B,2]

    loss = torch.tensor(0.0, device=device, requires_grad=True)

    if profile_time:
        time_profile['initialization'] = time.time() - start_init
        start_output_loss = time.time()

    output_layer = model.layers[-1]
    for q in output_layer:
        bq0 = node_ab_values[q]['bp0'].to(device)   # softmax fanin value at output node
        fq0 = f[q][:, 0]                             # desired value set from spec (±1 or 0)
        loss = loss + (torch.abs(fq0) * (bq0 - torch.sign(fq0))**2).sum()

    if profile_time:
        time_profile['output_layer_loss'] = time.time() - start_output_loss
        start_loop1 = time.time()

    # OPTIMIZATION: Wrap all DCBP construction (m, t, g, f) in no_grad()
    # Only bp0/bp1 need gradients - the supervision construction doesn't
    with torch.no_grad():
        # traverse layers backwards (excluding the output layer which is set)
        for layer_idx in range(len(model.layers) - 2, -1, -1):
            layer_nodes = model.layers[layer_idx]
            
            for p in layer_nodes:
                t_p = torch.tensor(0.0, device=device)
                
                # Step 1: Calculate m[p,q] and accumulate t[p] for all next layers
                # Iterate through all layers after current layer
                for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                    for q in model.layers[next_layer_idx]:
                        # Check if q is in the output layer
                        is_output_layer = (next_layer_idx == len(model.layers) - 1)

                        if is_output_layer:
                            # q is in output layer: m[p,q] comes directly from q's w0
                            # Use softmax(w0) as s[q,0,:]
                            row_idx = q - n_inputs
                            w0_row = model.param_w0[row_idx].detach()  # Detach weights
                            s0_row = F.softmax(w0_row, dim=0)  # Apply softmax
                            
                            # idx0 = p
                            # idx1 = p + count_nodes_in_layers_before(model, q)
                            cand_list_q = model.candidates[q - model.n_inputs]   # Index list of candidates for node q
                            idx0 = cand_list_q.index(p)
                            idx1 = idx0 + len(cand_list_q)
                            
                            m[(p, q, 0, 0)] = s0_row[idx0]
                            m[(p, q, 0, 1)] = s0_row[idx1]

                            # Accumulate t[p] (only w0 terms for output layer)
                            t_p = t_p + m[(p, q, 0, 0)] + m[(p, q, 0, 1)]
                            
                        else:
                            # Accumulate t[p] (both w0 and w1 terms for non-output layers)
                            t_p = t_p + m[(p, q, 0, 0)] + m[(p, q, 0, 1)] + m[(p, q, 1, 0)] + m[(p, q, 1, 1)]
                
                # Step 2: Now that t[p] is fully calculated, compute g[p]
                t[p] = t_p
                t_p_safe = torch.where(t_p.abs() > eps, t_p, t_p + eps)

                if profile_time and p == layer_nodes[0]:  # Time once per layer
                    time_profile['loop1_m_pq_calculation'] += time.time() - start_loop1
                    start_loop2 = time.time()

                g_p = torch.zeros(B, device=device)
                for next_layer_idx in range(layer_idx + 1, len(model.layers)):
                    for q in model.layers[next_layer_idx]:
                        is_output_layer = (next_layer_idx == len(model.layers) - 1)
                        
                        fq0 = f[q][:, 0]
                        fq1 = f[q][:, 1]

                        if is_output_layer:
                            # Output layer: only use w0 terms
                            g_p = g_p + (m[(p, q, 0, 0)] / t_p_safe) * fq0 + (m[(p, q, 0, 1)] / t_p_safe) * (-fq0)
                        else:
                            # Non-output layer: use both w0 and w1 terms
                            g_p = g_p + (m[(p, q, 0, 0)] / t_p_safe) * fq0 + (m[(p, q, 1, 0)] / t_p_safe) * fq1 \
                                       + (m[(p, q, 0, 1)] / t_p_safe) * (-fq0) + (m[(p, q, 1, 1)] / t_p_safe) * (-fq1)
                
                g[p] = g_p
                
                if profile_time and p == layer_nodes[0]:  # Time once per layer
                    time_profile['loop2_g_p_calculation'] += time.time() - start_loop2
                    start_loop3 = time.time()
                
                # Set f[p] according to rules
                ap0 = node_ab_values[p]['ap0'].to(device)
                ap1 = node_ab_values[p]['ap1'].to(device)
                
                mask0 = (g_p < 0) | (ap0 == 1)
                mask1 = (ap1 == -1)
                maskA = mask0 & mask1
                
                mask2 = (g_p > 0) | (ap1 == 1)
                mask3 = (ap0 == -1)
                maskB = mask2 & mask3
                
                f_p0 = g_p.clone()
                f_p1 = g_p.clone()
                f_p0 = torch.where(maskA, torch.zeros_like(f_p0), f_p0)
                f_p1 = torch.where(maskB, torch.zeros_like(f_p1), f_p1)
                
                f[p] = torch.stack([f_p0, f_p1], dim=1)
                
                if profile_time and p == layer_nodes[0]:  # Time once per layer
                    time_profile['loop3_f_p_calculation'] += time.time() - start_loop3
                    start_loop4 = time.time()
                
                # Calculate m[r,p] for all previous layers
                # m[r,p,i,j] = t[p] * s[p,i,j] where s[p,i,j] is softmax of p's weight
                for prev_layer_idx in range(layer_idx):
                    for r in model.layers[prev_layer_idx]:
                        row_idx_p = p - n_inputs
                        w0_row_p = model.param_w0[row_idx_p].detach()  # Detach weights
                        w1_row_p = model.param_w1[row_idx_p].detach()  # Detach weights
                        
                        # Apply softmax to get s[p,0,:] and s[p,1,:]
                        s0_row_p = F.softmax(w0_row_p, dim=0)
                        s1_row_p = F.softmax(w1_row_p, dim=0)
                        
                        # r_idx0 = r
                        # r_idx1 = r + count_nodes_in_layers_before(model, p)
                        cand_list_p = model.candidates[row_idx_p]
                        r_idx0 = cand_list_p.index(r)
                        r_idx1 = r_idx0 + len(cand_list_p)
                        
                        m[(r, p, 0, 0)] = t_p * s0_row_p[r_idx0]
                        m[(r, p, 0, 1)] = t_p * s0_row_p[r_idx1]
                        m[(r, p, 1, 0)] = t_p * s1_row_p[r_idx0]
                        m[(r, p, 1, 1)] = t_p * s1_row_p[r_idx1]

                if profile_time and p == layer_nodes[0]:  # Time once per layer
                    time_profile['loop4_m_rp_calculation'] += time.time() - start_loop4
    
    # OUTSIDE torch.no_grad(): Loss accumulation needs gradients for bp0/bp1
    if profile_time:
        start_loss_accum = time.time()
    
    # Add to loss using the constructed f values (which are detached)
    for layer_idx in range(len(model.layers) - 2, -1, -1):
        for p in model.layers[layer_idx]:
            # bp0/bp1 are differentiable (from forward pass)
            bp0 = node_ab_values[p]['bp0'].to(device)
            bp1 = node_ab_values[p]['bp1'].to(device)
            
            # f_p0 and f_p1 are detached (from DCBP construction)
            f_p0 = f[p][:, 0]
            f_p1 = f[p][:, 1]
            
            sign_f0 = torch.sign(f_p0)
            sign_f1 = torch.sign(f_p1)
            term0 = torch.abs(f_p0) * ((bp0 - sign_f0) ** 2)
            term1 = torch.abs(f_p1) * ((bp1 - sign_f1) ** 2)
            loss = loss + term0.sum() + term1.sum()
    
    if profile_time:
        time_profile['loss_accumulation'] = time.time() - start_loss_accum

    # final: average over batch size
    loss = loss / float(B)
    
    # Return based on flags
    if profile_time and return_internals:
        internals = {'m': m, 't': t, 'f': f, 'g': g}
        return loss, time_profile, internals
    elif profile_time:
        return loss, time_profile
    elif return_internals:
        internals = {'m': m, 't': t, 'f': f, 'g': g}
        return loss, internals
    else:
        return loss
