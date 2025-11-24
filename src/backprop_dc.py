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

def backprop_dc(model, node_ab_values, output_bits, eps=1e-8, profile_time=False, return_internals=False, debug_output=False):
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
        debug_output: if True, write m, t, g, f for each node to debug.txt
    
    Returns:
        If profile_time=False and return_internals=False: loss (scalar tensor)
        If profile_time=True: (loss, time_profile dict)
        If return_internals=True: (loss, {'m': dict, 't': dict, 'f': dict, 'g': dict})
        If both True: (loss, time_profile, internals)
    """
    # Clear debug file at the start if debug output is enabled
    if debug_output:
        with open('debug.txt', 'w') as debug_file:
            debug_file.write("="*80 + "\n")
            debug_file.write("DEBUG OUTPUT: backprop_dc - m, t, g, f for each node\n")
            debug_file.write("="*80 + "\n\n")
    
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

    # Matrix-form containers
    num_nodes = max_node_id + 1
    f_tensor = torch.zeros(num_nodes, B, 2, device=device)
    g_tensor = torch.zeros(num_nodes, B, device=device)
    t_tensor = torch.zeros(num_nodes, device=device)

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
        f_tensor[q] = torch.stack([val, torch.zeros_like(val)], dim=1)   # [B,2]

    loss = torch.tensor(0.0, device=device, requires_grad=True)

    if profile_time:
        time_profile['initialization'] = time.time() - start_init
        start_output_loss = time.time()

    output_layer = model.layers[-1]
    for q in output_layer:
        bq0 = node_ab_values[q]['bp0'].to(device)   # softmax fanin value at output node
        fq0 = f_tensor[q, :, 0]                      # desired value set from spec (±1 or 0)
        loss = loss + (torch.abs(fq0) * (bq0 - torch.sign(fq0))**2).sum()

    if profile_time:
        time_profile['output_layer_loss'] = time.time() - start_output_loss
        start_loop1 = time.time()

    # SAFE OPTIMIZATION 1: Build candidate index map ONCE (static structure)
    # This map never changes, so we can cache it in the model object
    if not hasattr(model, 'cand_map'):
        model.cand_map = {}
        for node_id in range(n_inputs, n_inputs + len(model.param_w0)):
            cand_list = model.candidates[node_id - n_inputs]
            M_i = len(cand_list)
            for cand_idx, cand in enumerate(cand_list):
                # Store both idx0 and idx1 for O(1) lookup
                model.cand_map[(node_id, cand)] = (cand_idx, cand_idx + M_i)

    # SAFE OPTIMIZATION 2: Pre-compute softmax for CURRENT weights (updates each step)
    # This MUST be inside backprop_dc() to use the updated weights after optimizer.step()
    softmax_w0_cache = {}
    softmax_w1_cache = {}
    with torch.no_grad():
        for node_id in range(n_inputs, max_node_id + 1):
            if node_id - n_inputs < len(model.param_w0):
                # Use current weights (these change every step!)
                softmax_w0_cache[node_id] = F.softmax(model.param_w0[node_id - n_inputs].detach(), dim=0)
                softmax_w1_cache[node_id] = F.softmax(model.param_w1[node_id - n_inputs].detach(), dim=0)

    # Precompute per-layer candidate metadata for vectorized operations
    node_candidates_by_layer = []
    prev_node_to_idx_by_layer = []
    layer_node_tensors = []
    param_idx = 0
    for layer_nodes in model.layers:
        layer_candidates = model.candidates[param_idx]
        node_candidates_by_layer.append(layer_candidates)
        prev_node_to_idx_by_layer.append({nid: idx for idx, nid in enumerate(layer_candidates)})
        layer_node_tensors.append(torch.tensor(layer_nodes, device=device, dtype=torch.long))
        param_idx += len(layer_nodes)

    # Precompute output-layer m values once (only w0 is relevant)
    output_prev_nodes = node_candidates_by_layer[-1] if node_candidates_by_layer else []
    prev_node_to_idx_out = {nid: idx for idx, nid in enumerate(output_prev_nodes)}
    m_output_tensor = None
    if output_layer:
        M_out = len(output_prev_nodes)
        idx0_out = torch.arange(M_out, device=device)
        idx1_out = idx0_out + M_out
        s0_stack_out = torch.stack([softmax_w0_cache[q] for q in output_layer], dim=0)  # [P_out, 2*M_out]
        m_output_tensor = torch.stack(
            [s0_stack_out[:, idx0_out], s0_stack_out[:, idx1_out]], dim=-1
        ).permute(1, 0, 2)  # [M_out, P_out, 2]

    # OPTIMIZATION: Wrap all DCBP construction (m, t, g, f) in no_grad()
    # Only bp0/bp1 need gradients - the supervision construction doesn't
    with torch.no_grad():
        m_by_layer = {}
        # traverse layers backwards (excluding the output layer which is set)
        for layer_idx in range(len(model.layers) - 2, -1, -1):
            layer_nodes = model.layers[layer_idx]
            layer_nodes_tensor = layer_node_tensors[layer_idx]
            prev_nodes = node_candidates_by_layer[layer_idx]
            M_current = len(prev_nodes)
            P_current = len(layer_nodes)

            if profile_time:
                start_loop1 = time.time()

            # Step 1: Calculate t[p] for all nodes in the current layer
            t_layer = torch.zeros(P_current, device=device)
            m_sel_out = None

            if m_output_tensor is not None:
                idxs_out = torch.tensor([prev_node_to_idx_out[p] for p in layer_nodes], device=device)
                m_sel_out = m_output_tensor[idxs_out]  # [P_current, P_out, 2]
                t_layer = t_layer + m_sel_out.sum(dim=(1, 2))

            for future_idx in range(layer_idx + 1, len(model.layers) - 1):
                m_future = m_by_layer[future_idx]
                idx_map_future = prev_node_to_idx_by_layer[future_idx]
                idxs_future = torch.tensor([idx_map_future[p] for p in layer_nodes], device=device)
                m_sel = m_future[idxs_future]  # [P_current, num_future, 2, 2]
                t_layer = t_layer + m_sel.sum(dim=(1, 2, 3))

            t_tensor[layer_nodes_tensor] = t_layer
            t_layer_safe = torch.where(t_layer.abs() > eps, t_layer, t_layer + eps)

            if profile_time:
                time_profile['loop1_m_pq_calculation'] += time.time() - start_loop1
                start_loop2 = time.time()

            # Step 2: Compute g[p] for all nodes in the current layer
            g_layer = torch.zeros(P_current, B, device=device)

            if m_sel_out is not None:
                fq0_out = f_tensor[layer_node_tensors[-1], :, 0]  # [P_out, B]
                contrib_out = (
                    m_sel_out[:, :, 0].unsqueeze(-1) * fq0_out.unsqueeze(0) +
                    m_sel_out[:, :, 1].unsqueeze(-1) * (-fq0_out.unsqueeze(0))
                ).sum(dim=1)
                g_layer = g_layer + contrib_out / t_layer_safe.unsqueeze(-1)

            for future_idx in range(layer_idx + 1, len(model.layers) - 1):
                future_nodes_tensor = layer_node_tensors[future_idx]
                m_future = m_by_layer[future_idx]
                idx_map_future = prev_node_to_idx_by_layer[future_idx]
                idxs_future = torch.tensor([idx_map_future[p] for p in layer_nodes], device=device)
                m_sel = m_future[idxs_future]  # [P_current, P_future, 2, 2]

                fq = f_tensor[future_nodes_tensor]  # [P_future, B, 2]
                fq0 = fq[:, :, 0]
                fq1 = fq[:, :, 1]

                contrib_future = (
                    m_sel[:, :, 0, 0].unsqueeze(-1) * fq0.unsqueeze(0) +
                    m_sel[:, :, 1, 0].unsqueeze(-1) * fq1.unsqueeze(0) +
                    m_sel[:, :, 0, 1].unsqueeze(-1) * (-fq0.unsqueeze(0)) +
                    m_sel[:, :, 1, 1].unsqueeze(-1) * (-fq1.unsqueeze(0))
                ).sum(dim=1)

                g_layer = g_layer + contrib_future / t_layer_safe.unsqueeze(-1)

            g_tensor[layer_nodes_tensor] = g_layer

            if profile_time:
                time_profile['loop2_g_p_calculation'] += time.time() - start_loop2
                start_loop3 = time.time()

            # Step 3: Compute f[p] for all nodes in the current layer
            ap0_layer = torch.stack([node_ab_values[p]['ap0'].to(device) for p in layer_nodes], dim=0)
            ap1_layer = torch.stack([node_ab_values[p]['ap1'].to(device) for p in layer_nodes], dim=0)

            maskA = ((g_layer < 0) | (ap0_layer == 1)) & (ap1_layer == -1)
            maskB = ((g_layer > 0) | (ap1_layer == 1)) & (ap0_layer == -1)

            f_p0 = torch.where(maskA, torch.zeros_like(g_layer), g_layer)
            f_p1 = torch.where(maskB, torch.zeros_like(g_layer), g_layer)

            f_layer = torch.stack([f_p0, f_p1], dim=2)  # [P_current, B, 2]
            f_tensor[layer_nodes_tensor] = f_layer

            if profile_time:
                time_profile['loop3_f_p_calculation'] += time.time() - start_loop3
                start_loop4 = time.time()

            # Step 4: Calculate m[r,p] for all previous layers (matrix form)
            idx0 = torch.arange(M_current, device=device)
            idx1 = idx0 + M_current
            s0_stack = torch.stack([softmax_w0_cache[p] for p in layer_nodes], dim=0)
            s1_stack = torch.stack([softmax_w1_cache[p] for p in layer_nodes], dim=0)

            m00 = t_layer.unsqueeze(1) * s0_stack[:, idx0]  # [P_current, M_current]
            m01 = t_layer.unsqueeze(1) * s0_stack[:, idx1]
            m10 = t_layer.unsqueeze(1) * s1_stack[:, idx0]
            m11 = t_layer.unsqueeze(1) * s1_stack[:, idx1]

            m_layer_tensor = torch.zeros(M_current, P_current, 2, 2, device=device)
            m_layer_tensor[:, :, 0, 0] = m00.transpose(0, 1)
            m_layer_tensor[:, :, 0, 1] = m01.transpose(0, 1)
            m_layer_tensor[:, :, 1, 0] = m10.transpose(0, 1)
            m_layer_tensor[:, :, 1, 1] = m11.transpose(0, 1)

            m_by_layer[layer_idx] = m_layer_tensor

            if profile_time:
                time_profile['loop4_m_rp_calculation'] += time.time() - start_loop4
    
    def _build_internal_dicts():
        f_dict = {nid: f_tensor[nid] for nid in range(num_nodes)}
        g_dict = {nid: g_tensor[nid] for nid in range(num_nodes)}
        t_dict = {nid: t_tensor[nid] for nid in range(num_nodes)}
        m_dict = {}

        if m_output_tensor is not None:
            for r_idx, r in enumerate(output_prev_nodes):
                for q_idx, q in enumerate(output_layer):
                    m_dict[(r, q, 0, 0)] = m_output_tensor[r_idx, q_idx, 0]
                    m_dict[(r, q, 0, 1)] = m_output_tensor[r_idx, q_idx, 1]

        for l_idx, layer_nodes in enumerate(model.layers[:-1]):
            if l_idx not in m_by_layer:
                continue
            m_tensor = m_by_layer[l_idx]
            prev_nodes = node_candidates_by_layer[l_idx]
            for r_idx, r in enumerate(prev_nodes):
                for p_idx, p in enumerate(layer_nodes):
                    m_dict[(r, p, 0, 0)] = m_tensor[r_idx, p_idx, 0, 0]
                    m_dict[(r, p, 0, 1)] = m_tensor[r_idx, p_idx, 0, 1]
                    m_dict[(r, p, 1, 0)] = m_tensor[r_idx, p_idx, 1, 0]
                    m_dict[(r, p, 1, 1)] = m_tensor[r_idx, p_idx, 1, 1]

        return m_dict, t_dict, f_dict, g_dict

    if debug_output or return_internals:
        m_dict, t_dict, f_dict, g_dict = _build_internal_dicts()
    else:
        m_dict = t_dict = f_dict = g_dict = None

    if debug_output:
        with open('debug.txt', 'a') as debug_file:
            debug_file.write(f"\n{'#'*80}\n")
            debug_file.write("VECTORIZED BACKPROP DEBUG SNAPSHOT\n")
            debug_file.write(f"{'#'*80}\n\n")

            all_network_nodes = []
            for l_idx in range(len(model.layers) - 1):
                all_network_nodes.extend(model.layers[l_idx])
            all_network_nodes.extend(output_layer)

            debug_file.write(f"Total network nodes: {sorted(all_network_nodes)}\n")
            debug_file.write(f"Number of nodes: {len(all_network_nodes)}\n\n")

            for node_id in sorted(all_network_nodes):
                debug_file.write(f"\n{'='*80}\n")
                debug_file.write(f"Node = {node_id}")
                if node_id in output_layer:
                    debug_file.write(f" (Output Layer - Layer {len(model.layers)-1})\n")
                else:
                    for l_idx, layer in enumerate(model.layers[:-1]):
                        if node_id in layer:
                            debug_file.write(f" (Hidden Layer {l_idx})\n")
                            break
                debug_file.write(f"{'='*80}\n")

                t_val = t_tensor[node_id]
                t_str = f"{t_val.item():.6f}" if isinstance(t_val, torch.Tensor) else f"{t_val:.6f}"
                debug_file.write(f"\nt[{node_id}] = {t_str}\n")

                g_arr = g_tensor[node_id].cpu().numpy()
                debug_file.write(f"\ng[{node_id}] (shape {g_tensor[node_id].shape}):\n")
                if len(g_arr) > 5:
                    debug_file.write(f"  [First 5]: {g_arr[:5]}\n")
                else:
                    debug_file.write(f"  {g_arr}\n")

                f_arr = f_tensor[node_id].cpu().numpy()
                debug_file.write(f"\nf[{node_id}] (shape {f_tensor[node_id].shape}):\n")
                if len(f_arr) > 5:
                    debug_file.write(f"  [First 5 rows]:\n{f_arr[:5]}\n")
                else:
                    debug_file.write(f"{f_arr}\n")

                if node_id >= n_inputs and node_id - n_inputs < len(model.param_w0):
                    node_index = node_id - n_inputs
                    debug_file.write(f"\nNode {node_id} forward pass values:\n")
                    debug_file.write("-" * 80 + "\n")

                    if node_id in node_ab_values:
                        ap0 = node_ab_values[node_id]['ap0'].to(device)
                        ap1 = node_ab_values[node_id]['ap1'].to(device)
                        bp0 = node_ab_values[node_id]['bp0'].to(device)
                        bp1 = node_ab_values[node_id]['bp1'].to(device)

                        ap0_arr = ap0.cpu().numpy()
                        ap1_arr = ap1.cpu().numpy()
                        bp0_arr = bp0.cpu().numpy()
                        bp1_arr = bp1.cpu().numpy()

                        debug_file.write(f"ap0 (shape {ap0.shape}): ")
                        debug_file.write(f"[First 5]: {ap0_arr[:5]}\n" if len(ap0_arr) > 5 else f"{ap0_arr}\n")

                        debug_file.write(f"ap1 (shape {ap1.shape}): ")
                        debug_file.write(f"[First 5]: {ap1_arr[:5]}\n" if len(ap1_arr) > 5 else f"{ap1_arr}\n")

                        debug_file.write(f"bp0 (shape {bp0.shape}): ")
                        debug_file.write(f"[First 5]: {bp0_arr[:5]}\n" if len(bp0_arr) > 5 else f"{bp0_arr}\n")

                        debug_file.write(f"bp1 (shape {bp1.shape}): ")
                        debug_file.write(f"[First 5]: {bp1_arr[:5]}\n" if len(bp1_arr) > 5 else f"{bp1_arr}\n")
                    else:
                        debug_file.write("Forward pass values not available for this node\n")

                    cand_list = model.candidates[node_index]
                    debug_file.write(f"\nCandidates: {cand_list}\n")

                    if node_id in softmax_w0_cache:
                        s0 = softmax_w0_cache[node_id].cpu().numpy()
                        s1 = softmax_w1_cache[node_id].cpu().numpy()
                        M_i = len(cand_list)
                        debug_file.write(f"\nSoftmax weights (w0 and w1) for each candidate:\n")
                        for idx, cand_node in enumerate(cand_list):
                            debug_file.write(f"  Candidate {idx} (node {cand_node}):\n")
                            debug_file.write(f"    w0: [{s0[idx]:.6f}, {s0[idx + M_i]:.6f}]\n")
                            debug_file.write(f"    w1: [{s1[idx]:.6f}, {s1[idx + M_i]:.6f}]\n")
                    else:
                        debug_file.write("Softmax weights not computed yet\n")

                debug_file.write(f"\nm values for node {node_id}:\n")
                debug_file.write("-" * 80 + "\n")

                if m_dict is not None:
                    m_keys_first = sorted([k for k in m_dict.keys() if k[0] == node_id])
                    if m_keys_first:
                        debug_file.write(f"m[(p={node_id}, q, i, j)] values:\n")
                        for key in m_keys_first:
                            val = m_dict[key]
                            val_str = f"{val.item():.6f}" if isinstance(val, torch.Tensor) else f"{val:.6f}"
                            debug_file.write(f"  m[{key}] = {val_str}\n")

                    m_keys_second = sorted([k for k in m_dict.keys() if k[1] == node_id])
                    if m_keys_second:
                        debug_file.write(f"m[(r, p={node_id}, i, j)] values:\n")
                        for key in m_keys_second:
                            val = m_dict[key]
                            val_str = f"{val.item():.6f}" if isinstance(val, torch.Tensor) else f"{val:.6f}"
                            debug_file.write(f"  m[{key}] = {val_str}\n")

                    if not m_keys_first and not m_keys_second:
                        debug_file.write(f"  No m values computed for this node yet\n")

                debug_file.write("\n")

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
            f_p0 = f_tensor[p, :, 0]
            f_p1 = f_tensor[p, :, 1]
            
            sign_f0 = torch.sign(f_p0)
            sign_f1 = torch.sign(f_p1)

            term0 = torch.abs(f_p0) * ((bp0 - sign_f0) ** 2)
            # print("term0.shape:", term0.shape)
            term1 = torch.abs(f_p1) * ((bp1 - sign_f1) ** 2)
            loss = loss + term0.sum() + term1.sum()
    
    if profile_time:
        time_profile['loss_accumulation'] = time.time() - start_loss_accum

    # final: average over batch size
    loss = loss / float(B)
    
    # Return based on flags
    if profile_time and return_internals:
        internals = {'m': m_dict, 't': t_dict, 'f': f_dict, 'g': g_dict}
        return loss, time_profile, internals
    elif profile_time:
        return loss, time_profile
    elif return_internals:
        internals = {'m': m_dict, 't': t_dict, 'f': f_dict, 'g': g_dict}
        return loss, internals
    else:
        return loss
