"""Neural network model to learn selecting nodes to remove from VRP solutions."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Main model for selecting nodes to remove from VRP solutions.

    Architecture:
    - Encoder: Encodes problem instance and current solution structure
    - Decoder: Autoregressively selects nodes to remove
    """

    def __init__(self, **model_params):
        """
        Initialize model.

        Args:
            **model_params: Model configuration parameters
        """
        super().__init__()
        self.model_params = model_params
        self.problem = model_params["problem"]
        embedding_dim = model_params["embedding_dim"]

        # Encoder and decoder
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        # Cached encoded nodes
        self.encoded_nodes = None

        # Learnable start token for decoder
        self.start_last_node = nn.Parameter(
            torch.zeros(embedding_dim), requires_grad=True
        )

    def pre_forward(self, reset_state, z: torch.Tensor) -> None:
        """
        Encode problem instance and solution structure.

        Args:
            reset_state: State containing problem features and solution structure
            z: Latent seed vectors for conditioning
        """
        device = z.device

        # Extract and move features to device
        depot_xy = reset_state.problem_feat.depot_xy.to(device)
        node_xy = reset_state.problem_feat.node_xy.to(device)
        node_demand = reset_state.problem_feat.node_demand.to(device)

        # Build node features (coordinates + demand)
        node_feat = torch.cat((node_xy, node_demand[:, :, None]), dim=2)

        # Add problem-specific features
        if self.problem == "vrptw":
            node_feat = torch.cat(
                (node_feat, reset_state.problem_feat.node_tw.to(device)), dim=2
            )
        elif self.problem == "pcvrp":
            node_prizes = reset_state.problem_feat.node_prizes.to(device)
            node_feat = torch.cat((node_feat, node_prizes[:, :, None]), dim=2)

        # Solution structure
        solution_neighbours = reset_state.neighbours.to(device)
        tour_index = reset_state.tour_index.to(device)

        # Encode
        self.encoded_nodes = self.encoder(
            depot_xy, node_feat, solution_neighbours, tour_index
        )
        self.decoder.set_kv(self.encoded_nodes, z)

    def forward(
        self, state, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Select next node to remove.

        Args:
            state: Current selection state
            temperature: Softmax temperature

        Returns:
            Tuple of (selected nodes, probabilities, all probabilities)
        """
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)

        # Get encoding of last selected node (or start token)
        if state.current_node is None:
            encoded_last_node = self.start_last_node[None, None].expand(
                batch_size, rollout_size, -1
            )
        else:
            encoded_last_node = _gather_by_index(self.encoded_nodes, state.current_node)

        # Get selection probabilities
        probs = self.decoder(
            encoded_last_node, ninf_mask=state.ninf_mask, temperature=temperature
        )

        # Sample or take argmax
        if self.training or self.model_params["eval_type"] == "softmax":
            selected, prob = self._sample_from_probs(
                probs, state, batch_size, rollout_size
            )
        else:
            selected = probs.argmax(dim=2)
            prob = None

        return selected, prob, probs[:, :, 1:]  # Exclude depot from probs

    def _sample_from_probs(
        self, probs: torch.Tensor, state, batch_size: int, rollout_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from probability distribution (with retry for zero-probability bug).

        Args:
            probs: Selection probabilities
            state: Current state
            batch_size: Batch size
            rollout_size: Rollout size

        Returns:
            Tuple of (selected nodes, their probabilities)
        """
        # Workaround for PyTorch multinomial bug with zero probabilities
        while True:
            with torch.no_grad():
                selected = (
                    probs.reshape(batch_size * rollout_size, -1)
                    .multinomial(1)
                    .squeeze(dim=1)
                    .reshape(batch_size, rollout_size)
                )
            # Gather selected action probabilities with gradient flow
            prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected]
            # Use detached values for the zero-probability check to avoid creating large graphs
            if (prob.detach() != 0).all():
                break

        return selected, prob


def _gather_by_index(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather elements from tensor using indices.

    Args:
        tensor: Tensor to gather from, shape (batch, seq_len, dim)
        indices: Indices to gather, shape (batch, rollout)

    Returns:
        Gathered tensor, shape (batch, rollout, dim)
    """
    batch_size = indices.size(0)
    rollout_size = indices.size(1)
    embedding_dim = tensor.size(2)

    gathering_index = indices[:, :, None].expand(
        batch_size, rollout_size, embedding_dim
    )
    return tensor.gather(dim=1, index=gathering_index)


########################################
# ENCODER
########################################


class CVRP_Encoder(nn.Module):
    """
    Encoder for VRP instances and solution structures.

    Architecture:
    1. Initial embedding layers
    2. Self-attention layers
    3. Tour aggregation layer (optional)
    4. Message passing layers
    5. Final self-attention layers
    """

    def __init__(self, **model_params):
        """Initialize encoder."""
        super().__init__()
        self.model_params = model_params
        self.problem = model_params["problem"]
        self.embedding_dim = model_params["embedding_dim"]

        # Input embedding layers
        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_node = self._create_node_embedding()

        # Encoder layers (self-attention + feedforward)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(**model_params)
                for _ in range(model_params["encoder_layer_num"])
            ]
        )

        # Tour aggregation layer
        self.tour_layer = (
            TourLayer(**model_params) if model_params["tour_layer"] else None
        )

        # Message passing layers (leverage solution structure)
        self.mp_layers = nn.ModuleList(
            [
                MessagePassingLayer(**model_params)
                for _ in range(model_params["message_passing_layer_num"])
            ]
        )

        # Final encoder layers
        self.layers_2 = nn.ModuleList(
            [
                EncoderLayer(**model_params)
                for _ in range(model_params["encoder_layer_num_2"])
            ]
        )

    def _create_node_embedding(self) -> nn.Linear:
        """Create node embedding layer based on problem type."""
        problem_to_features = {
            "cvrp": 3,  # x, y, demand
            "vrptw": 5,  # x, y, demand, tw_start, tw_end
            "pcvrp": 4,  # x, y, demand, prize
        }

        num_features = problem_to_features.get(self.problem)
        if num_features is None:
            raise ValueError(f"Unsupported problem type: {self.problem}")

        return nn.Linear(num_features, self.embedding_dim)

    def forward(
        self,
        depot_xy: torch.Tensor,
        node_feat: torch.Tensor,
        solution_neighbours: torch.Tensor,
        tour_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode problem instance and solution.

        Args:
            depot_xy: Depot coordinates, shape (batch, 1, 2)
            node_feat: Node features, shape (batch, problem, feat_dim)
            solution_neighbours: Neighbour relationships, shape (batch, problem, 2)
            tour_index: Tour assignments, shape (batch, problem)

        Returns:
            Encoded representations, shape (batch, problem+1, embedding)
        """
        batch_size = node_feat.shape[0]
        num_customers = node_feat.shape[1]

        # Embed depot and nodes
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_feat)

        # Concatenate depot and customer embeddings
        out = torch.cat((embedded_depot, embedded_node), dim=1)

        # Initial self-attention layers
        for layer in self.layers:
            out = layer(out)

        # Tour aggregation (if enabled)
        if self.tour_layer is not None:
            out = self.tour_layer(batch_size, num_customers, tour_index, out)

        # Message passing (leverage solution structure)
        for layer in self.mp_layers:
            out = layer(batch_size, num_customers, solution_neighbours, out)

        # Final self-attention layers
        for layer in self.layers_2:
            out = layer(out)

        return out


class TourLayer(nn.Module):
    """
    Aggregates customer embeddings by tour to capture tour-level information.

    For each customer, computes a tour embedding by summing embeddings of all
    customers in the same tour, then combines with customer embedding.
    """

    def __init__(self, **model_params):
        """Initialize tour layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]

        self.embedding_dim = embedding_dim
        self.tour_combiner = nn.Linear(embedding_dim * 2, embedding_dim)
        self.feedforward_layer = nn.Linear(embedding_dim, embedding_dim)
        self.add_and_normalize = AddAndInstanceNormalization(**model_params)

    def forward(
        self,
        batch_size: int,
        num_customers: int,
        tour_index: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply tour aggregation.

        Args:
            batch_size: Batch size
            num_customers: Number of customers
            tour_index: Tour assignments, shape (batch, problem)
            embeddings: Node embeddings, shape (batch, problem+1, embedding)

        Returns:
            Updated embeddings, shape (batch, problem+1, embedding)
        """
        # Extract customer embeddings (exclude depot)
        customer_embeddings = embeddings[:, 1:]

        # Handle unvisited customers (tour_index == -1, e.g., in PCVRP)
        max_tour_idx = tour_index.max()
        has_unvisited = tour_index.min() == -1

        if has_unvisited:
            tour_index = tour_index.clone()
            tour_index[tour_index == -1] = max_tour_idx + 1
            max_tour_idx += 1

        # Initialize tour embeddings
        tour_embeddings = torch.zeros(
            batch_size,
            max_tour_idx + 1,
            self.embedding_dim,
            dtype=customer_embeddings.dtype,
            device=customer_embeddings.device,
        )

        # Aggregate customer embeddings by tour
        expand_dim = customer_embeddings.shape[2]
        tour_embeddings.scatter_add_(
            1, tour_index[:, :, None].expand(-1, -1, expand_dim), customer_embeddings
        )

        # Zero out dummy tour for unvisited customers
        if has_unvisited:
            tour_embeddings[:, -1] = 0

        # Gather tour embedding for each customer
        customer_tour_embeddings = torch.gather(
            tour_embeddings, 1, tour_index[:, :, None].expand(-1, -1, expand_dim)
        )

        # Combine customer and tour embeddings
        combined = torch.cat((customer_embeddings, customer_tour_embeddings), dim=2)
        combined = F.relu(self.tour_combiner(combined))
        combined = self.feedforward_layer(combined)

        # Residual connection with normalization
        updated_customers = self.add_and_normalize(customer_embeddings, combined)

        # Re-attach depot
        return torch.cat((embeddings[:, [0]], updated_customers), dim=1)


class MessagePassingLayer(nn.Module):
    """
    Message passing layer that leverages solution structure (neighbour relationships).

    For each customer, aggregates information from its left and right neighbours
    in the current solution.
    """

    def __init__(self, **model_params):
        """Initialize message passing layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]

        self.embedding_dim = embedding_dim
        self.directed_graph = model_params["problem"] == "vrptw"

        # Neighbour projection layers
        if self.directed_graph:
            self.left_neighbour_projector = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
            self.right_neighbour_projector = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
        else:
            self.neighbour_projector = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )

        # Combination layers
        self.neighbour_combiner = nn.Linear(embedding_dim * 2, embedding_dim)
        self.feedforward_layer = nn.Linear(embedding_dim, embedding_dim)
        self.add_and_normalize = AddAndInstanceNormalization(**model_params)

    def forward(
        self,
        batch_size: int,
        num_customers: int,
        solution_neighbours: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply message passing.

        Args:
            batch_size: Batch size
            num_customers: Number of customers
            solution_neighbours: Neighbour indices, shape (batch, problem, 2)
            embeddings: Node embeddings, shape (batch, problem+1, embedding)

        Returns:
            Updated embeddings, shape (batch, problem+1, embedding)
        """
        # Gather left and right neighbour embeddings
        left_neighbours = self._gather_neighbours(
            embeddings, solution_neighbours[:, :, 0], batch_size, num_customers
        )
        right_neighbours = self._gather_neighbours(
            embeddings, solution_neighbours[:, :, 1], batch_size, num_customers
        )

        # Project neighbour embeddings
        if self.directed_graph:
            left_neighbours = self.left_neighbour_projector(left_neighbours)
            right_neighbours = self.right_neighbour_projector(right_neighbours)
        else:
            left_neighbours = self.neighbour_projector(left_neighbours)
            right_neighbours = self.neighbour_projector(right_neighbours)

        # Aggregate neighbour information
        neighbour_info = left_neighbours + right_neighbours

        # Combine with customer embeddings
        customer_embeddings = embeddings[:, 1:]
        combined = torch.cat((customer_embeddings, neighbour_info), dim=2)
        combined = F.relu(self.neighbour_combiner(combined))
        combined = self.feedforward_layer(combined)

        # Residual connection with normalization
        updated_customers = self.add_and_normalize(customer_embeddings, combined)

        # Re-attach depot
        return torch.cat((embeddings[:, [0]], updated_customers), dim=1)

    def _gather_neighbours(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        num_customers: int,
    ) -> torch.Tensor:
        """Gather neighbour embeddings by indices."""
        return torch.gather(
            embeddings,
            1,
            indices[:, :, None].expand(batch_size, num_customers, self.embedding_dim),
        )


class EncoderLayer(nn.Module):
    """Standard Transformer encoder layer with multi-head attention and feedforward."""

    def __init__(self, **model_params):
        """Initialize encoder layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]

        # Multi-head attention
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Normalization and feedforward
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.head_num = head_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply encoder layer.

        Args:
            x: Input embeddings, shape (batch, seq_len, embedding)

        Returns:
            Output embeddings, shape (batch, seq_len, embedding)
        """
        # Multi-head self-attention
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)

        attn_out = fast_multi_head_attention(q, k, v)
        attn_out = self.multi_head_combine(attn_out)

        # First residual connection
        x = self.add_n_normalization_1(x, attn_out)

        # Feedforward
        ff_out = self.feed_forward(x)

        # Second residual connection
        x = self.add_n_normalization_2(x, ff_out)

        return x


########################################
# DECODER
########################################


class CVRP_Decoder(nn.Module):
    """
    Autoregressive decoder for selecting nodes to remove.

    Uses GRU to maintain decoding state and multi-head attention to attend
    to encoded nodes. Conditioned on seed vectors as described in PolyNet paper to increase diversity.
    """

    def __init__(self, **model_params):
        """Initialize decoder."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        poly_embedding_dim = model_params["poly_embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        z_dim = model_params["z_dim"]

        # Query, key, value projections
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # GRU for maintaining decoding state
        self.GRU = nn.GRUCell(embedding_dim, embedding_dim)

        # Polynomial network for conditioning on latent vectors
        self.poly_layer_1 = nn.Linear(embedding_dim + z_dim, poly_embedding_dim)
        self.poly_layer_2 = nn.Linear(poly_embedding_dim, embedding_dim)

        # Model parameters
        self.head_num = head_num
        self.sqrt_embedding_dim = model_params["sqrt_embedding_dim"]
        self.logit_clipping = model_params["logit_clipping"]

        # Cached values
        self.k = None
        self.v = None
        self.single_head_key = None
        self.z = None
        self.GRU_hidden = None

    def set_kv(self, encoded_nodes: torch.Tensor, z: torch.Tensor) -> None:
        """
        Set keys and values for attention.

        Args:
            encoded_nodes: Encoded node representations
            z: Latent seed vectors
        """
        self.k = reshape_by_heads(self.Wk(encoded_nodes), self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.z = z
        self.GRU_hidden = None

    def forward(
        self,
        encoded_last_node: torch.Tensor,
        ninf_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute selection probabilities.

        Args:
            encoded_last_node: Encoding of last selected node
            ninf_mask: Mask for invalid selections
            temperature: Softmax temperature

        Returns:
            Selection probabilities, shape (batch, rollout, problem+1)
        """
        batch_size = encoded_last_node.shape[0]
        rollout_size = encoded_last_node.shape[1]
        embedding_dim = encoded_last_node.shape[2]

        # Update GRU hidden state
        self.GRU_hidden = self.GRU(
            encoded_last_node.reshape(batch_size * rollout_size, embedding_dim),
            self.GRU_hidden,
        )
        context = self.GRU_hidden.reshape(batch_size, rollout_size, embedding_dim)

        # Multi-head attention
        q = reshape_by_heads(self.Wq_last(context), self.head_num)
        attn_out = fast_multi_head_attention(
            q, self.k, self.v, rank3_ninf_mask=ninf_mask
        )
        mh_out = self.multi_head_combine(attn_out)

        # Polynomial network (condition on latent vectors)
        poly_out = self.poly_layer_1(torch.cat((mh_out, self.z), dim=2))
        poly_out = F.relu(poly_out)
        poly_out = self.poly_layer_2(poly_out)

        # Add polynomial output
        context_out = mh_out + poly_out

        # Single-head attention for scoring
        scores = torch.matmul(context_out, self.single_head_key)
        scores = scores / self.sqrt_embedding_dim
        scores = self.logit_clipping * torch.tanh(scores)
        scores = scores + ninf_mask

        # Convert to probabilities
        probs = F.softmax(scores / temperature, dim=2)

        return probs


########################################
# UTILITY LAYERS AND FUNCTIONS
########################################


class AddAndInstanceNormalization(nn.Module):
    """Residual connection with instance normalization."""

    def __init__(self, **model_params):
        """Initialize normalization layer."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Apply residual connection with normalization.

        Args:
            input1: Original input
            input2: Transformed input

        Returns:
            Normalized sum
        """
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    """Two-layer feedforward network with ReLU activation."""

    def __init__(self, **model_params):
        """Initialize feedforward network."""
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward network."""
        return self.W2(F.relu(self.W1(x)))


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """
    Reshape tensor for multi-head attention.

    Args:
        qkv: Input tensor, shape (batch, seq_len, head_num * qkv_dim)
        head_num: Number of attention heads

    Returns:
        Reshaped tensor, shape (batch, head_num, seq_len, qkv_dim)
    """
    batch_size = qkv.size(0)
    seq_len = qkv.size(1)

    qkv = qkv.reshape(batch_size, seq_len, head_num, -1)
    return qkv.transpose(1, 2)


def fast_multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank3_ninf_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Efficient multi-head attention using PyTorch's scaled_dot_product_attention.

    Args:
        q: Queries, shape (batch, head_num, seq_len, qkv_dim)
        k: Keys, shape (batch, head_num, seq_len, qkv_dim)
        v: Values, shape (batch, head_num, seq_len, qkv_dim)
        rank3_ninf_mask: Optional attention mask

    Returns:
        Attention output, shape (batch, seq_len, head_num * qkv_dim)
    """
    batch_size = q.size(0)
    head_num = q.size(1)
    seq_len = q.size(2)
    qkv_dim = q.size(3)

    # Prepare mask if provided
    mask = None
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, seq_len, -1)

    # Efficient attention using PyTorch's fused kernel
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Reshape output
    out = out.transpose(1, 2)
    out = out.reshape(batch_size, seq_len, head_num * qkv_dim)

    return out
