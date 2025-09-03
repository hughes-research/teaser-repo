"""Hormone GNN Controller - Core Neural Control System
====================================================

A biologically-inspired Graph Neural Network controller that uses synthetic hormones
to regulate control signals. This system demonstrates how neural networks can be
enhanced with hormone-like regulatory mechanisms for stable, adaptive control.

The system consists of:
- Synthetic hormone dynamics with decay and impulse mechanisms
- Knob configuration for translating hormone states to control signals  
- Graph Convolutional Network for value function approximation
- Policy gradient learning for adaptive control parameter updates

Example:
    Basic usage of the hormone GNN controller:
    
    >>> import torch
    >>> from gnn_controller import HormoneState, KnobConfig, ValueGCN
    >>> 
    >>> # Initialize hormone system
    >>> hormones = HormoneState(["curiosity", "stress"])
    >>> hormones.add_impulses({"curiosity": 0.3, "stress": 0.1})
    >>> 
    >>> # Create control configuration
    >>> knob_cfg = KnobConfig(
    ...     baseline=0.75,
    ...     deltas={"curiosity": 0.2, "stress": -0.3},
    ...     powers={"curiosity": 1.0, "stress": 1.0},
    ...     weight_fn=lambda h: {"curiosity": 1.0, "stress": 1.0}
    ... )
    >>> 
    >>> # Generate control signal
    >>> control_signal = knob_cfg.propose(hormones.activations)

Authors:
    Resilient AI Research July 2025

License:
    Apache License 2.0 - See LICENSE file for details
"""

from __future__ import annotations

import math
import random
import argparse
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Callable, Tuple, Optional, Union, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Type aliases for clarity
HormoneActivations = Dict[str, float]
HormoneHalfLives = Dict[str, float]  
HormoneImpulses = Dict[str, float]
WeightFunction = Callable[[HormoneActivations], Dict[str, float]]


@dataclass
class HormoneState:
    """Manages synthetic hormone levels with biological-inspired dynamics.
    
    This class simulates hormone-like substances that decay over time and can
    receive impulse inputs. Each hormone has an activation level between 0.0 and 1.0.
    
    Attributes:
        names: List of hormone names to track
        activations: Current activation levels for each hormone (0.0 to 1.0)
        
    Example:
        >>> hormones = HormoneState(["dopamine", "cortisol"])
        >>> hormones.add_impulses({"dopamine": 0.5})
        >>> hormones.decay({"dopamine": 30.0, "cortisol": 45.0}, dt=1.0)
    """
    
    names: List[str]
    activations: HormoneActivations = field(init=False)

    def __post_init__(self) -> None:
        """Initialize hormone activations to zero."""
        if not self.names:
            raise ValueError("At least one hormone name must be provided")
        
        for name in self.names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Hormone name must be a non-empty string, got: {name}")
        
        self.activations = {hormone: 0.0 for hormone in self.names}

    def decay(self, half_lives: HormoneHalfLives, dt: float) -> None:
        """Apply exponential decay to hormone levels.
        
        Each hormone decays according to its half-life using the formula:
        new_level = current_level * exp(-ln(2) * dt / half_life)
        
        Args:
            half_lives: Dictionary mapping hormone names to their half-life values
            dt: Time step for decay calculation
            
        Raises:
            ValueError: If dt is negative or half_lives contains invalid values
            KeyError: If half_lives doesn't contain all hormone names
        """
        if dt < 0:
            raise ValueError(f"Time step dt must be non-negative, got: {dt}")
        
        for hormone, tau in half_lives.items():
            if hormone not in self.activations:
                raise KeyError(f"Unknown hormone '{hormone}' in half_lives")
            
            if tau <= 0:
                raise ValueError(f"Half-life must be positive, got {tau} for hormone '{hormone}'")
            
            decay_rate = math.log(2.0) / tau
            self.activations[hormone] *= math.exp(-decay_rate * dt)

    def add_impulses(self, impulses: HormoneImpulses) -> None:
        """Add impulse inputs to hormone levels.
        
        Impulses are added to current levels, with the result clamped to [0.0, 1.0].
        
        Args:
            impulses: Dictionary mapping hormone names to impulse magnitudes
            
        Raises:
            KeyError: If impulses contains unknown hormone names
            ValueError: If impulse values are invalid
        """
        for hormone, delta in impulses.items():
            if hormone not in self.activations:
                raise KeyError(f"Unknown hormone '{hormone}' in impulses")
            
            if not isinstance(delta, (int, float)) or math.isnan(delta):
                raise ValueError(f"Impulse value must be a valid number, got: {delta}")
            
            self.activations[hormone] = max(0.0, min(1.0, self.activations[hormone] + delta))


@dataclass
class KnobConfig:
    """Configuration for translating hormone states into control signals.
    
    This class defines how hormone activations are combined and transformed
    into a single control signal (the "knob" value). It supports:
    - Baseline offset
    - Per-hormone scaling factors (deltas)  
    - Non-linear hormone transformations (powers)
    - Weighted combination of hormone contributions
    
    Attributes:
        baseline: Base control value when all hormones are at zero
        deltas: Scaling factors for each hormone's contribution
        powers: Exponents for non-linear hormone transformations
        weight_fn: Function to compute hormone weights for combination
        min_val: Minimum allowable control value
        max_val: Maximum allowable control value
        
    Example:
        >>> def weight_func(h):
        ...     return {"curiosity": 1.0, "stress": 1.0 + h["stress"]}
        >>> 
        >>> config = KnobConfig(
        ...     baseline=0.5,
        ...     deltas={"curiosity": 0.3, "stress": -0.2}, 
        ...     powers={"curiosity": 1.0, "stress": 1.2},
        ...     weight_fn=weight_func
        ... )
    """
    
    baseline: float
    deltas: Dict[str, float]
    powers: Dict[str, float]
    weight_fn: WeightFunction
    min_val: float = 0.0
    max_val: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.baseline, (int, float)) or math.isnan(self.baseline):
            raise ValueError(f"Baseline must be a valid number, got: {self.baseline}")
        
        if self.min_val >= self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be less than max_val ({self.max_val})")
        
        if not self.deltas:
            raise ValueError("At least one delta value must be provided")
        
        # Validate deltas and powers have matching keys
        delta_keys = set(self.deltas.keys())
        power_keys = set(self.powers.keys())
        
        if not delta_keys.issubset(power_keys):
            missing = delta_keys - power_keys
            raise ValueError(f"Missing power values for hormones: {missing}")

    def propose(self, hormone_levels: HormoneActivations) -> float:
        """Generate control signal from current hormone levels.
        
        The control signal is computed as:
        1. Transform each hormone level by its power: h_i^p_i
        2. Scale by delta: baseline + delta_i * h_i^p_i  
        3. Weight each contribution using weight_fn
        4. Compute weighted average
        5. Clamp to [min_val, max_val]
        
        Args:
            hormone_levels: Current hormone activation levels
            
        Returns:
            Control signal value clamped to [min_val, max_val]
            
        Raises:
            KeyError: If required hormones are missing from hormone_levels
            ValueError: If hormone levels contain invalid values
        """
        if not hormone_levels:
            warnings.warn("Empty hormone_levels provided, using baseline only")
            return max(self.min_val, min(self.max_val, self.baseline))
        
        # Validate hormone levels
        for hormone, level in hormone_levels.items():
            if not isinstance(level, (int, float)) or math.isnan(level):
                raise ValueError(f"Invalid hormone level for '{hormone}': {level}")
            if not (0.0 <= level <= 1.0):
                warnings.warn(f"Hormone level for '{hormone}' outside [0,1]: {level}")
        
        # Compute raw contributions
        raw_contributions = {}
        for hormone_name, delta in self.deltas.items():
            if hormone_name not in hormone_levels:
                raise KeyError(f"Required hormone '{hormone_name}' not found in hormone_levels")
            
            level = hormone_levels[hormone_name]
            power = self.powers.get(hormone_name, 1.0)
            
            # Apply power transformation
            try:
                transformed_level = level ** power
            except (OverflowError, ZeroDivisionError):
                warnings.warn(f"Power transformation failed for {hormone_name}, using level=1.0")
                transformed_level = 1.0
            
            raw_contributions[hormone_name] = self.baseline + delta * transformed_level
        
        # Get weights and compute weighted average
        try:
            weights = self.weight_fn(hormone_levels)
        except Exception as e:
            warnings.warn(f"Weight function failed: {e}, using equal weights")
            weights = {name: 1.0 for name in self.deltas.keys()}
        
        # Validate weights
        total_weight = 0.0
        weighted_sum = 0.0
        
        for hormone_name in raw_contributions.keys():
            weight = weights.get(hormone_name, 1.0)
            if not isinstance(weight, (int, float)) or math.isnan(weight) or weight < 0:
                warnings.warn(f"Invalid weight for '{hormone_name}': {weight}, using 1.0")
                weight = 1.0
            
            total_weight += weight
            weighted_sum += weight * raw_contributions[hormone_name]
        
        if total_weight == 0:
            warnings.warn("Total weight is zero, using baseline")
            final_value = self.baseline
        else:
            final_value = weighted_sum / total_weight
        
        return max(self.min_val, min(self.max_val, final_value))


@dataclass
class Experience:
    """Single experience tuple for reinforcement learning.
    
    Represents one timestep of interaction with the environment.
    
    Attributes:
        state_vec: Hormone activation levels at this timestep
        action_val: Control signal value that was applied
        reward: Reward received from the environment
    """
    
    state_vec: HormoneActivations
    action_val: float
    reward: float

    def __post_init__(self) -> None:
        """Validate experience data."""
        if not isinstance(self.action_val, (int, float)) or math.isnan(self.action_val):
            raise ValueError(f"Invalid action value: {self.action_val}")
        
        if not isinstance(self.reward, (int, float)) or math.isnan(self.reward):
            raise ValueError(f"Invalid reward value: {self.reward}")


class ReplayBuffer:
    """Fixed-size buffer for storing and sampling experiences.
    
    Implements a circular buffer that automatically overwrites oldest
    experiences when full. Used for experience replay in reinforcement learning.
    
    Attributes:
        maxlen: Maximum number of experiences to store
        buf: Internal deque storing experiences
        
    Example:
        >>> buffer = ReplayBuffer(maxlen=1000)
        >>> exp = Experience({"hormone1": 0.5}, 0.8, 1.0)
        >>> buffer.push(exp)
        >>> batch = buffer.sample_batch(32)
    """

    def __init__(self, maxlen: int = 10_000) -> None:
        """Initialize replay buffer.
        
        Args:
            maxlen: Maximum buffer size
            
        Raises:
            ValueError: If maxlen is not positive
        """
        if maxlen <= 0:
            raise ValueError(f"Buffer size must be positive, got: {maxlen}")
        
        self.maxlen = maxlen
        self.buf: deque[Experience] = deque(maxlen=maxlen)

    def push(self, experience: Experience) -> None:
        """Add experience to buffer.
        
        Args:
            experience: Experience tuple to add
        """
        if not isinstance(experience, Experience):
            raise TypeError(f"Expected Experience object, got: {type(experience)}")
        
        self.buf.append(experience)

    def sample_batch(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of randomly sampled experiences
            
        Raises:
            ValueError: If batch_size is invalid
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {batch_size}")
        
        available = len(self.buf)
        if available == 0:
            return []
        
        sample_size = min(batch_size, available)
        return random.sample(list(self.buf), sample_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buf)


@dataclass
class DeltaParams:
    """Learnable parameters for hormone-to-control mapping.
    
    These parameters determine how much each hormone affects the control signal.
    They are updated via policy gradient methods during learning.
    
    Attributes:
        deltas: Current delta values for each hormone
        lr: Learning rate for parameter updates
        
    Example:
        >>> params = DeltaParams(
        ...     deltas={"curiosity": 0.2, "stress": -0.3},
        ...     lr=0.01
        ... )
        >>> grads = {"curiosity": 0.1, "stress": -0.05}
        >>> params.pg_update(grads, advantage=1.5)
    """
    
    deltas: Dict[str, float]
    lr: float = 1e-2

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got: {self.lr}")
        
        if not self.deltas:
            raise ValueError("At least one delta parameter must be provided")
        
        for name, value in self.deltas.items():
            if not isinstance(value, (int, float)) or math.isnan(value):
                raise ValueError(f"Invalid delta value for '{name}': {value}")

    def pg_update(self, gradients: Dict[str, float], advantage: float) -> None:
        """Update parameters using policy gradient.
        
        Applies the update: delta += lr * advantage * gradient
        
        Args:
            gradients: Gradient for each parameter
            advantage: Advantage estimate from value function
            
        Raises:
            KeyError: If gradients contains unknown parameter names
            ValueError: If advantage is invalid
        """
        if not isinstance(advantage, (int, float)) or math.isnan(advantage):
            raise ValueError(f"Invalid advantage value: {advantage}")
        
        for param_name, gradient in gradients.items():
            if param_name not in self.deltas:
                raise KeyError(f"Unknown parameter '{param_name}' in gradients")
            
            if not isinstance(gradient, (int, float)) or math.isnan(gradient):
                warnings.warn(f"Invalid gradient for '{param_name}': {gradient}, skipping update")
                continue
            
            self.deltas[param_name] += self.lr * advantage * gradient


@dataclass  
class GraphBatch:
    """Graph data structure for GCN processing.
    
    Represents a batch of graph data with node features and edge connectivity.
    
    Attributes:
        x: Node feature matrix [num_nodes, num_features]
        edge_index: Edge connectivity in COO format [2, num_edges]
    """
    
    x: torch.Tensor
    edge_index: torch.LongTensor

    def __post_init__(self) -> None:
        """Validate graph data."""
        if not isinstance(self.x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got: {type(self.x)}")
        
        if not isinstance(self.edge_index, torch.LongTensor):
            raise TypeError(f"edge_index must be a torch.LongTensor, got: {type(self.edge_index)}")
        
        if self.x.dim() != 2:
            raise ValueError(f"x must be 2D tensor, got shape: {self.x.shape}")
        
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be [2, num_edges] tensor, got shape: {self.edge_index.shape}")

    @property
    def num_nodes(self) -> int:
        """Return number of nodes in the graph."""
        return self.x.size(0)

    @property 
    def num_edges(self) -> int:
        """Return number of edges in the graph."""
        return self.edge_index.size(1)


def fully_connected_edge_index(num_nodes: int, device: Optional[torch.device] = None) -> torch.LongTensor:
    """Create edge index for fully connected graph.
    
    Args:
        num_nodes: Number of nodes in the graph
        device: Device to create tensor on
        
    Returns:
        Edge index tensor [2, num_nodes^2] representing all possible connections
        
    Raises:
        ValueError: If num_nodes is not positive
    """
    if num_nodes <= 0:
        raise ValueError(f"Number of nodes must be positive, got: {num_nodes}")
    
    row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes, device=device).repeat(num_nodes)
    return torch.stack([row, col], dim=0)


class GCNLayer(nn.Module):
    """Single layer of Graph Convolutional Network.
    
    Implements the GCN operation: X' = D^(-1/2) A D^(-1/2) X W
    where A is the adjacency matrix, D is the degree matrix, and W are learnable weights.
    
    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node
        
    Example:
        >>> layer = GCNLayer(16, 32)
        >>> x = torch.randn(10, 16)  # 10 nodes, 16 features each
        >>> edge_index = fully_connected_edge_index(10)
        >>> output = layer(x, edge_index)  # [10, 32]
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize GCN layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            
        Raises:
            ValueError: If feature dimensions are not positive
        """
        super().__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive")
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with small random values
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """Forward pass through GCN layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
            
        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected x shape [*, {self.in_features}], got: {x.shape}")
        
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index shape [2, *], got: {edge_index.shape}")
        
        num_nodes = x.size(0)
        row, col = edge_index
        
        # Validate edge indices
        if torch.any(row >= num_nodes) or torch.any(col >= num_nodes):
            raise ValueError("Edge indices contain out-of-bounds node indices")
        
        if torch.any(row < 0) or torch.any(col < 0):
            raise ValueError("Edge indices contain negative values")
        
        # Compute node degrees
        degree = torch.bincount(row, minlength=num_nodes).float()
        
        # Compute normalization: D^(-1/2)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0.0
        
        # Apply normalization to edges
        edge_weight = degree_inv_sqrt[row] * degree_inv_sqrt[col]
        
        # Message passing: aggregate neighbor features
        out = torch.zeros(num_nodes, self.out_features, device=x.device, dtype=x.dtype)
        messages = (x[col] @ self.weight) * edge_weight.unsqueeze(1)
        out.index_add_(0, row, messages)
        
        return out


class ValueGCN(nn.Module):
    """Two-layer Graph Convolutional Network for value function approximation.
    
    Processes hormone graph to estimate state values for reinforcement learning.
    Architecture: GCN -> ReLU -> GCN -> Global Mean Pool -> Scalar Output
    
    Args:
        num_hormones: Number of hormone types (not used directly, inferred from input)
        hidden_dim: Hidden layer dimension
        
    Example:
        >>> value_net = ValueGCN(num_hormones=2, hidden_dim=16)
        >>> graph = GraphBatch(x=torch.randn(3, 1), edge_index=edge_index)
        >>> value = value_net(graph)  # Scalar value estimate
    """

    def __init__(self, num_hormones: int, hidden_dim: int = 16) -> None:
        """Initialize value network.
        
        Args:
            num_hormones: Number of hormone types (for documentation)
            hidden_dim: Hidden layer dimension
            
        Raises:
            ValueError: If dimensions are not positive
        """
        super().__init__()
        
        if num_hormones <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")
        
        self.num_hormones = num_hormones
        self.hidden_dim = hidden_dim
        
        # Two-layer GCN
        self.gcn1 = GCNLayer(1, hidden_dim)  # Input: 1D hormone levels
        self.gcn2 = GCNLayer(hidden_dim, 1)   # Output: 1D values

    def forward(self, graph_batch: GraphBatch) -> torch.Tensor:
        """Forward pass through value network.
        
        Args:
            graph_batch: Graph containing hormone states
            
        Returns:
            Scalar value estimate
            
        Raises:
            ValueError: If graph_batch is malformed
        """
        if not isinstance(graph_batch, GraphBatch):
            raise TypeError(f"Expected GraphBatch, got: {type(graph_batch)}")
        
        x, edge_index = graph_batch.x, graph_batch.edge_index
        
        # First GCN layer with ReLU activation
        x = F.relu(self.gcn1(x, edge_index))
        
        # Second GCN layer  
        x = self.gcn2(x, edge_index)
        
        # Global mean pooling to get scalar output
        return x.mean()


def build_graph(
    hormone_activations: HormoneActivations, 
    device: Optional[torch.device] = None
) -> GraphBatch:
    """Build graph representation from hormone activations.
    
    Creates a fully connected graph where:
    - Each hormone becomes a node with its activation as the feature
    - An additional global node represents the mean activation
    - All nodes are connected to all other nodes
    
    Args:
        hormone_activations: Dictionary of hormone names to activation levels
        device: Device to create tensors on
        
    Returns:
        GraphBatch representing the hormone state
        
    Raises:
        ValueError: If hormone_activations is empty or contains invalid values
    """
    if not hormone_activations:
        raise ValueError("hormone_activations cannot be empty")
    
    # Validate and extract activation values
    activations = []
    for name, value in hormone_activations.items():
        if not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"Invalid activation for hormone '{name}': {value}")
        activations.append(value)
    
    # Create node features: [num_hormones + 1, 1]
    hormone_features = torch.tensor(activations, dtype=torch.float32, device=device).unsqueeze(1)
    global_feature = hormone_features.mean(dim=0, keepdim=True)
    
    # Concatenate hormone nodes with global node
    x = torch.cat([hormone_features, global_feature], dim=0)
    
    # Create fully connected graph
    num_nodes = x.size(0)
    edge_index = fully_connected_edge_index(num_nodes, device)
    
    return GraphBatch(x, edge_index)


def weight_rule(hormone_activations: HormoneActivations) -> Dict[str, float]:
    """Default weight function for hormone combination.
    
    Implements stress-adaptive weighting where stress increases its own weight.
    This creates a feedback mechanism where high stress makes the system more
    stress-sensitive.
    
    Args:
        hormone_activations: Current hormone levels
        
    Returns:
        Dictionary of weights for each hormone
    """
    weights = {}
    
    # Base weights
    for hormone in hormone_activations:
        weights[hormone] = 1.0
    
    # Stress-adaptive weighting: high stress increases stress influence
    if "stress" in hormone_activations:
        stress_level = hormone_activations["stress"]
        weights["stress"] = 1.0 + 2.0 * stress_level
    
    return weights


def demo(
    num_ticks: int, 
    device: torch.device, 
    show_plot: bool = False,
    target_range: Tuple[float, float] = (0.7, 0.9)
) -> None:
    """Demonstrate the hormone GNN controller.
    
    Runs a simulation where the controller learns to maintain a control signal
    within a target range while experiencing random hormone impulses.
    
    Args:
        num_ticks: Number of simulation steps
        device: PyTorch device for computation
        show_plot: Whether to display performance plot
        target_range: Target range for control signal (min, max)
        
    Raises:
        ValueError: If parameters are invalid
    """
    if num_ticks <= 0:
        raise ValueError(f"num_ticks must be positive, got: {num_ticks}")
    
    if target_range[0] >= target_range[1]:
        raise ValueError(f"Invalid target range: {target_range}")
    
    print(f"Running demo with {num_ticks} ticks, target range: {target_range}")
    print("=" * 60)
    
    # Initialize system components
    hormones = HormoneState(["curiosity", "stress"])
    half_lives = {"curiosity": 30.0, "stress": 45.0}

    delta_params = DeltaParams({"curiosity": +0.25, "stress": -0.35})
    knob_config = KnobConfig(
        baseline=0.70, 
        deltas=delta_params.deltas, 
        powers={"curiosity": 1.2, "stress": 1.0}, 
        weight_fn=weight_rule, 
        min_val=0.1, 
        max_val=1.0
    )

    value_net = ValueGCN(num_hormones=len(hormones.names)).to(device)
    optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Tracking
    rewards = []
    dt = 1.0

    # Simulation loop
    for tick in range(1, num_ticks + 1):
        # Natural hormone decay
        hormones.decay(half_lives, dt)
        
        # Random impulses (simulate external events)
        if random.random() < 0.05:  # 5% chance
            hormones.add_impulses({"curiosity": 0.25})
        if random.random() < 0.02:  # 2% chance  
            hormones.add_impulses({"stress": 0.40})

        # Generate control signal
        knob_value = knob_config.propose(hormones.activations)
        
        # Environment feedback (reward for staying in target range)
        reward = 1.0 if target_range[0] <= knob_value <= target_range[1] else -1.0
        rewards.append(reward)

        # Value function estimation
        graph = build_graph(hormones.activations, device)
        baseline_value = value_net(graph)
        advantage = reward - baseline_value.item()

        # Policy gradient update
        gradients = {k: (knob_value - knob_config.baseline) for k in delta_params.deltas}
        delta_params.pg_update(gradients, advantage)

        # Value network update
        loss = F.mse_loss(baseline_value, torch.tensor(reward, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodic logging
        if tick % 5000 == 0 or tick == 1:
            print(f"Tick {tick:6d} | Reward: {reward:+.0f} | Knob: {knob_value:.3f} | "
                  f"Curiosity: {hormones.activations['curiosity']:.2f} | "
                  f"Stress: {hormones.activations['stress']:.2f}")

    # Results summary
    success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    print(f"\nDemo Results:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Final Knob Value: {knob_value:.3f}")
    print(f"Final Delta Params: {delta_params.deltas}")

    # Optional visualization
    if show_plot and plt is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(rewards, linewidth=0.8, alpha=0.7)
        plt.title("Control Performance Over Time")
        plt.xlabel("Tick")
        plt.ylabel("Reward (+1 = Success, -1 = Failure)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    elif show_plot:
        print("Matplotlib not available - skipping plot")


def main() -> None:
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="Hormone-based GNN Controller Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ticks", 
        type=int, 
        default=20000, 
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--cuda", 
        action="store_true", 
        help="Use CUDA if available"
    )
    parser.add_argument(
        "--plot", 
        action="store_true", 
        help="Show performance plot"
    )
    
    args = parser.parse_args()

    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Run demo
    try:
        demo(args.ticks, device, args.plot)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
