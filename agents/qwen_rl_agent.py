"""
Qwen RL Agent for Financial Trading

Uses Qwen 0.5B as a transformer-based policy network trained with RL
on financial trading environments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenTradingPolicy(nn.Module):
    """
    Qwen 0.5B adapted for RL trading tasks.
    Uses the transformer as a policy network with financial observations as input.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        obs_dim: int = 512,
        action_dim: int = 3,
        hidden_dim: int = 1024,
        device: str = "auto"
    ):
        super().__init__()

        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Load Qwen model
        logger.info(f"Loading Qwen model: {model_name}")

        try:
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.qwen_model = self.qwen_model.to(self.device)

            # Freeze most layers initially, only fine-tune later layers
            for param in self.qwen_model.parameters():
                param.requires_grad = False

            # Unfreeze the last few layers for fine-tuning
            for param in self.qwen_model.model.layers[-2:].parameters():
                param.requires_grad = True

        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise

        # Get hidden size from Qwen config
        qwen_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.qwen_hidden_size = qwen_config.hidden_size

        # Observation encoder - transform financial obs to text-like input
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.qwen_hidden_size),
            nn.LayerNorm(self.qwen_hidden_size)
        )

        # Policy head - map Qwen outputs to actions
        self.policy_head = nn.Sequential(
            nn.Linear(self.qwen_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value head for critic (if using actor-critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.qwen_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # Token embedding for numerical data
        self.num_token_embed = nn.Embedding(1000, self.qwen_hidden_size)  # For discretized values

        logger.info(f"Qwen policy initialized on {self.device}")

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode financial observations into a format Qwen can process.
        Discretize continuous values and create token-like inputs.
        """
        batch_size = obs.shape[0]

        # Discretize observations to token-like representations
        # Clip and scale to reasonable token range
        obs_clipped = torch.clamp(obs, -10, 10)
        obs_scaled = ((obs_clipped + 10) / 20 * 999).long()  # Scale to 0-999
        obs_scaled = torch.clamp(obs_scaled, 0, 999)

        # Create embedding for each dimension
        obs_embeds = self.num_token_embed(obs_scaled)  # [batch, obs_dim, hidden_size]

        # Add positional information
        positions = torch.arange(obs.shape[1], device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.num_token_embed(positions % 1000)

        # Combine observation and position embeddings
        combined_embeds = obs_embeds + pos_embeds.unsqueeze(-1)

        # Aggregate across observation dimensions
        encoded_obs = combined_embeds.mean(dim=1)  # [batch, hidden_size]

        # Additional encoding layer
        encoded_obs = self.obs_encoder(obs)  # [batch, hidden_size]

        return encoded_obs

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Qwen-based policy.

        Args:
            obs: Financial observations [batch, obs_dim]

        Returns:
            action_logits: Action probabilities [batch, action_dim]
            value: State value estimate [batch, 1]
        """
        batch_size = obs.shape[0]

        # Encode observations
        encoded_obs = self.encode_observation(obs)  # [batch, hidden_size]

        # Create input for Qwen (reshape for transformer input)
        # We'll use the encoded observation as a "prompt" for Qwen
        inputs_embeds = encoded_obs.unsqueeze(1)  # [batch, 1, hidden_size]

        # Get attention mask
        attention_mask = torch.ones(batch_size, 1, device=self.device)

        # Forward through Qwen
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # Get action logits and value
        action_logits = self.policy_head(last_hidden_state)  # [batch, action_dim]
        value = self.value_head(last_hidden_state)  # [batch, 1]

        return action_logits, value.squeeze(-1)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Get action from policy.

        Args:
            obs: Single observation [obs_dim]
            deterministic: Whether to sample or take argmax

        Returns:
            action: Selected action
            value: State value estimate
        """
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_logits, value = self.forward(obs_tensor)

            action_probs = torch.softmax(action_logits, dim=-1)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                action = torch.multinomial(action_probs, 1).item()

            return action, value.item()

    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for training.
        """
        action_logits, value = self.forward(obs)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, value

class QwenPPOAgent:
    """
    PPO agent using Qwen as the policy network.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 1e-5,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        batch_size: int = 64,
        ppo_epochs: int = 4,
        device: str = "auto"
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize policy
        self.policy = QwenTradingPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        ).to(self.device)

        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Storage for trajectories
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
        self.done_buffer = []

        logger.info(f"Qwen PPO Agent initialized on {self.device}")

    def store_transition(self, obs, action, reward, value, logprob, done):
        """Store a transition in the buffer."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.logprob_buffer.append(logprob)
        self.done_buffer.append(done)

    def compute_gae(self):
        """Compute Generalized Advantage Estimation."""
        values = np.array(self.value_buffer)
        rewards = np.array(self.reward_buffer)
        dones = np.array(self.done_buffer)

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # Compute returns and advantages
        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage

            last_return = rewards[t] + self.gamma * next_value * (1 - dones[t])
            returns[t] = last_return

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        """Update policy using PPO."""
        if len(self.obs_buffer) < self.batch_size:
            return

        # Compute GAE
        advantages, returns = self.compute_gae()

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        action_tensor = torch.LongTensor(np.array(self.action_buffer)).to(self.device)
        old_logprob_tensor = torch.FloatTensor(np.array(self.logprob_buffer)).to(self.device)
        advantage_tensor = torch.FloatTensor(advantages).to(self.device)
        return_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO update
        for epoch in range(self.ppo_epochs):
            # Sample random minibatches
            indices = np.random.permutation(len(obs_tensor))

            for start in range(0, len(obs_tensor), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]

                mb_obs = obs_tensor[mb_indices]
                mb_actions = action_tensor[mb_indices]
                mb_old_logprobs = old_logprob_tensor[mb_indices]
                mb_advantages = advantage_tensor[mb_indices]
                mb_returns = return_tensor[mb_indices]

                # Evaluate current policy
                _, new_logprob, new_value = self.policy.get_action_and_logprob(mb_obs)
                entropy = torch.distributions.Categorical(
                    torch.softmax(self.policy.policy_head(
                        self.policy.encode_observation(mb_obs)
                    ), dim=-1)
                ).entropy().mean()

                # Compute ratio
                ratio = torch.exp(new_logprob - mb_old_logprobs)

                # PPO loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_value, mb_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Clear buffer
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.value_buffer.clear()
        self.logprob_buffer.clear()
        self.done_buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

def create_qwen_agent(env, learning_rate: float = 1e-5) -> QwenPPOAgent:
    """
    Create a Qwen PPO agent for the given environment.

    Args:
        env: The trading environment
        learning_rate: Learning rate for optimization

    Returns:
        Configured Qwen PPO agent
    """
    obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.shape[0]

    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    agent = QwenPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=learning_rate
    )

    return agent