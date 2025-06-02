import torch

class ClippedRPG:
    """
    Implements a Clipped Policy Gradient algorithm, similar to PPO.
    The reward structure is assumed to be:
    reward = delta_pnl - lambda_val * risk_penalty
    Advantages are calculated based on these rewards.
    """

    def __init__(self, epsilon: float = 0.2):
        """
        Initializes the ClippedRPG algorithm.

        Args:
            epsilon (float): The clipping parameter for the policy ratio.
        """
        self.epsilon = epsilon

    def calculate_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the policy gradient loss using a clipped objective.

        Args:
            old_log_probs (torch.Tensor): Log probabilities of actions taken by the old policy.
                                           Shape: (batch_size,)
            new_log_probs (torch.Tensor): Log probabilities of actions taken by the current policy.
                                           Shape: (batch_size,)
            advantages (torch.Tensor): Calculated advantage values for the actions taken.
                                       Shape: (batch_size,)

        Returns:
            torch.Tensor: The calculated policy gradient loss (a scalar tensor).
        """
        if not isinstance(old_log_probs, torch.Tensor):
            raise TypeError("old_log_probs must be a torch.Tensor")
        if not isinstance(new_log_probs, torch.Tensor):
            raise TypeError("new_log_probs must be a torch.Tensor")
        if not isinstance(advantages, torch.Tensor):
            raise TypeError("advantages must be a torch.Tensor")

        if not (old_log_probs.shape == new_log_probs.shape == advantages.shape):
            raise ValueError(
                "Shapes of old_log_probs, new_log_probs, and advantages must match."
            )
        if len(old_log_probs.shape) != 1:
             raise ValueError(
                "Input tensors (old_log_probs, new_log_probs, advantages) must be 1-dimensional (batch_size,)."
            )


        # Calculate the ratio r = exp(new_log_probs - old_log_probs)
        # Ensure numerical stability by subtracting the max of new_log_probs for the exponentiation
        # This is a common trick, but since we are subtracting log_probs, it might be less critical
        # than direct softmax. However, exp can still overflow for large positive differences.
        # log_r = new_log_probs - old_log_probs
        # r = torch.exp(log_r)
        r = torch.exp(new_log_probs - old_log_probs)

        # Calculate the clipped ratio: clipped_r = clamp(r, 1 - epsilon, 1 + epsilon)
        clipped_r = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon)

        # Policy gradient loss components
        loss_unclipped = r * advantages
        loss_clipped = clipped_r * advantages

        # The policy gradient loss is the negative mean of the minimum of the two components
        # We take the minimum because we want to discourage policy updates that are too large.
        # The negative sign is because optimizers typically minimize, and we want to maximize the objective.
        loss = -torch.mean(torch.min(loss_unclipped, loss_clipped))

        return loss

if __name__ == '__main__':
    # Example Usage (for basic testing)
    # These would typically come from your policy network and advantage calculation logic
    old_lp = torch.tensor([-0.5, -0.6, -0.7]) # Log probs from old policy
    new_lp = torch.tensor([-0.4, -0.55, -0.8]) # Log probs from current policy
    adv = torch.tensor([1.0, -0.5, 2.0])    # Calculated advantages

    agent = ClippedRPG(epsilon=0.2)
    loss_val = agent.calculate_loss(old_lp, new_lp, adv)
    print(f"Calculated Loss: {loss_val.item()}")

    # Test case with potential for clipping
    old_lp_clip = torch.tensor([-0.5, -0.5, -0.5])
    new_lp_clip_pos = torch.tensor([-0.1, -0.2, -0.1]) # ratio > 1 + eps for some
    new_lp_clip_neg = torch.tensor([-0.9, -0.8, -0.9]) # ratio < 1 - eps for some
    adv_clip = torch.tensor([1.0, 1.0, -1.0]) # positive and negative advantages

    print("\n--- Positive Advantage Clipping Test ---")
    # Here r will be exp(0.4) approx 1.49. 1+eps = 1.2. clipped_r = 1.2
    # r * adv = 1.49. clipped_r * adv = 1.2. min is 1.2
    loss_pos_adv = agent.calculate_loss(old_lp_clip, new_lp_clip_pos, torch.tensor([1.0, 1.0, 1.0]))
    print(f"Loss (Positive Advantage): {loss_pos_adv.item()}")
    # Expected: -(min(exp(0.4)*1, 1.2*1) + min(exp(0.3)*1, 1.2*1) + min(exp(0.4)*1, 1.2*1)) / 3
    # -(1.2 + exp(0.3) + 1.2)/3 = -(1.2 + 1.349 + 1.2)/3 = -3.749/3 = -1.249 (approx)
    # Actual calculation:
    # r1 = exp(0.4) = 1.4918, clipped_r1 = 1.2. term1 = 1.2 * 1.0 = 1.2
    # r2 = exp(0.3) = 1.3498, clipped_r2 = 1.2. term2 = 1.2 * 1.0 = 1.2
    # r3 = exp(0.4) = 1.4918, clipped_r3 = 1.2. term3 = 1.2 * 1.0 = 1.2
    # loss = -(1.2 + 1.2 + 1.2)/3 = -1.2.  There was an error in my manual trace for exp(0.3) not being clipped.
    # exp(0.3) is ~1.3498 which is > 1.2, so it's clipped.

    print("\n--- Negative Advantage Clipping Test (ratio > 1+eps) ---")
    # Here r will be exp(0.4) approx 1.49. 1+eps = 1.2. clipped_r = 1.2
    # r * adv = 1.49 * -1 = -1.49. clipped_r * adv = 1.2 * -1 = -1.2.
    # For negative advantages, we want min to pick the one that is *less negative* (larger value)
    # So min(-1.49, -1.2) = -1.49. This is what PPO does.
    loss_neg_adv_high_ratio = agent.calculate_loss(old_lp_clip, new_lp_clip_pos, torch.tensor([-1.0, -1.0, -1.0]))
    print(f"Loss (Negative Advantage, High Ratio): {loss_neg_adv_high_ratio.item()}")
    # Expected: -(min(exp(0.4)*-1, 1.2*-1) + min(exp(0.3)*-1, 1.2*-1) + min(exp(0.4)*-1, 1.2*-1)) / 3
    # -(min(-1.4918, -1.2) + min(-1.3498, -1.2) + min(-1.4918, -1.2)) / 3
    # -(-1.4918 -1.3498 -1.4918) / 3 = (1.4918 + 1.3498 + 1.4918)/3 = 4.3334 / 3 = 1.444 (approx)

    print("\n--- Positive Advantage Clipping Test (ratio < 1-eps) ---")
    # Here r will be exp(-0.4) approx 0.67. 1-eps = 0.8. clipped_r = 0.8
    # r * adv = 0.67 * 1 = 0.67. clipped_r * adv = 0.8 * 1 = 0.8
    # min(0.67, 0.8) = 0.67
    loss_pos_adv_low_ratio = agent.calculate_loss(old_lp_clip, new_lp_clip_neg, torch.tensor([1.0, 1.0, 1.0]))
    print(f"Loss (Positive Advantage, Low Ratio): {loss_pos_adv_low_ratio.item()}")
    # Expected: -(min(exp(-0.4)*1, 0.8*1) + min(exp(-0.3)*1, 0.8*1) + min(exp(-0.4)*1, 0.8*1)) / 3
    # -(exp(-0.4) + exp(-0.3) + exp(-0.4)) / 3
    # -(0.6703 + 0.7408 + 0.6703) / 3 = -2.0814 / 3 = -0.6938 (approx)

    print("\n--- Shape Mismatch Test ---")
    try:
        agent.calculate_loss(torch.randn(3,1), torch.randn(3,1), torch.randn(3,1))
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        agent.calculate_loss(torch.randn(3), torch.randn(2), torch.randn(3))
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Type Error Test ---")
    try:
        agent.calculate_loss([1.0,2.0], torch.randn(2), torch.randn(2))
    except TypeError as e:
        print(f"Caught expected error: {e}")
