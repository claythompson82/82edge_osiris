import torch
import pytest
from osiris_rl.algorithms.clipped_rpg import ClippedRPG

# Helper function for manual calculation if needed, or just direct calculation in tests
def manual_loss_calculation(old_log_probs, new_log_probs, advantages, epsilon):
    r = torch.exp(new_log_probs - old_log_probs)
    clipped_r = torch.clamp(r, 1 - epsilon, 1 + epsilon)
    loss_unclipped = r * advantages
    loss_clipped = clipped_r * advantages
    loss = -torch.mean(torch.min(loss_unclipped, loss_clipped))
    return loss

class TestClippedRPG:
    EPSILON = 0.2 # Default epsilon for ClippedRPG

    @pytest.fixture
    def agent(self):
        """Pytest fixture to create a ClippedRPG agent."""
        return ClippedRPG(epsilon=self.EPSILON)

    def test_loss_calculation_simple(self, agent):
        """
        Test basic loss calculation with simple, non-clipping values.
        """
        old_log_probs = torch.tensor([-0.5, -0.6])
        new_log_probs = torch.tensor([-0.4, -0.55]) # ratios: exp(0.1), exp(0.05) -> ~1.105, ~1.051
        advantages = torch.tensor([1.0, 2.0])       # Both ratios are < 1 + EPSILON (1.2)

        # Manual calculation for this case:
        # r1 = exp(0.1) = 1.10517
        # r2 = exp(0.05) = 1.05127
        # clipped_r1 = r1 (since 0.8 < 1.10517 < 1.2)
        # clipped_r2 = r2 (since 0.8 < 1.05127 < 1.2)
        # term1 = r1 * adv1 = 1.10517 * 1.0 = 1.10517
        # term2 = r2 * adv2 = 1.05127 * 2.0 = 2.10254
        # loss = -torch.mean(torch.tensor([1.10517, 2.10254])) = -(1.10517 + 2.10254) / 2 = -3.20771 / 2 = -1.603855
        expected_loss = manual_loss_calculation(old_log_probs, new_log_probs, advantages, self.EPSILON)
        
        calculated_loss = agent.calculate_loss(old_log_probs, new_log_probs, advantages)
        torch.testing.assert_close(calculated_loss, expected_loss)

    def test_loss_calculation_with_clipping_positive_advantage(self, agent):
        """
        Test loss calculation where ratio is clipped by 1 + epsilon (positive advantage).
        """
        # old_lp = -0.5, new_lp = -0.1. log_r = 0.4. r = exp(0.4) ~ 1.4918
        # 1 + epsilon = 1.2. So, r will be clipped to 1.2.
        # Advantage is positive (1.0).
        # unclipped_term = r * adv = 1.4918 * 1.0 = 1.4918
        # clipped_term = clipped_r * adv = 1.2 * 1.0 = 1.2
        # min(unclipped_term, clipped_term) = 1.2
        # loss = -mean(1.2) = -1.2
        old_log_probs = torch.tensor([-0.5])
        new_log_probs = torch.tensor([-0.1]) 
        advantages = torch.tensor([1.0])
        
        expected_loss = torch.tensor(-1.2) # Manually derived: - (min(exp(0.4)*1, 1.2*1))
        calculated_loss = agent.calculate_loss(old_log_probs, new_log_probs, advantages)
        torch.testing.assert_close(calculated_loss, expected_loss)

    def test_loss_calculation_with_clipping_negative_advantage_ratio_gt_one_plus_eps(self, agent):
        """
        Test loss calculation where ratio > 1+epsilon (negative advantage).
        PPO rule: For negative advantages, we want to make the objective value larger (less negative)
        if the ratio is large. So, we take min(r*A, clipped_r*A).
        If A < 0, then r*A < clipped_r*A (since r > clipped_r). So min picks r*A.
        """
        # old_lp = -0.5, new_lp = -0.1. log_r = 0.4. r = exp(0.4) ~ 1.4918
        # 1 + epsilon = 1.2. clipped_r = 1.2
        # Advantage is negative (-1.0).
        # unclipped_term = r * adv = 1.4918 * -1.0 = -1.4918
        # clipped_term = clipped_r * adv = 1.2 * -1.0 = -1.2
        # min(unclipped_term, clipped_term) = min(-1.4918, -1.2) = -1.4918
        # loss = -mean(-1.4918) = 1.4918
        old_log_probs = torch.tensor([-0.5])
        new_log_probs = torch.tensor([-0.1])
        advantages = torch.tensor([-1.0])
        
        expected_loss = -(torch.exp(torch.tensor(0.4)) * -1.0) # Corrected: Remove redundant torch.tensor()
        calculated_loss = agent.calculate_loss(old_log_probs, new_log_probs, advantages)
        torch.testing.assert_close(calculated_loss, expected_loss)

    def test_loss_calculation_with_clipping_positive_advantage_ratio_lt_one_minus_eps(self, agent):
        """
        Test loss calculation where ratio is clipped by 1 - epsilon (positive advantage).
        """
        # old_lp = -0.1, new_lp = -0.5. log_r = -0.4. r = exp(-0.4) ~ 0.6703
        # 1 - epsilon = 0.8. So, r will be clipped to 0.8.
        # Advantage is positive (1.0).
        # unclipped_term = r * adv = 0.6703 * 1.0 = 0.6703
        # clipped_term = clipped_r * adv = 0.8 * 1.0 = 0.8
        # min(unclipped_term, clipped_term) = 0.6703
        # loss = -mean(0.6703) = -0.6703
        old_log_probs = torch.tensor([-0.1])
        new_log_probs = torch.tensor([-0.5])
        advantages = torch.tensor([1.0])

        expected_loss = -(torch.exp(torch.tensor(-0.4)) * 1.0) # Corrected: Remove redundant torch.tensor()
        calculated_loss = agent.calculate_loss(old_log_probs, new_log_probs, advantages)
        torch.testing.assert_close(calculated_loss, expected_loss)
        
    def test_loss_calculation_with_clipping_negative_advantage_ratio_lt_one_minus_eps(self, agent):
        """
        Test loss calculation where ratio < 1-epsilon (negative advantage).
        PPO rule: For negative advantages, we want to make the objective value larger (less negative)
        if the ratio is small. So, we take min(r*A, clipped_r*A).
        If A < 0, then r*A > clipped_r*A (since r < clipped_r). So min picks clipped_r*A.
        """
        # old_lp = -0.1, new_lp = -0.5. log_r = -0.4. r = exp(-0.4) ~ 0.6703
        # 1 - epsilon = 0.8. clipped_r = 0.8
        # Advantage is negative (-1.0).
        # unclipped_term = r * adv = 0.6703 * -1.0 = -0.6703
        # clipped_term = clipped_r * adv = 0.8 * -1.0 = -0.8
        # min(unclipped_term, clipped_term) = min(-0.6703, -0.8) = -0.8
        # loss = -mean(-0.8) = 0.8
        old_log_probs = torch.tensor([-0.1])
        new_log_probs = torch.tensor([-0.5])
        advantages = torch.tensor([-1.0])

        expected_loss = torch.tensor(-( (1-self.EPSILON) * -1.0)) # Corrected shape
        calculated_loss = agent.calculate_loss(old_log_probs, new_log_probs, advantages)
        torch.testing.assert_close(calculated_loss, expected_loss)

    def test_loss_calculation_mixed_advantages_and_clipping(self, agent):
        """
        Test with a mix of positive/negative advantages and clipping scenarios.
        """
        old_log_probs = torch.tensor([-0.5,  -0.1, -0.5,  -0.1])
        new_log_probs = torch.tensor([-0.1,  -0.5, -0.1,  -0.5]) # r ~ 1.49, 0.67, 1.49, 0.67
        advantages    = torch.tensor([ 1.0,   1.0, -1.0,  -1.0])

        # Expected terms for min(unclipped, clipped):
        # 1. r=1.49 (clip to 1.2), adv=1.0: min(1.49*1, 1.2*1) = 1.2
        # 2. r=0.67 (clip to 0.8), adv=1.0: min(0.67*1, 0.8*1) = 0.67 (no, r < 1-eps, so r*adv is taken)
        #    r=exp(-0.4)=0.6703. clipped_r=0.8. r*adv=0.6703. clipped_r*adv=0.8. min is 0.6703.
        # 3. r=1.49 (clip to 1.2), adv=-1.0: min(1.49*(-1), 1.2*(-1)) = min(-1.49, -1.2) = -1.49
        # 4. r=0.67 (clip to 0.8), adv=-1.0: min(0.67*(-1), 0.8*(-1)) = min(-0.67, -0.8) = -0.8
        
        term1 = (1 + self.EPSILON) * advantages[0] # Clipped by 1+eps
        term2 = torch.exp(new_log_probs[1] - old_log_probs[1]) * advantages[1] # Not clipped (r < 1-eps, pos adv)
        term3 = torch.exp(new_log_probs[2] - old_log_probs[2]) * advantages[2] # Not clipped (r > 1+eps, neg adv)
        term4 = (1 - self.EPSILON) * advantages[3] # Clipped by 1-eps (r < 1-eps, neg adv)
        
        expected_loss_val = -torch.mean(torch.stack([term1, term2, term3, term4]))
        
        calculated_loss = agent.calculate_loss(old_log_probs, new_log_probs, advantages)
        torch.testing.assert_close(calculated_loss, expected_loss_val, rtol=1e-4, atol=1e-6)


    def test_input_validation(self, agent):
        """Test that calculate_loss raises errors for invalid inputs."""
        # Incorrect types
        with pytest.raises(TypeError, match="old_log_probs must be a torch.Tensor"):
            agent.calculate_loss([1.0], torch.tensor([1.0]), torch.tensor([1.0]))
        with pytest.raises(TypeError, match="new_log_probs must be a torch.Tensor"):
            agent.calculate_loss(torch.tensor([1.0]), [1.0], torch.tensor([1.0]))
        with pytest.raises(TypeError, match="advantages must be a torch.Tensor"):
            agent.calculate_loss(torch.tensor([1.0]), torch.tensor([1.0]), [1.0])

        # Shape mismatches
        with pytest.raises(ValueError, match="Shapes of old_log_probs, new_log_probs, and advantages must match."):
            agent.calculate_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0]), torch.tensor([1.0]))
        
        # Incorrect dimensions (must be 1D)
        with pytest.raises(ValueError, match="Input tensors .* must be 1-dimensional"):
            agent.calculate_loss(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[1.0]]))

    def test_epsilon_effect(self):
        """Test that changing epsilon has the expected effect on clipping."""
        epsilon_small = 0.1
        epsilon_large = 0.5
        agent_small_eps = ClippedRPG(epsilon=epsilon_small)
        agent_large_eps = ClippedRPG(epsilon=epsilon_large)

        old_log_probs = torch.tensor([-0.5])
        new_log_probs = torch.tensor([-0.1]) # r = exp(0.4) ~ 1.4918
        advantages = torch.tensor([1.0])

        # For small epsilon (0.1), 1+eps = 1.1. r > 1.1, so clipped_r = 1.1. Loss term = 1.1
        loss_small_eps = agent_small_eps.calculate_loss(old_log_probs, new_log_probs, advantages)
        expected_loss_small_eps = -torch.tensor( (1 + epsilon_small) * 1.0 ) # This one is fine (float to tensor)
        torch.testing.assert_close(loss_small_eps, expected_loss_small_eps)
        
        # For large epsilon (0.5), 1+eps = 1.5. r < 1.5, so not clipped by 1+eps. Loss term = r * adv
        loss_large_eps = agent_large_eps.calculate_loss(old_log_probs, new_log_probs, advantages)
        expected_loss_large_eps = -(torch.exp(torch.tensor(0.4)) * 1.0) # Corrected: Remove redundant torch.tensor()
        torch.testing.assert_close(loss_large_eps, expected_loss_large_eps)

        # Ensure they are different
        assert not torch.isclose(loss_small_eps, loss_large_eps)

if __name__ == "__main__":
    pytest.main()
# Example to run tests:
# Ensure you are in the root directory of the project.
# PYTHONPATH=. pytest tests/osiris_rl/algorithms/test_clipped_rpg.py
# or if osiris_rl is installed/discoverable
# pytest tests/osiris_rl/algorithms/test_clipped_rpg.py
