from dataclasses import dataclass
from typing import Optional

import torch


def rand_topk_idx(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    generator: torch.Generator,
    device: torch.device,
    *,
    restrict_to_dp_group: bool = False,
    restrict_to_local_experts: bool = False,
    dp_rank: int | None = None,
    dp_size: int | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    scores = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    scores = scores.abs() + 1

    if restrict_to_dp_group or restrict_to_local_experts:
        assert dp_rank is not None
        assert dp_size is not None
        if restrict_to_local_experts:
            mask = torch.ones(num_experts, dtype=torch.bool, device=device)
            mask[dp_rank::dp_size] = False
            scores[:, mask] = float("-inf")
        else:
            assert world_size is not None
            num_local_experts = num_experts // world_size
            dp_position = num_local_experts * dp_size * dp_rank
            scores[:, :dp_position] = float("-inf")
            scores[:, dp_position + num_local_experts * dp_size :] = float("-inf")
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    return topk_idx.to(torch.uint32)


@dataclass
class RankTestData:
    indices: torch.Tensor
    weights: torch.Tensor
    dp_x: torch.Tensor
    dp_x_scale: Optional[torch.Tensor]
    bound_m: Optional[torch.Tensor]
    expected_num_tokens: torch.Tensor

    @classmethod
    def rand_indices_and_count(
        cls,
        num_experts: int,
        num_experts_per_token: int,
        max_num_tokens: int,
        generator: torch.Generator,
        device: torch.device,
        *,
        restrict_to_dp_group: bool = False,
        restrict_to_local_experts: bool = False,
        dp_rank: int | None = None,
        dp_size: int | None = None,
        world_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = rand_topk_idx(
            max_num_tokens,
            num_experts,
            num_experts_per_token,
            generator,
            device,
            restrict_to_dp_group=restrict_to_dp_group,
            restrict_to_local_experts=restrict_to_local_experts,
            dp_rank=dp_rank,
            dp_size=dp_size,
            world_size=world_size,
        )
        expected_num_tokens = torch.bincount(
            indices.flatten().long(),
            minlength=num_experts,
        ).to(torch.int32)

#        print("indices="+str(indices)+"  expected_num_tokens="+str(expected_num_tokens))

        return indices, expected_num_tokens

    @classmethod
    def create(
        cls,
        *,
        dp_rank: int,
        dp_size: int,
        world_size: int,
        num_experts: int,
        num_experts_per_token: int,
        max_num_tokens: int,
        hidden_dim: int,
        hidden_dim_scale: Optional[int],
        in_dtype: torch.dtype,
        scale_dtype: Optional[torch.dtype],
        generator: torch.Generator,
        device: torch.device,
        restrict_to_dp_group: bool = False,
        restrict_to_local_experts: bool = False,
    ) -> "RankTestData":
        assert num_experts_per_token <= num_experts

        indices, expected_num_tokens = cls.rand_indices_and_count(
            num_experts,
            num_experts_per_token,
            max_num_tokens,
            generator,
            device,
            restrict_to_dp_group=restrict_to_dp_group,
            restrict_to_local_experts=restrict_to_local_experts,
            dp_rank=dp_rank,
            dp_size=dp_size,
            world_size=world_size,
        )
        dp_x = torch.randn(
            (max_num_tokens, hidden_dim),
            device=device,
            generator=generator,
        ).to(in_dtype)

        dp_x_scale: Optional[torch.Tensor]
        if hidden_dim_scale is not None or scale_dtype is not None:
            assert hidden_dim_scale is not None
            assert scale_dtype is not None
            dp_x_scale = torch.randn(
                (max_num_tokens, hidden_dim_scale),
                device=device,
                generator=generator,
            ).to(scale_dtype)
        else:
            dp_x_scale = None

        weights = torch.rand(
            (max_num_tokens, num_experts_per_token),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        weights = weights / torch.sum(weights, dim=-1, keepdim=True) # each token gives uniform weight to its experts (for the calculation test)

        return cls(
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            expected_num_tokens=expected_num_tokens,
            bound_m=None,
        )
