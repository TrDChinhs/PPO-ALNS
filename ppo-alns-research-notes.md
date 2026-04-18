# PPO-ALNS for VRPTW — Research & Implementation Notes

## Paper Summary

**Title:** Reinforcement learning-guided adaptive large neighborhood search for vehicle routing problem with time windows
**Authors:** Wang et al. (2025), Journal of Combinatorial Optimization
**DOI:** https://doi.org/10.1007/s10878-025-01364-6
**Code:** https://github.com/Kikujjy/ppo-alns

## Problem: VRPTW (Vehicle Routing Problem with Time Windows)

- Mỗi xe xuất phát từ depot, phải phục vụ các customer trong time windows cho phép
- Ràng buộc: vehicle capacity, time windows
- Mục tiêu: minimize total cost (distance)

## Core Idea: PPO-ALNS

Biến ALNS thành MDP, dùng PPO agent hướng dẫn 4 quyết định tại mỗi iteration:

1. **Destroy operator selection** (A1) — chọn 1 trong 5 operators
2. **Repair operator selection** (A2) — chọn 1 trong 3 operators
3. **Solution acceptance** (A3) — accept/reject new solution
4. **Search termination** (A4) — continue/stop early

## State Space (Appendix A)

| Feature | Description | Range |
|---------|-------------|-------|
| `search_progress` | Relative progress toward max iterations | [0, 1] |
| `solution_delta` | Improvement coefficient from previous iteration | [-1, 1] |
| `init_cost` | Initial solution objective value | [0, 1] |
| `best_cost` | Best solution found so far | [0, 1] |
| `destroy_usage` | Frequency of each destroy operator | [0, 1] |
| `repair_usage` | Frequency of each repair operator | [0, 1] |
| `demand` | Customer demand data | — |
| `time_windows` | Time window data per customer | — |
| `service_times` | Service time per customer | — |
| `travel_times` | Pairwise travel time between customers | — |

## Action Space (4-dimensional)

### A1 — Destroy Operators (5 operators)
1. **Random Destroy**: Loại bỏ ngẫu nhiên 1 tập customer
2. **String Destroy**: Loại bỏ 1 đoạn route liên tục chứa anchor customer
3. **Route Destroy**: Loại bỏ cả route dựa trên inverse-length probability
4. **Worst-Cost Destroy**: Ưu tiên loại bỏ node có impact cao (có randomization)
5. **Sequence Destroy**: Loại bỏ contiguous subsequence từ concatenated route

### A2 — Repair Operators (3 operators)
1. **Greedy Repair**: Chèn customer vào vị trí tốt nhất, minimize cost increase
2. **Criticality-Based Repair**: Ưu tiên customer theo composite metric (demand, time window tightness, depot proximity)
3. **Regret-k Repair**: Chọn insertion maximize cost difference giữa best và kế tiếp (dynamic k)

### A3 — Acceptance (2 actions)
- 0: Reject new solution
- 1: Accept new solution

### A4 — Termination (2 actions)
- 0: Continue search
- 1: Terminate early

## Reward Function

### Immediate Reward (per step)
```
Rt = {
  α * (c(x_{t-1}) - c(x_t)) / c(x_{t-1}),    if A3=0 and c(x_t) < c(x_{t-1})
  β * (c(x_{t-1}) - c(x_t)) / c(x_{t-1}),    if A3=1 and c(x_t) > c(x_{t-1})
  γ * (c(x_best) - c(x_t)) / c(x_best),       if c(x_t) < c(x_best)
}
```
- α, β, γ: positive constants controlling reward magnitude

### Terminal Reward
```
R_final = f1 * (c(x_best) - c(x_init)) / c(x_init) + f2 * (1 - t / T_max)
```
- f1, f2: scaling factors
- t: iterations taken, T_max: max iterations
- Encourages both quality improvement AND efficiency

### Objective Function Value (c(x))
```
c(x) = total_distance + λ * TW_violation_penalty
```
- TW penalty for time window violations
- Paper states PPO-ALNS maintains **strict constraint adherence** (no soft constraints)

## PPO Algorithm

- Clipped surrogate objective (equation 5)
- GAE for advantage estimation (equation 7)
- Discount factor γ = 0.99
- Network: MLP [512, 256, 128] with ReLU, shared feature extractor for policy and value

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Parallel environments | 32 |
| Total training steps | 1,600,000 |
| Learning rate | 0.001 (Adam) |
| Batch size | 128 |
| Discount factor γ | 0.99 |
| Max iterations | 100 |
| GPU | RTX 3090 |
| CPU | Intel Xeon E5-2678 v3 |

Training time: ~1h (n=20), ~1.5h (n=50), ~3.5h (n=100)

## Baseline ALNS Parameters

- Roulette wheel scores: [25, 5, 1, 0] for 4 quality levels
- Decay coefficient: 0.9 for score adjustment
- SA-based acceptance with geometric cooling

## Initial Solutions

1. **Greedy Initialization**: Iteratively insert nearest feasible customer
2. **Clarke-Wright (CW) Savings**: Route consolidation maximizing savings value

## Results (Table 1)

| Method | n=20 | n=50 | n=100 |
|--------|------|------|-------|
| ALNS(Greedy) | 6.78 | 13.73 | 24.76 |
| ALNS(CW) | 6.33 | 12.48 | 21.84 |
| ACO | 6.39 | 13.48 | 25.30 |
| GRLOS(CW) | 6.11 | 12.15 | 21.65 |
| Ours(Greedy) | 6.14 | 12.53 | 22.00 |
| Ours(CW) | **6.01** | **11.92** | **20.45** |

**Improvement over ALNS(Greedy):** 11.37%, 13.15%, 17.41%

## Ablation Study (Table 2)

- Ours-D(CW): PPO guides only destroy → 3.55%, 3.33%, 5.93%
- Ours-R(CW): PPO guides only repair → 3.08%, 3.05%, 4.79%
- Ours(CW): PPO guides both → 5.13%, 4.45%, 6.34%
- Synergistic effect: combined > sum of individual

## Time Window Sensitivity (Table 3)

- S-window (tight): 2.15-3.78% improvement
- M-window (medium): **4.45-6.34%** — best performance
- L-window (loose): 2.78-4.10% improvement
- PPO-ALNS works best in moderate complexity scenarios

## Key Implementation Details from Paper

### Problem Instance Generation
- Customer coords: uniform [0,1]×[0,1]
- Depot: geometric center of customers
- Vehicle capacity: 64
- Demands: [1, 16]
- Service times: [0.05, 0.1]
- Time windows: [0.1, 0.3]
- Max travel time: 1
- Distances: Euclidean / 50

### Architecture
- Shared MLP feature extractor: [512, 256, 128] ReLU
- Policy head: linear → action dimension
- Value head: linear → scalar
- Same architecture for all problem sizes

### ALNS Core Loop
1. Chọn destroy operator (theo weight hoặc PPO)
2. Apply destroy (remove d customers)
3. Chọn repair operator
4. Apply repair (reinsert customers)
5. SA acceptance check
6. Update weights if using ALNS baseline
7. Check termination

### Destroy Scale (d)
- Fixed hyperparameter hoặc dynamically sampled
- Controls magnitude of perturbation

## What's NOT in Paper (Need to Infer)

- Exact values of α, β, γ, f1, f2 reward parameters
- Exact PPO hyperparameters (epsilon for clipping, GAE lambda, etc.)
- How state features are normalized
- Exact destroy scale d value
- How distance matrix is normalized
- CUDA/PyTorch specifics for training
- Exact random seed strategy
- How the 4 actions are sampled from policy (categorical for discrete, or separate heads)
- Whether actions are autoregressive or independent
- Batch normalization / layer norm usage
- How time window feasibility is checked during repair

## Reference Algorithms to Implement for Comparison

1. **ALNS baseline** — roulette wheel selection, SA acceptance
2. **ACO** — for comparison (pheromone-based)
3. Optional: **GRLOS** if time permits

## Questions for User

See conversation — to be filled in based on user discussion.
