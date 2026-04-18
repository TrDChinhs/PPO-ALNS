# PPO-ALNS for VRPTW — Project Specification

> **Đây là file SPEC chính.** Đọc file này TRƯỚC KHI viết bất kỳ code nào.
> Cập nhật ngay khi có thay đổi thiết kế hoặc quyết định mới.

---

## 1. Mục tiêu

Implement hybrid PPO-ALNS framework để giải **Vehicle Routing Problem with Time Windows (VRPTW)**.
So sánh PPO-ALNS (CW + Greedy init) với ALNS baseline (CW + Greedy + D-only + R-only).

---

## 2. Cấu trúc thư mục

```
d:/PPO-ALNS/
├── SPEC.md                      ← FILE CHÍNH, đọc TRƯỚC KHI code
├── ppo-alns-research-notes.md    ← Ghi chú đọc paper
│
├── data/                        # Dữ liệu đầu vào
│   ├── README.md                 ← Hướng dẫn folder data
│   ├── n20/                     # 20 customers × S/M/L × N instances
│   ├── n50/
│   └── n100/
│
├── results/                     # Kết quả output
│   ├── README.md                 ← Hướng dẫn folder results
│   └── *.json                    # Kết quả từng experiment
│
├── src/                        # Source code
│   ├── README.md                 ← Hướng dẫn folder src
│   │
│   │   # ==== PROBLEM ==== (đọc trước)
│   │   ├── problem.py            # VRPTW data structures, validation
│   │
│   │   # ==== ALNS CORE ====
│   │   ├── operators.py          # 5 destroy + 3 repair operators
│   │   ├── alns.py               # Base ALNS (roulette wheel, SA acceptance)
│   │   ├── solution.py           # Solution representation & utilities
│   │
│   │   # ==== PPO + RL ====
│   │   ├── ppo_agent.py         # PPO agent & PyTorch MLP network
│   │   ├── env.py               # MDP environment wrapping ALNS
│   │   ├── state_encoder.py     # State representation (10 features)
│   │   ├── trainer.py            # Training loop, parallel envs
│   │
│   │   # ==== EVALUATION ====
│   │   ├── evaluator.py         # Evaluate & compare methods
│   │   └── comparator.py        # Statistical comparison tools
│   │
│   └── utils.py                 # Shared utilities
│
├── generate_data.py             # Script generate random instances
├── run.py                       # Main entry point
└── requirements.txt
```

---

## 3. Problem Definition — VRPTW

### 3.1 Input

- **1 Depot**: tọa độ (x, y), thời gian mở cửa [0, T_max]
- **N Customers**: tọa độ (x, y), demand (1-16), time window [a, b], service time
- **K Vehicles**: capacity = 64, số lượng đủ nhiều (≥ N)
- **Travel**: Euclidean distance / 50, normalized

### 3.2 Constraints

1. **Capacity**: tổng demand trên mỗi route ≤ vehicle_capacity
2. **Time Windows**: xe phải đến customer trong [a, b]
   - Đến sớm → chờ (wait OK)
   - Đến muộn → TW violation (penalized)
3. **Route**: mỗi customer được phục vụ đúng 1 lần, quay về depot

### 3.3 Objective (3-stage weighted sum — Option B)

```
cost(x) = distance(x) + λ1 * tw_violations(x) + λ2 * num_vehicles(x)
```

Priority đúng: **time windows → distance → num_vehicles**

Các tham số λ1, λ2 sẽ được chọn sao cho:
- λ1 >> distance_scale (TW violation rất đắt)
- λ2 >> distance_scale (fewer vehicles rất quan trọng)

### 3.4 Instance Generation

| Parameter | Value |
|-----------|-------|
| Customer coords | Uniform [0, 1] × [0, 1] |
| Depot | Geometric center of all customers |
| Vehicle capacity | 64 |
| Demands | Uniform [1, 16] |
| Service times | Uniform [0.05, 0.1] |
| Max travel time | 1.0 |
| Distance normalization | Euclidean / 50 |

### 3.5 Time Window Difficulty Levels

| Level | Name | TW Width | Approx % of T_max |
|-------|------|----------|---------------------|
| S | Small/Tight | [0.06, 0.16] | ~10% |
| M | Medium | [0.10, 0.30] | ~20% |
| L | Large/Loose | [0.30, 0.50] | ~40% |

Số instances: **mỗi group (n × difficulty) tạo 10 instances** → tổng 9 groups × 10 = 90 instances.

### 3.6 Data Format (JSON)

```jsonc
{
  "name": "n20_M_1",            // n{n_customers}_{difficulty}_{index}
  "n_customers": 20,
  "difficulty": "M",
  "depot": {"x": 0.5, "y": 0.5},
  "customers": [
    {
      "id": 1,
      "x": 0.234,
      "y": 0.567,
      "demand": 8,
      "time_window": [0.15, 0.42],   // [earliest, latest]
      "service_time": 0.073
    }
    // ... 19 more
  ],
  "vehicle_capacity": 64,
  "max_travel_time": 1.0
}
```

---

## 4. Solution Representation

### 4.1 Route-Based Encoding

```
solution = [route1, route2, route3, ...]
route = [depot_id=0, customer_id_1, customer_id_2, ..., depot_id=0]
```

Ví dụ: `[[0, 3, 7, 0], [0, 1, 5, 2, 0], [0, 4, 6, 8, 0]]`

### 4.2 Feasibility Check

Hàm `is_feasible(solution)` kiểm tra:
- Mỗi customer xuất hiện đúng 1 lần
- Tổng demand mỗi route ≤ capacity
- Đến mỗi customer trong time window (return: violations count)

### 4.3 Cost Calculation

```
total_distance = sum(Euclidean distance between consecutive nodes / 50)
tw_violations = sum(max(0, arrival_time - latest_allowed) for each customer)
cost = total_distance + λ1 * tw_violations + λ2 * num_vehicles
```

---

## 5. ALNS Framework

### 5.1 Destroy Operators (5 operators)

| ID | Name | Description |
|----|------|-------------|
| D1 | Random Destroy | Xóa ngẫu nhiên d customers |
| D2 | String Destroy | Xóa đoạn route liên tục chứa random anchor |
| D3 | Route Destroy | Xóa cả route (short routes có xác suất cao hơn) |
| D4 | Worst-Cost Destroy | Xóa node có cost impact cao nhất (có randomization) |
| D5 | Sequence Destroy | Xóa contiguous subsequence từ concatenated routes |

**Destroy scale `d`**: số lượng customer bị xóa. Có thể fix hoặc sample ngẫu nhiên trong [1, max(2, n//10)].

### 5.2 Repair Operators (3 operators)

| ID | Name | Description |
|----|------|-------------|
| R1 | Greedy Repair | Chèn vào vị trí minimize cost increase |
| R2 | Criticality-Based Repair | Ưu tiên demand lớn, TW tight, gần depot |
| R3 | Regret-k Repair | Chọn insertion maximize (2nd_best - best) |

### 5.3 Operator Selection

#### ALNS Baseline (roulette wheel)
- 4 quality levels: global_best(25), local_best(5), accepted(1), rejected(0)
- Weight update: exponential moving average với RW = 0.9
- Selection probability: proportional to normalized weight

#### PPO-Guided
- Agent chọn operator dựa trên state (xem Section 7)

### 5.4 Acceptance Criterion

Simulated Annealing:
```
P(accept) = min(1, exp(-delta_cost / temperature))
temperature = temperature_0 * (cooling_rate ^ iteration)
```

### 5.5 Termination

- Fixed max iterations (T_max = 100 trong training, configurable)
- PPO-guided early termination (A4 action)

---

## 6. PPO Algorithm

### 6.1 State Space (10 features — normalized [0,1])

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 1 | search_progress | t / T_max | [0, 1] |
| 2 | solution_delta | (c_prev - c_curr) / c_prev | [-1, 1] |
| 3 | init_cost | c_init / baseline | [0, 1] |
| 4 | best_cost | c_best / baseline | [0, 1] |
| 5 | destroy_usage | frequency of each D op | [0, 1] × 5 |
| 6 | repair_usage | frequency of each R op | [0, 1] × 3 |
| 7 | demand | customer demand data | (n, ) |
| 8 | time_windows | TW [a, b] per customer | (n, 2) |
| 9 | service_times | per customer | (n, ) |
| 10 | travel_times | pairwise Euclidean / 50 | (n, n) |

**Total state dim**: 1 + 1 + 1 + 1 + 5 + 3 + n + 2n + n + n² = n² + 4n + 12
- n=20: 492 dims
- n=50: 2712 dims
- n=100: 10412 dims

Với MLP, có thể flatten toàn bộ hoặc dùng attention. Flatten + MLP là baseline.

### 6.2 Action Space (4-dimensional)

| Action | # Choices | Description |
|--------|-----------|-------------|
| A1 (destroy) | 5 | Chọn 1 trong 5 destroy operators |
| A2 (repair) | 3 | Chọn 1 trong 3 repair operators |
| A3 (acceptance) | 2 | 0=reject, 1=accept |
| A4 (termination) | 2 | 0=continue, 1=stop |

### 6.3 Reward Function

#### Immediate Reward (per step)
```
if A3=0 and c(x_t) < c(x_{t-1}):    # Reject but better
    R_t = α * (c_prev - c_curr) / c_prev
elif A3=1 and c(x_t) > c(x_{t-1}):  # Accept but worse
    R_t = β * (c_prev - c_curr) / c_prev  # negative
elif c(x_t) < c(x_best):             # New best found
    R_t = γ * (c_best - c(x_t)) / c_best
else:
    R_t = 0
```

#### Terminal Reward (end of episode)
```
R_final = f1 * (c_best - c_init) / c_init + f2 * (1 - t / T_max)
```

**Chưa xác định**: exact values của α, β, γ, f1, f2.
→ Sẽ thử nghiệm, bắt đầu với α=1, β=0.5, γ=2, f1=10, f2=5.

### 6.4 Network Architecture

```
Shared MLP:
  Input: state_dim (variable per problem size)
  Layer1: Linear → 512 → ReLU
  Layer2: Linear → 256 → ReLU
  Layer3: Linear → 128 → ReLU

Policy Head:
  Linear(128, 5)  → A1: destroy (Categorical)
  Linear(128, 3)  → A2: repair (Categorical)
  Linear(128, 2)  → A3: accept (Bernoulli/ Categorical)
  Linear(128, 2)  → A4: terminate (Bernoulli/ Categorical)

Value Head:
  Linear(128, 1) → V(s)
```

### 6.5 PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total training steps | 200,000 - 400,000 |
| Parallel environments | 16-32 |
| Learning rate | 0.001 (Adam, có thể decay) |
| Batch size | 128 |
| PPO epochs per update | 4-10 |
| Clip epsilon (ε) | 0.2 |
| GAE lambda (λ) | 0.95 |
| Discount factor (γ) | 0.99 |
| Max episode length | 100 iterations |
| Entropy coefficient | 0.01 (for exploration) |

### 6.6 Training Loop

1. Khởi tạo parallel envs (mỗi env = 1 VRPTW instance)
2. For each step:
   - Agent chọn actions (A1, A2, A3, A4) từ policy
   - Apply vào ALNS, nhận reward + next state
   - Store (s, a, r, s', done) vào replay buffer
3. Khi buffer đầy:
   - Compute GAE advantages
   - PPO update (clip objective, entropy bonus)
   - Reset buffer
4. Sau training → evaluate on test set

---

## 7. Methods to Implement

| Method | Init | Operator Selection | Acceptance | Termination |
|--------|------|-------------------|------------|-------------|
| ALNS(Greedy) | Greedy | Roulette wheel | SA | Fixed T_max |
| ALNS(CW) | Clarke-Wright | Roulette wheel | SA | Fixed T_max |
| ALNS-D(Greedy) | Greedy | PPO for D only, roulette for R | SA | Fixed T_max |
| ALNS-R(Greedy) | Greedy | Roulette for D, PPO for R only | SA | Fixed T_max |
| PPO-ALNS(Greedy) | Greedy | PPO (A1, A2) | PPO (A3) | PPO (A4) |
| PPO-ALNS(CW) | Clarke-Wright | PPO (A1, A2) | PPO (A3) | PPO (A4) |

---

## 8. Output Format

### 8.1 Per-Instance Result JSON

```jsonc
{
  "instance": "n20_M_1",
  "method": "PPO_ALNS_CW",
  "config": {
    "n_customers": 20,
    "difficulty": "M",
    "training_steps": 300000,
    "batch_size": 128,
    "seed": 42
  },
  "results": {
    "best_cost": 6.01,
    "distance": 285.3,
    "num_vehicles": 5,
    "tw_violations": 0,
    "runtime_seconds": 14.5,
    "iterations": 87,
    "best_found_at_iter": 45
  },
  "routes": [
    [0, 3, 7, 0],
    [0, 1, 5, 2, 0],
    [0, 4, 6, 8, 0]
  ],
  "training_curves": {
    "steps": [0, 1000, 2000, ...],
    "avg_cost": [7.5, 6.8, 6.3, ...]
  }
}
```

### 8.2 Summary Comparison JSON

```jsonc
{
  "summary": true,
  "date": "2026-04-17",
  "instances": ["n20_S", "n20_M", "n20_L", "n50_S", ...],
  "methods": ["ALNS_Greedy", "ALNS_CW", "PPO_ALNS_Greedy", "PPO_ALNS_CW"],
  "comparison": {
    "n20_M": {
      "ALNS_Greedy": {"avg_cost": 6.78, "win_rate": 0.0},
      "PPO_ALNS_CW": {"avg_cost": 6.01, "win_rate": 1.0}
    }
  }
}
```

---

## 9. Implementation Priorities

### Phase 1: Baseline (đọc trước)
1. `problem.py` — data structures
2. `solution.py` — route representation, cost calc, feasibility
3. `operators.py` — 5 destroy + 3 repair operators
4. `alns.py` — base ALNS với roulette wheel + SA

### Phase 2: PPO Core
5. `state_encoder.py` — 10-feature state extraction
6. `ppo_agent.py` — PPO agent + MLP network
7. `env.py` — MDP environment wrapping ALNS
8. `trainer.py` — training loop với parallel envs

### Phase 3: Evaluation & Data
9. `generate_data.py` — generate random instances
10. `evaluator.py` — evaluate all methods
11. `comparator.py` — statistical comparison
12. `run.py` — main entry point

---

## 10. Open Questions / TODO

- [ ] Exact reward parameters (α, β, γ, f1, f2) — thử nghiệm
- [ ] λ1, λ2 cho objective function — thử nghiệm
- [ ] PPO hyperparameters fine-tune
- [ ] Whether to use shared or separate networks for 4 action heads
- [ ] State normalization strategy (layer norm vs manual)
- [ ] Visualize routes (phase sau)

---

## 11. Dependencies

```
torch
stable-baselines3  # tham khảo, có thể dùng hoặc tự implement PPO
numpy
pandas
matplotlib  # cho visualize
tqdm  # progress bar
```

---

## 12. Ghi chú quan trọng

- **Đọc SPEC.md TRƯỚC KHI code.** Mỗi file trong `src/` đều có ghi chú đầu file mô tả ý nghĩa và cách dùng.
- **Mỗi folder có README.md** giải thích mục đích folder.
- **Code theo Python rules** trong `.claude/rules/`:
  - Type annotations everywhere
  - Immutable patterns (tạo new objects, không mutate)
  - Small files (< 800 lines)
  - Functions < 50 lines
  - Proper error handling
  - docstrings cho public functions
