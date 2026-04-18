# Source Code Folder (`src/`)

## Mục đích

Chứa toàn bộ source code của PPO-ALNS framework.

## Đọc theo thứ tự (IMPORTANT)

Đọc **theo thứ tự** dưới đây để hiểu flow của code. Mỗi file đều có ghi chú đầu file.

### ==== PROBLEM LAYER ====
Đọc **TRƯỚC TIÊN** — không hiểu problem thì không hiểu gì cả.

1. **`problem.py`** — Data structures cho VRPTW instance
   - Định nghĩa `VRPTWInstance`, `Customer`, `Depot`
   - Cách load từ JSON
   - Distance calculation, feasibility check

2. **`solution.py`** — Solution representation
   - `Solution` class: list of routes
   - Cost calculation (distance + TW violations + vehicles)
   - Route feasibility validation
   - Initial solution generators (Greedy, Clarke-Wright)

### ==== ALNS CORE ====
Sau khi hiểu problem + solution.

3. **`operators.py`** — Destroy & Repair operators
   - 5 destroy operators (D1-D5)
   - 3 repair operators (R1-R3)
   - Mỗi operator có docstring mô tả thuật toán

4. **`alns.py`** — Base ALNS framework
   - Roulette wheel operator selection
   - SA acceptance criterion
   - Main ALNS loop
   - PPO-guided ALNS variant

### ==== PPO + RL ====
Sau khi hiểu ALNS core.

5. **`state_encoder.py`** — State representation
   - 10 features cho MDP state (xem SPEC.md Section 6.1)
   - Normalization utilities

6. **`ppo_agent.py`** — PPO agent
   - PyTorch MLP network (shared extractor + 4 action heads)
   - PPO update logic (clip objective, GAE, entropy)
   - Action sampling

7. **`env.py`** — MDP Environment
   - Wraps ALNS thành gym-style environment
   - Implements step(), reset(), observe()
   - Kết nối state_encoder ↔ alns ↔ ppo_agent

8. **`trainer.py`** — Training loop
   - Parallel environments (vectorized)
   - Rollout collection
   - PPO update
   - Checkpoint saving

### ==== EVALUATION ====
Sau khi có trained policy.

9. **`evaluator.py`** — Evaluate & benchmark
   - Chạy ALNS baseline
   - Chạy PPO-ALNS
   - Lưu kết quả ra JSON

10. **`comparator.py`** — Statistical comparison
    - Pairwise win rate
    - Average improvement %
    - Summary tables

11. **`utils.py`** — Shared utilities
    - Random seed management
    - Time measurement
    - Config loading

## Thứ tự implementation (theo SPEC.md Section 9)

```
Phase 1: problem.py → solution.py → operators.py → alns.py
Phase 2: state_encoder.py → ppo_agent.py → env.py → trainer.py
Phase 3: generate_data.py → evaluator.py → comparator.py → run.py
```

## Cập nhật

- Mỗi file khi thay đổi logic → cập nhật docstring đầu file.
- Khi thêm file mới → thêm vào danh sách trên và cập nhật SPEC.md.
- Ghi chú trong code: **tại sao** làm vậy, không phải **làm gì** (vì code đã tự giải thích).
