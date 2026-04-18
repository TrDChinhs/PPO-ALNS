# Results Folder

## Mục đích

Chứa kết quả đầu ra của các thuật toán — routes, cost, runtime, training curves.

## Cấu trúc

```
results/
├── README.md           ← File này
├── n20/                # Kết quả cho instances n20
│   ├── n20_M_1__PPO_ALNS_CW__2026-04-17.json
│   ├── n20_M_1__ALNS_Greedy__2026-04-17.json
│   └── ...
├── n50/
│   └── ...
├── n100/
│   └── ...
└── summary/            # Bảng tổng hợp so sánh
    └── comparison_2026-04-17.json
```

## Format JSON per instance

Xem `SPEC.md` Section 8.1.

## Cách lưu kết quả

Kết quả được lưu tự động bởi `src/evaluator.py` sau mỗi experiment.

## Quy ước đặt tên file

```
{instance_name}__{method}__{date}.json
```

Ví dụ: `n20_M_1__PPO_ALNS_CW__2026-04-17.json`

## Summary Comparison

File `summary/comparison_*.json` chứa bảng tổng hợp so sánh tất cả methods trên tất cả instances, bao gồm:
- Average cost per method per group
- Win rate (pairwise comparison)
- Statistical significance

## Cập nhật

- Không edit tay các file kết quả.
- Chỉ evaluator.py được phép ghi vào folder này.
- Khi format output thay đổi → cập nhật cả file này và SPEC.md Section 8.
