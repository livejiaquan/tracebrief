| Metric | old (0.40) | v3 (0.35) | v4-T (0.45) | v4-S (0.25) |
|---|---|---|---|---|
| C1a Stage I pair P / R | 0.981 / 0.864 (51/1/8) | 0.914 / 0.898 (53/5/6) | 0.964 / 0.898 (53/2/6) | 0.964 / 0.898 (53/2/6) |
| C1b events strict / 33 | 26 | 26 | 25 | 27 |
| event precision | 27/31 | 26/31 | 25/30 | 27/32 |
| GDINO-24 pair P / R (secondary) | 0.912 / 0.852 | 0.966 / 0.934 | 0.958 / 0.926 | 0.958 / 0.943 |
| C2a LOCO s3 forklift AP50 | 0.000 | 0.005 | 0.271 | 0.337 |
| C2a LOCO s3 forklift P / R at conf | 0.000 / 0.000 (0 TP, 30 FP) | 0.071 / 0.005 (1 TP, 13 FP) | 0.744 / 0.154 (29 TP, 10 FP) | 0.667 / 0.340 (64 TP, 32 FP) |
| C4 LOCO s3 false forklift on pallet trucks / stillages | 0.000 / 0.005 | 0.006 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| C2b keremberke forklift AP50 | 0.000 | 0.000 | 0.913 | 0.894 |
| C2b keremberke person AP50 | 0.087 | 0.033 | 0.514 | 0.511 |
| C2b keremberke pair recall (of 25) | 0.000 (0) | 0.000 (0) | 0.680 (17) | 0.680 (17) |
| keremberke forklift / person P / R at conf | 0.00/0.00, 0.36/0.09 | —/0.00, 0.05/0.07 | 0.86/0.88, 0.72/0.48 | 0.83/0.91, 0.63/0.50 |
| C3 SS forklift found (pooled) | 0.122 | 0.273 | 0.276 | 0.332 |
| C3 SS person P / R (pooled) | 0.675 / 0.254 | 0.794 / 0.564 | 0.920 / 0.539 | 0.821 / 0.565 |
| C3 SS alerts strict / covered / false | 8 / 25 / 1 | 14 / 36 / 0 | 20 / 50 / 1 | 21 / 52 / 8 |

| Criterion | v4-T | v4-S |
|---|---|---|
| C1a | pass | pass |
| C1b | pass | pass |
| C2a | FAIL | pass |
| C2b | pass | pass |
| C3 | pass | pass |
| C4 | pass | pass |
| export | pass | pass |
| all_pass | FAIL | pass |

SmartSpaces per scene (forklift found per camera, person P/R):
old {'Warehouse_010': (0.201, 0.636, 0.244), 'Warehouse_018': (0.021, 0.43, 0.321), 'Warehouse_019': (0.029, 0.823, 0.279), 'Warehouse_022': (0.0, 0.931, 0.278)}
v3 {'Warehouse_010': (0.481, 0.796, 0.588), 'Warehouse_018': (0.014, 0.699, 0.729), 'Warehouse_019': (0.001, 0.833, 0.585), 'Warehouse_022': (0.0, 0.793, 0.433)}
v4-T {'Warehouse_010': (0.365, 0.918, 0.489), 'Warehouse_018': (0.139, 0.978, 0.795), 'Warehouse_019': (0.196, 0.83, 0.673), 'Warehouse_022': (0.452, 0.946, 0.667)}
v4-S {'Warehouse_010': (0.414, 0.84, 0.512), 'Warehouse_018': (0.209, 0.946, 0.94), 'Warehouse_019': (0.257, 0.759, 0.712), 'Warehouse_022': (0.418, 0.765, 0.681)}

| Model | ONNX MB | session.run ms mean / median / p95 | preprocess+run+decode ms mean |
|---|---|---|---|
| old | 20.2 | 22.4 / 17.9 / 31.3 | 23.9 |
| v3 | 20.2 | 22.5 / 18.4 / 30.9 | 24.0 |
| v4-T | 20.2 | 27.8 / 31.0 / 33.9 | 29.4 |
| v4-S | 35.8 | 40.5 / 33.0 / 54.1 | 42.2 |
