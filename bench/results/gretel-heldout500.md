Dataset: `gretelai/gretel-pii-masking-en-v1` test split, 500 documents (seed 0, offset 500; median 181 characters, max 1246). Threshold 0.5. Hardware: NVIDIA GeForce RTX 5080 (CUDA 12.8, torch 2.11.0+cu128).

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` |
|---|---:|---|
| person | 235 | 0.601 / 0.872 / 0.712 |
| email | 113 | 0.958 / 1.000 / 0.978 |
| phone number | 101 | 0.880 / 0.941 / 0.909 |
| ssn | 146 | 0.482 / 0.849 / 0.615 |
| credit card number | 63 | 0.611 / 0.921 / 0.734 |
| date of birth | 226 | 0.690 / 0.947 / 0.799 |
| ip address | 109 | 0.877 / 0.917 / 0.897 |
| **micro average** | 993 | **0.677 / 0.915 / 0.778** |
| micro average, overlap match | | 0.697 / 0.943 / 0.801 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.6 ms | 13.4 ms | 16.8 ms | 21.5 ms | 1.00 |
