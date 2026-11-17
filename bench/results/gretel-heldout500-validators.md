Dataset: `gretelai/gretel-pii-masking-en-v1` test split, 500 documents (seed 0, offset 500; median 181 characters, max 1246). Threshold 0.5. Hardware: NVIDIA GeForce RTX 5080 (CUDA 12.8, torch 2.11.0+cu128).

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` |
|---|---:|---|
| person | 235 | 0.601 / 0.872 / 0.712 |
| email | 113 | 0.974 / 1.000 / 0.987 |
| phone number | 101 | 0.905 / 0.941 / 0.922 |
| ssn | 146 | 0.969 / 0.849 / 0.905 |
| credit card number | 63 | 1.000 / 0.556 / 0.714 |
| date of birth | 226 | 0.690 / 0.947 / 0.799 |
| ip address | 109 | 0.952 / 0.917 / 0.935 |
| **micro average** | 993 | **0.777 / 0.892 / 0.831** |
| micro average, overlap match | | 0.793 / 0.910 / 0.848 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.0 ms | 12.7 ms | 15.7 ms | 24.1 ms | 1.00 |
