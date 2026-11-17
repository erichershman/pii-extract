Dataset: `gretelai/gretel-pii-masking-en-v1` test split, 500 documents joined 25 at a time into 20 long texts (seed 0; median 5544 characters, max 6767). Threshold 0.5. Hardware: NVIDIA GeForce RTX 5080 (CUDA 12.8, torch 2.11.0+cu128).

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.830 / 0.891 / 0.860 | 0.772 / 0.900 / 0.831 |
| email | 97 | 0.932 / 0.990 / 0.960 | 0.931 / 0.969 / 0.949 |
| phone number | 75 | 0.910 / 0.947 / 0.928 | 0.900 / 0.840 / 0.869 |
| ssn | 161 | 0.495 / 0.919 / 0.643 | 0.966 / 0.894 / 0.929 |
| credit card number | 64 | 0.728 / 0.922 / 0.814 | 0.912 / 0.812 / 0.860 |
| date of birth | 240 | 0.799 / 0.942 / 0.864 | 0.967 / 0.733 / 0.834 |
| ip address | 97 | 0.838 / 0.907 / 0.871 | 0.956 / 0.670 / 0.788 |
| **micro average** | 964 | **0.747 / 0.926 / 0.827** | **0.895 / 0.831 / 0.862** |
| micro average, overlap match | | 0.758 / 0.941 / 0.840 | 0.909 / 0.848 / 0.878 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 85.3 ms | 72.7 ms | 179.2 ms | 189.0 ms | 5.30 |
| `urchade/gliner_multi_pii-v1` | cuda | 105.6 ms | 86.7 ms | 202.1 ms | 207.5 ms | 6.50 |
