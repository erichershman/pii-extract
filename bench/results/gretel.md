Dataset: `gretelai/gretel-pii-masking-en-v1` test split, 500 documents (seed 0; median 188 characters, max 843). Threshold 0.5. Hardware: NVIDIA GeForce RTX 5080 (CUDA 12.8, torch 2.11.0+cu128).

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.611 / 0.922 / 0.735 | 0.558 / 0.943 / 0.701 |
| email | 97 | 0.932 / 0.990 / 0.960 | 0.913 / 0.969 / 0.940 |
| phone number | 75 | 0.841 / 0.920 / 0.879 | 0.918 / 0.893 / 0.905 |
| ssn | 161 | 0.442 / 0.919 / 0.597 | 0.959 / 0.876 / 0.916 |
| credit card number | 64 | 0.775 / 0.859 / 0.815 | 0.883 / 0.828 / 0.855 |
| date of birth | 240 | 0.704 / 0.950 / 0.809 | 0.836 / 0.871 / 0.853 |
| ip address | 97 | 0.863 / 0.907 / 0.884 | 0.915 / 0.670 / 0.774 |
| **micro average** | 964 | **0.657 / 0.929 / 0.770** | **0.774 / 0.878 / 0.823** |
| micro average, overlap match | | 0.674 / 0.953 / 0.790 | 0.788 / 0.898 / 0.839 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.9 ms | 13.6 ms | 16.8 ms | 19.9 ms | 1.00 |
| `urchade/gliner_multi_pii-v1` | cuda | 16.0 ms | 15.5 ms | 20.8 ms | 26.7 ms | 1.00 |
