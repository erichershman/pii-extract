Dataset: `gretelai/gretel-pii-masking-en-v1` test split, 500 documents (seed 0; median 188 characters, max 843). Threshold 0.5. Hardware: NVIDIA GeForce RTX 5080 (CUDA 12.8, torch 2.11.0+cu128).

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.611 / 0.922 / 0.735 | 0.558 / 0.943 / 0.701 |
| email | 97 | 0.941 / 0.990 / 0.965 | 0.949 / 0.969 / 0.959 |
| phone number | 75 | 0.863 / 0.920 / 0.890 | 0.931 / 0.893 / 0.912 |
| ssn | 161 | 0.974 / 0.919 / 0.946 | 0.986 / 0.876 / 0.928 |
| credit card number | 64 | 1.000 / 0.562 / 0.720 | 1.000 / 0.578 / 0.733 |
| date of birth | 240 | 0.704 / 0.950 / 0.809 | 0.836 / 0.871 / 0.853 |
| ip address | 97 | 0.936 / 0.907 / 0.921 | 0.970 / 0.670 / 0.793 |
| **micro average** | 964 | **0.773 / 0.910 / 0.836** | **0.785 / 0.861 / 0.821** |
| micro average, overlap match | | 0.786 / 0.925 / 0.850 | 0.793 / 0.871 / 0.830 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.6 ms | 13.4 ms | 16.6 ms | 20.6 ms | 1.00 |
| `urchade/gliner_multi_pii-v1` | cuda | 15.8 ms | 15.4 ms | 20.1 ms | 26.3 ms | 1.00 |
