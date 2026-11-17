> Note: the "validators on" rows in this file were produced before the `ssn` validator was widened. See `gretel-validators.md` for current validator results.

Dataset: `gretelai/gretel-pii-masking-en-v1` test split, 500 documents (seed 0; median 188 characters, max 843). Threshold 0.5. Hardware: NVIDIA GeForce RTX 5080 (CUDA 12.8, torch 2.11.0+cu128).

`ssn` and the micro average over the base 7 labels (so every row is scored on the same gold spans), precision / recall / F1:

| Model | Variant | ssn | micro average |
|---|---|---|---|
| `fastino/gliner2-privacy-filter-PII-multi` | baseline | 0.442 / 0.919 / 0.597 | 0.657 / 0.929 / 0.770 |
| `fastino/gliner2-privacy-filter-PII-multi` | validators on | 0.955 / 0.528 / 0.680 | 0.759 / 0.844 / 0.800 |
| `fastino/gliner2-privacy-filter-PII-multi` | + medical record number | 0.738 / 0.894 / 0.809 | 0.739 / 0.920 / 0.819 |
| `fastino/gliner2-privacy-filter-PII-multi` | validators on + medical record number | 0.964 / 0.503 / 0.661 | 0.769 / 0.835 / 0.801 |
| `urchade/gliner_multi_pii-v1` | baseline | 0.959 / 0.876 / 0.916 | 0.774 / 0.878 / 0.823 |
| `urchade/gliner_multi_pii-v1` | validators on | 0.976 / 0.503 / 0.664 | 0.772 / 0.799 / 0.785 |
| `urchade/gliner_multi_pii-v1` | + medical record number | 0.953 / 0.876 / 0.913 | 0.774 / 0.880 / 0.823 |
| `urchade/gliner_multi_pii-v1` | validators on + medical record number | 0.976 / 0.503 / 0.664 | 0.773 / 0.801 / 0.787 |

### baseline

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
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.1 ms | 12.8 ms | 15.7 ms | 20.5 ms | 1.00 |
| `urchade/gliner_multi_pii-v1` | cuda | 16.0 ms | 15.6 ms | 20.4 ms | 26.1 ms | 1.00 |

### validators on

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.611 / 0.922 / 0.735 | 0.558 / 0.943 / 0.701 |
| email | 97 | 0.941 / 0.990 / 0.965 | 0.949 / 0.969 / 0.959 |
| phone number | 75 | 0.863 / 0.920 / 0.890 | 0.931 / 0.893 / 0.912 |
| ssn | 161 | 0.955 / 0.528 / 0.680 | 0.976 / 0.503 / 0.664 |
| credit card number | 64 | 1.000 / 0.562 / 0.720 | 1.000 / 0.578 / 0.733 |
| date of birth | 240 | 0.704 / 0.950 / 0.809 | 0.836 / 0.871 / 0.853 |
| ip address | 97 | 0.936 / 0.907 / 0.921 | 0.970 / 0.670 / 0.793 |
| **micro average** | 964 | **0.759 / 0.844 / 0.800** | **0.772 / 0.799 / 0.785** |
| micro average, overlap match | | 0.773 / 0.860 / 0.814 | 0.780 / 0.809 / 0.794 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.6 ms | 13.3 ms | 16.0 ms | 19.6 ms | 1.00 |
| `urchade/gliner_multi_pii-v1` | cuda | 15.6 ms | 15.2 ms | 19.6 ms | 25.5 ms | 1.00 |

### + medical record number

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.628 / 0.909 / 0.742 | 0.559 / 0.943 / 0.702 |
| email | 97 | 0.923 / 0.990 / 0.955 | 0.913 / 0.969 / 0.940 |
| phone number | 75 | 0.841 / 0.920 / 0.879 | 0.919 / 0.907 / 0.913 |
| ssn | 161 | 0.738 / 0.894 / 0.809 | 0.953 / 0.876 / 0.913 |
| credit card number | 64 | 0.824 / 0.875 / 0.848 | 0.883 / 0.828 / 0.855 |
| date of birth | 240 | 0.710 / 0.938 / 0.808 | 0.833 / 0.875 / 0.854 |
| ip address | 97 | 0.863 / 0.907 / 0.884 | 0.915 / 0.670 / 0.774 |
| medical record number | 284 | 0.869 / 0.863 / 0.866 | 0.930 / 0.750 / 0.830 |
| **micro average** | 1248 | **0.763 / 0.907 / 0.829** | **0.801 / 0.850 / 0.825** |
| micro average, base 7 labels | 964 | 0.739 / 0.920 / 0.819 | 0.774 / 0.880 / 0.823 |
| micro average, overlap match | | 0.783 / 0.930 / 0.850 | 0.816 / 0.870 / 0.842 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.8 ms | 13.5 ms | 16.4 ms | 19.7 ms | 1.00 |
| `urchade/gliner_multi_pii-v1` | cuda | 15.8 ms | 15.4 ms | 20.0 ms | 24.9 ms | 1.00 |

### validators on + medical record number

Exact span match (start, end and label must all agree), precision / recall / F1:

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.628 / 0.909 / 0.742 | 0.559 / 0.943 / 0.702 |
| email | 97 | 0.941 / 0.990 / 0.965 | 0.949 / 0.969 / 0.959 |
| phone number | 75 | 0.863 / 0.920 / 0.890 | 0.932 / 0.907 / 0.919 |
| ssn | 161 | 0.964 / 0.503 / 0.661 | 0.976 / 0.503 / 0.664 |
| credit card number | 64 | 1.000 / 0.578 / 0.733 | 1.000 / 0.578 / 0.733 |
| date of birth | 240 | 0.710 / 0.938 / 0.808 | 0.833 / 0.875 / 0.854 |
| ip address | 97 | 0.936 / 0.907 / 0.921 | 0.970 / 0.670 / 0.793 |
| medical record number | 284 | 0.869 / 0.863 / 0.866 | 0.930 / 0.750 / 0.830 |
| **micro average** | 1248 | **0.790 / 0.841 / 0.815** | **0.802 / 0.789 / 0.796** |
| micro average, base 7 labels | 964 | 0.769 / 0.835 / 0.801 | 0.773 / 0.801 / 0.787 |
| micro average, overlap match | | 0.806 / 0.858 / 0.831 | 0.813 / 0.802 / 0.807 |

Latency per text through `Pipeline.extract` (one request at a time, after warm-up):

| Model | Device | Mean | p50 | p95 | Max | Windows/text |
|---|---|---:|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | cuda | 13.7 ms | 13.5 ms | 16.7 ms | 23.8 ms | 1.00 |
| `urchade/gliner_multi_pii-v1` | cuda | 15.7 ms | 15.3 ms | 19.9 ms | 25.1 ms | 1.00 |
