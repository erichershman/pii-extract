# Benchmark results

All runs: `gretelai/gretel-pii-masking-en-v1` test split, seed 0, threshold 0.5,
exact span match, RTX 5080. Precision / recall / F1.

## Headline figures: held-out sample

The `ssn` validator was widened after looking at what it rejected on the
default sample (documents 0-499 of the shuffled split), so that sample is a
tuning set for validator results. These figures are from documents 500-999,
which were not looked at before the validator was fixed
(`just bench --backends gliner2 --offset 500 [--validators on]`).

| GLiNER2-PII | ssn | Micro average (7 labels) |
|---|---|---|
| baseline | 0.482 / 0.849 / 0.615 | 0.677 / 0.915 / 0.778 |
| validators on | 0.969 / 0.849 / 0.905 | 0.777 / 0.892 / 0.831 |

Validators leave `ssn` recall unchanged and raise its precision from 0.48 to
0.97. Overall recall falls from 0.915 to 0.892, mostly on `credit card number`
(0.921 to 0.556), whose synthetic values often fail the Luhn check.

## Files

| File | What |
|---|---|
| `gretel-heldout500.*` | held-out sample, GLiNER2-PII, baseline (headline) |
| `gretel-heldout500-validators.*` | held-out sample, GLiNER2-PII, validators on (headline) |
| `gretel.*` | tuning sample, both backends, baseline |
| `gretel-validators.*` | tuning sample, both backends, validators on |
| `gretel-concat25.*` | tuning sample joined 25 documents per text, to exercise windowing |
| `gretel-ablation.*` | tuning sample, extra-label ablation; its validator rows predate the `ssn` validator fix |
