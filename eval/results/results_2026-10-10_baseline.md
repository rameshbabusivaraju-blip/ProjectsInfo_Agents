# Golden set results, 2026-10-10

Each question was answered 3 time(s) per agent. A run passes when every check of the question passes. Calls, tokens and live calls are averages per run; cost is the total for all runs of that row, estimated from `PRICES_PER_MILLION_TOKENS` in `eval/evaluate.py`.

| Question | Agent | Passed | Model calls | Tokens | Live calls | Cost |
|---|---|---|---|---|---|---|
| A1 | fixed | 3/3 | 2.0 | 2,608 | 0.0 | $0.0111 |
| A1 | loop | 3/3 | 2.0 | 8,466 | 0.0 | $0.0553 |
| C1 | fixed | 3/3 | 2.0 | 2,420 | 0.0 | $0.0088 |
| C1 | loop | 3/3 | 2.7 | 11,153 | 0.0 | $0.0706 |
| J1 | fixed | 3/3 | 2.0 | 2,393 | 0.0 | $0.0083 |
| J1 | loop | 3/3 | 2.3 | 9,727 | 0.0 | $0.0614 |
| C2 | fixed | 3/3 | 2.0 | 2,393 | 0.0 | $0.0083 |
| C2 | loop | 3/3 | 2.7 | 11,136 | 0.0 | $0.0700 |
| C3 | fixed | 3/3 | 2.0 | 2,413 | 0.0 | $0.0086 |
| C3 | loop | 0/3 | 3.3 | 17,127 | 1.3 | $0.1136 |
| C5 | fixed | 3/3 | 2.0 | 2,436 | 0.0 | $0.0089 |
| C5 | loop | 3/3 | 2.0 | 8,291 | 0.0 | $0.0525 |
| J3 | fixed | 3/3 | 2.0 | 2,588 | 0.0 | $0.0109 |
| J3 | loop | 3/3 | 2.0 | 8,502 | 0.0 | $0.0566 |
| B1 | fixed | 3/3 | 2.0 | 2,430 | 0.0 | $0.0089 |
| B1 | loop | 3/3 | 2.0 | 8,270 | 0.0 | $0.0522 |
| A5 | fixed | 3/3 | 2.0 | 2,833 | 0.0 | $0.0141 |
| A5 | loop | 3/3 | 2.3 | 10,544 | 0.0 | $0.0759 |
| D1 | fixed | 3/3 | 2.0 | 3,159 | 0.0 | $0.0218 |
| D1 | loop | 3/3 | 2.0 | 8,881 | 0.0 | $0.0620 |
| D6 | fixed | 3/3 | 2.0 | 2,500 | 0.0 | $0.0097 |
| D6 | loop | 3/3 | 3.0 | 12,874 | 0.0 | $0.0834 |
| N1 | fixed | 3/3 | 2.0 | 3,531 | 0.0 | $0.0172 |
| N1 | loop | 3/3 | 3.0 | 16,995 | 0.0 | $0.1125 |
| H1 | fixed | 3/3 | 2.0 | 3,712 | 0.0 | $0.0186 |
| H1 | loop | 3/3 | 4.3 | 30,511 | 0.0 | $0.2079 |

## Totals

| Agent | Passed | Pass rate | Cost |
|---|---|---|---|
| fixed | 39/39 | 100% | $0.1551 |
| loop | 36/39 | 92% | $1.0740 |

## Failures

- C3 (loop): rows differ (got 25 rows, expected 0)
