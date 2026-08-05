# Phase 2 API cost report

- Model/API: `gpt-5.6-luna`, OpenAI Responses API, Structured Outputs, no tools, `store=false`
- Actual paid responses: 432
- Actual tokens: `{"cache_write_tokens": 0, "cached_input_tokens": 0, "input_tokens": 155816, "output_tokens": 303830, "reasoning_tokens": 203692}`
- Paid cost: $0.3957592
- Hard paid limit: $4.50
- Remaining paid budget: $4.1042408
- Active reservations: 0
- Hypothetical full two-candidate Batch cost from medium mean: $12.99
- Conservative p95 + margin cost: $60.40

No main Batch was submitted because the medium audit failed. The conservative figure is informational and is not authorization to bypass the gate.

Official references: [Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [pricing](https://developers.openai.com/api/docs/pricing), and [Batch API](https://developers.openai.com/api/docs/guides/batch).
