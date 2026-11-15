# Free Model Aggregator

One server that fronts ten free-tier model providers behind the two APIs clients already speak:
OpenAI Chat Completions and Anthropic Messages, in JSON and in server-sent events.

It reads the trial brief — the provider directory, the sealed quality table and the workload
manifest — probes each provider's own model list, and registers one catalog. Each request is
answered by the best free model that can serve it:

- **Planned, not greedy.** The whole manifest is planned at once (weight × rating, with a
  per-provider allowance of primaries), so scarce capacity — vision models, 200k-context models,
  the top-rated provider's budget — is reserved for the requests that can only go there.
- **Failover by provider.** Budgets, outages and failure schedules are provider-level, so a
  refusal moves the request to the next provider, never to a second model on the one that just
  refused. Three refusals in a row mark a provider dead (a schedule holds at most two) and demote
  it behind every live provider.
- **Wire-faithful.** Requests are translated between the OpenAI, Anthropic and Gemini wires
  preserving exactly what a provider echoes back: system text, user text, tool names, max_tokens,
  stop sequences, image hashes and the JSON-mode flag. Upstream calls are never streamed; the
  client's stream is synthesized from the whole answer.
- **Deterministic.** Every choice is a function of the brief and the providers' answers — no
  clock, no randomness — so a replayed trial leaves the same state.

Run it with `node main.ts` (Node 22, standard library only), or `node main.ts --list` to print
the catalog it registers.

MIT licensed.
