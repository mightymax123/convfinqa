## Summary

Interface and architecture cleanup across the agent, judge, evaluator, and response-generation layers. No functional behaviour changes.

- `max_retries` parameter removed from `get_response` and `get_judge_response` public signatures; both functions now read `max_retries` directly from `get_settings()` internally.
- Retry loops corrected to `range(1, max_retries + 1)` to produce exactly `max_retries` attempts.
- Callers (`evaluator.py`, `generate_responses.py`) have their now-redundant `max_retries` attributes and `get_settings` imports removed.
- `ResultsWriter` construction in `main.py` moved to just before `writer.save_outputs(...)` to reflect actual execution order.
- All tests updated to match the new signatures and to patch `get_settings` directly for retry-count control.

## Why

Passing `max_retries` as an explicit argument to `get_response` / `get_judge_response` duplicated configuration that is already owned by `Settings`. Removing the parameter eliminates the duplication, makes the call sites simpler, and ensures there is a single source of truth for retry behaviour. The `main.py` reordering makes the construction sequence match the logical pipeline flow.

## Implementation notes

- `app/agent.py` and `app/judge.py`: `build_agent` / `build_judge_agent` now use a single `settings = get_settings()` call and read attributes off it, removing the intermediate `max_retries` local variable. `get_response` / `get_judge_response` each call `get_settings().max_retries` at the top of the function.
- `app/evaluator.py`: removed `self.max_retries` attribute and the `get_settings` import; updated `get_judge_response` call site to drop `max_retries` kwarg.
- `app/generate_responses.py`: removed `self.max_retries` attribute, the `settings = get_settings()` block, and the `get_settings` import; updated `get_response` call site to drop `max_retries` kwarg.
- `app/main.py`: `ResultsWriter(...)` construction moved immediately before `writer.save_outputs(...)`.
- Tests: retry-behaviour tests now patch `get_settings` at `app.agent.get_settings` / `app.judge.get_settings` with a `Settings(max_retries=N)` instance. Removed stale patch of `app.evaluator.get_settings` (no longer imported there).

## Testing

- All 66 existing tests updated and pass.
- All checks clean (`all-checks`).
- No new functionality introduced; no new tests required beyond updating existing call sites and patch targets.

## Screenshots (optional)

N/A

## Checklist

- [ ] Performed self-review
- [ ] Manually tested end-to-end
- [x] Written tests for any new functionality
- [ ] Updated the README if appropriate
- [ ] Updated `sample.env` if env vars changed
- [ ] Updated the README env var table if env vars changed
- [x] Conventional commit message used (`feat:`, `fix:`, `chore:`, etc.)
