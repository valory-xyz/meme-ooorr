# External Request Resilience Audit

Deep analysis of every external HTTP dependency: what happens under each failure mode
(HTTP errors, unreachable, malformed data, empty 200s) and how failures propagate through the FSM.

---

## How the framework handles HTTP

### Stack 1: Framework path (`get_http_response` via `BaseBehaviour`)

Used by behaviours in `base.py` for Olas Subgraph, IPFS metadata, Meme Subgraph, and Fireworks AI.

- Returns `HttpMessage` with `status_code`, `status_text`, `body`
- **Unreachable / DNS / timeout**: HTTP client catches all exceptions, returns `status_code=600` with traceback in `body`
- Behaviours check `status_code != 200` and return `None` or skip processing
- **Round timeout**: `MemeooorrEvent.ROUND_TIMEOUT` = 300s (30s base x10 multiplier). On timeout, most rounds self-loop.

### Stack 2: Direct `requests` library

Used for IPFS media download (`mech.py`), LiFi quote (`handlers.py`), and Web3 RPC (`handlers.py`).

- Raises `requests.exceptions.RequestException` hierarchy on network errors
- `JSONDecodeError` from `response.json()` is a `ValueError`, NOT `RequestException`

### Stack 3: Third-party library wrappers

- **Twikit** (`twikit/connection.py`): async `Connection` using twikit library. Internal HTTP managed by twikit. **Uses blocking `time.sleep()` in async methods.**
- **Tweepy** (`tweepy/connection.py`): `BaseSyncConnection` with `MAX_WORKER_THREADS=1`. Uses tweepy library for Twitter API v2.
- **GenAI** (`genai/connection.py`): `BaseSyncConnection` with `MAX_WORKER_THREADS=1`. Uses `google.generativeai` library or x402-wrapped `requests`.
- **MirrorDB** (`mirror_db/connection.py`): async `Connection` using `aiohttp`. Exponential backoff retry (5 retries, factor 2).

### Handler dispatch

`HttpHandler.handle()` at `handlers.py:458` dispatches to handler methods **without a global try-except**. If a handler crashes, no HTTP response is sent.

### FSM composition

The chained app composes 6 FSMs:
```
Registration → AgentPerformanceSummary → Memeooorr → TransactionSettlement → ResetAndPause → MechInteract
```

Terminal round mappings:
| Memeooorr Terminal | Maps To |
|---|---|
| `FinishedToResetRound` | `ResetAndPauseRound` |
| `FinishedToSettlementRound` | `RandomnessTransactionSubmissionRound` |
| `FinishedForMechRequestRound` | `MechVersionDetectionRound` |
| `FinishedForMechResponseRound` | `MechResponseRound` |

Timeouts: `MemeooorrEvent.ROUND_TIMEOUT` = 300s, `MechInteractEvent.ROUND_TIMEOUT` = 600s, `ResetPauseEvent.ROUND_TIMEOUT` = 30s.

---

## 1. Olas Subgraph (GraphQL)

| | |
|---|---|
| **Base URL** | Configurable via `params.olas_subgraph_url` (default: `https://api.subgraph.autonolas.tech/api/proxy/autonolas-base`) |
| **Endpoints** | GraphQL POST |
| **Called from** | `base.py:625` (`get_packages`) |
| **Method** | POST |
| **Purpose** | Query registered agent/service packages |

### Failure matrix

| Failure mode | `get_packages` (base.py:625) |
|---|---|
| **HTTP 500** | status_code != 200 → returns `None` |
| **Unreachable / DNS / timeout** | status_code=600 → returns `None` |
| **200 but non-JSON body** | `json.loads(response.body)` at line 639 → **JSONDecodeError CRASH** |
| **200 but `{}`** | `"data" not in response_body` at line 642 → returns `None` |
| **200 but `{"data": null}`** | `"data" not in response_body` is False → returns `None` (null is still a key) — then callers access `None["units"]` → **TypeError CRASH** |

### FSM impact

```
Subgraph unreachable → get_packages() returns None
→ get_memeooorr_handles_from_subgraph() returns empty list
→ Twitter engagement proceeds without Memeooorr peer handles
→ No crash, degraded functionality
```

```
Subgraph returns {"data": null} → get_packages() returns None
→ Same as above, graceful degradation (callers check for None)
```

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **LOW** | `base.py:639` | `json.loads(response.body)` after 200 check — CDN could return 200 with HTML. Very unlikely for a GraphQL endpoint. |
| **MEDIUM** | `base.py:648` | Returns `response_body["data"]` — if value is `None`, callers iterate on `None`. However, `get_memeooorr_handles_from_subgraph` at line 654 checks `if not services: return handles`, which catches `None`. Safe. |

---

## 2. IPFS Gateway (Service Metadata)

| | |
|---|---|
| **Base URL** | Hardcoded: `https://gateway.autonolas.tech/ipfs/{ipfs_hash}` |
| **Called from** | `base.py:728` (`get_memeooorr_handles_from_chain`) |
| **Method** | GET |
| **Purpose** | Fetch service metadata JSON from IPFS |

### Failure matrix

| Failure mode | `get_memeooorr_handles_from_chain` (base.py:728) |
|---|---|
| **HTTP 500 / Unreachable** | status_code != 200 → `continue` (skips this service, proceeds to next) |
| **200 but non-JSON body** | `json.loads(response.body)` at line 739 → **JSONDecodeError CRASH** |
| **200 but missing "description" key** | `metadata["description"]` at line 740 → **KeyError CRASH** |

### FSM impact

```
IPFS returns non-JSON for one service → json.loads crashes
→ Behaviour generator raises JSONDecodeError (uncaught)
→ Agent process crashes
→ Requires external restart; crash-loops if IPFS keeps returning bad data
```

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **HIGH** | `base.py:739` | `json.loads(response.body)` can crash on non-JSON 200 response from IPFS gateway/CDN (BP5/BP1) |
| **HIGH** | `base.py:740` | `metadata["description"]` — direct key access without `.get()` (BP6). If IPFS metadata lacks "description", KeyError crashes the behaviour. |

---

## 3. IPFS Gateway (Media Download)

| | |
|---|---|
| **Base URL** | Hardcoded: `https://gateway.autonolas.tech/ipfs/{ipfs_hash}` |
| **Called from** | `mech.py:274` (`_fetch_media_from_ipfs_hash`) |
| **Method** | GET (streaming) |
| **Purpose** | Download video/image media from IPFS |

### Failure matrix

| Failure mode | `_fetch_media_from_ipfs_hash` (mech.py:274) |
|---|---|
| **HTTP 500 / 4xx** | `raise_for_status()` → `HTTPError` caught at line 296 → returns `None` |
| **Unreachable / timeout** | `Timeout` caught at line 291, `RequestException` caught at line 301 → returns `None` |
| **200 but empty body** | Downloaded 0 bytes → `_download_and_save_media` returns `None` → returns `None` |

### FSM impact

```
IPFS unreachable → _fetch_media_from_ipfs_hash returns None
→ PostMechResponseBehaviour sends payload with mech_for_twitter=False, failed_mech=False
→ PostMechResponseRound Event.DONE → EngageTwitterRound
→ Agent continues without media
```

### Bugs found

None. Error handling is comprehensive with separate except clauses for Timeout, HTTPError, and RequestException. Streaming download with size verification.

---

## 4. Meme Subgraph (GraphQL)

| | |
|---|---|
| **Base URL** | Configurable via `params.meme_subgraph_url` (default: `https://agentsfun-indexer-production.up.railway.app`) |
| **Called from** | `base.py:779` (`get_meme_coins_from_subgraph`) |
| **Method** | POST |
| **Purpose** | Query meme token data (1000 tokens per request) |

### Failure matrix

| Failure mode | `get_meme_coins_from_subgraph` (base.py:779) |
|---|---|
| **HTTP 500 / Unreachable** | status_code != 200 → returns `[]` |
| **200 but non-JSON body** | `json.loads(response.body)` at line 792 → **JSONDecodeError CRASH** |
| **200 but `{"data": null}`** | `response_json["data"]` at line 813 returns `None` → `None["memeTokens"]` → **TypeError CRASH** |
| **200 but missing nested keys** | `response_json["data"]["memeTokens"]["items"]` at line 813 → **KeyError CRASH** |

### FSM impact

```
Subgraph returns {"data": null} → TypeError at line 813
→ Behaviour generator crashes in PullMemesBehaviour
→ Agent process crashes
→ Requires external restart; crash-loops if subgraph keeps returning bad data
```

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **HIGH** | `base.py:792` | `json.loads(response.body)` can crash on non-JSON 200 response (BP1) |
| **HIGH** | `base.py:813` | `response_json["data"]["memeTokens"]["items"]` — triple direct key access without `.get()` (BP6). Any missing or null key crashes the behaviour. This is the **most likely crash path** since the meme subgraph is a custom service (not infrastructure-grade). |

---

## 5. Fireworks AI LLM API

| | |
|---|---|
| **Base URL** | Configurable via `model_config.url` (default: `https://api.fireworks.ai/inference/v1/chat/completions`) |
| **Called from** | `base.py:958` (`replace_tweet_with_alternative_model`) |
| **Method** | POST |
| **Purpose** | Generate alternative tweet text using LLM |

### Failure matrix

| Failure mode | `replace_tweet_with_alternative_model` (base.py:958) |
|---|---|
| **HTTP 500** | Logs error at line 967 but **falls through** to `json.loads(response.body)` at line 972 — if body is JSON, checks for error key; if body is non-JSON → **JSONDecodeError CRASH** |
| **Unreachable (status=600)** | Same: logs error, falls through to `json.loads` on traceback body → **JSONDecodeError CRASH** |
| **200 but `{"error": ...}`** | Caught at line 974, returns `None` |
| **200 but unexpected schema** | Caught by broad `except Exception` at line 982, returns `None` |

### FSM impact

```
Fireworks unreachable → status_code=600 → json.loads(traceback) crashes
→ Behaviour (EngageTwitter or ActionDecision) generator raises JSONDecodeError
→ Agent process crashes
→ Requires external restart; crash-loops if Fireworks stays unreachable
```

Called from `twitter.py:1432` (EngageTwitterBehaviour) and `llm.py:371` (ActionDecisionBehaviour).

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **HIGH** | `base.py:966-972` | **Missing early return on non-200**. The code logs a non-200 error but does NOT return — it falls through to `json.loads(response.body)` which crashes on non-JSON body (status 600 traceback, HTML error pages). Fix: add `return None` after the error log at line 969. |

---

## 6. LiFi Quote API

| | |
|---|---|
| **Base URL** | Configurable via `params.lifi_quote_to_amount_url` (default: `https://li.quest/v1/quote/toAmount`) |
| **Called from** | `handlers.py:1298` (`_get_lifi_quote_sync`) |
| **Method** | GET |
| **Purpose** | Get token swap quotes for x402 USDC top-up |

### Failure matrix

| Failure mode | `_get_lifi_quote_sync` (handlers.py:1298) |
|---|---|
| **HTTP 500 / 4xx** | status_code != 200 → returns `None` |
| **Unreachable / timeout** | Caught by `except Exception` → returns `None` |
| **200 but non-JSON** | `response.json()` raises `JSONDecodeError` → caught by `except Exception` → returns `None` |

### FSM impact

This runs in a **background thread** (ThreadPoolExecutor, max_workers=1) during handler `setup()`. Not on the FSM path.

```
LiFi unreachable → returns None → _ensure_sufficient_funds returns False
→ shared_state.sufficient_funds_for_x402_payments = False
→ Agent proceeds but x402 payments may fail
```

### Bugs found

None. Broad `except Exception` catches all failures. Timeout of 30s configured.

---

## 7. Web3 RPC (Base Chain)

| | |
|---|---|
| **Base URL** | Configurable via `params.base_ledger_rpc` (default: `https://1rpc.io/base`) |
| **Called from** | `handlers.py:1243` (`_get_web3_instance`), used by `_check_usdc_balance`, `_sign_and_submit_tx_web3`, `_check_transaction_status`, `_get_nonce_and_gas_web3`, `_estimate_gas` |
| **Method** | POST (JSON-RPC) |
| **Purpose** | On-chain interactions for x402 USDC fund management |

### Failure matrix

| Failure mode | All Web3 methods |
|---|---|
| **RPC unreachable** | Caught by broad `except Exception` in each method → returns `None`/`False` |
| **RPC timeout** | Web3 HTTPProvider default timeout applies |
| **Transaction timeout** | `wait_for_transaction_receipt(timeout=60)` at line 1344 → `TimeExhausted` exception → caught → returns `False` |

### FSM impact

All Web3 calls run in the **background thread** (handler executor). Not on FSM path.

```
RPC unreachable → _ensure_sufficient_funds returns False
→ sufficient_funds_for_x402_payments = False
→ x402 payments disabled
```

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **MEDIUM** | `handlers.py:1389` | **BP11: `_estimate_gas` returns `False` instead of `None`**. Declared return type is `Optional[int]`, but returns `False` when `w3 is None`. Caller at line 1482 checks `if tx_gas is None:` — `False is not None` so it passes through. `tx_data["gas"] = False` → `eoa_account.sign_transaction(tx_data)` likely crashes. Mitigated by broad `except Exception` in `_sign_and_submit_tx_web3` at line 1326, but masks the real error. |
| **LOW** | `handlers.py:1243` | Creates a new `Web3(HTTPProvider(rpc_url))` on every call. The comment at line 1240-1242 notes this is suboptimal for TCP recycling. |

---

## 8. MirrorDB Backend

| | |
|---|---|
| **Base URL** | Configurable via `mirror_db_base_url` |
| **Endpoints** | Dynamic via `endpoint` parameter (CRUD operations) |
| **Called from** | `mirror_db/connection.py:348-405` (`create_`, `read_`, `update_`, `delete_`) |
| **Method** | POST/GET/PUT/DELETE |
| **Purpose** | Backend data persistence |

### Failure matrix

| Failure mode | All CRUD methods |
|---|---|
| **HTTP 500** | `_raise_for_response` → `response.json()` → if body is not JSON → **aiohttp.ContentTypeError** (not caught by retry decorator's `ClientResponseError`/`ClientConnectionError`) → propagates to `_get_response`'s `except Exception` → error response sent |
| **HTTP 429** | Retry decorator catches `ClientResponseError` with status 429 → exponential backoff (max 5 retries) |
| **Unreachable** | `ClientConnectionError` → retry with backoff |
| **Timeout** | aiohttp ClientSession default timeout → `asyncio.TimeoutError` → NOT caught by retry decorator → propagates to `_get_response`'s `except Exception` |

### FSM impact

MirrorDB is used by behaviours via the SRR protocol. Failures result in error responses sent back to the behaviour, which handles them gracefully.

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **LOW** | `mirror_db/connection.py:339` | `_raise_for_response` calls `await response.json()` on non-200 responses. If error body is not JSON (e.g., HTML 502 from proxy), `aiohttp.ContentTypeError` is raised. Not caught by the retry decorator (only catches `ClientResponseError`/`ClientConnectionError`), but caught by `_get_response`'s broad `except Exception`. |
| **LOW** | `mirror_db/connection.py:181` | No explicit timeout on `aiohttp.ClientSession`. Default is 300s. `asyncio.TimeoutError` not caught by retry decorator. |

---

## 9. Twitter API (Twikit)

| | |
|---|---|
| **Base URL** | Internal to twikit library (Twitter API) |
| **Called from** | `twikit/connection.py` (search, post, get_user_tweets, like, retweet, follow, upload_media) |
| **Purpose** | Twitter posting, searching, engagement |

### Failure matrix

| Failure mode | All twikit methods |
|---|---|
| **Twitter API error** | `twikit.errors.TwitterException` → caught at method level → error returned |
| **Account locked/suspended** | `AccountLocked`/`AccountSuspended`/`Unauthorized` → error message returned |
| **Network timeout** | Depends on twikit library internals — **no explicit timeout configured** |

### FSM impact

Twikit is called from behaviours via SRR protocol. Errors result in error responses. The EngageTwitterBehaviour handles errors by returning Event.ERROR, which self-loops EngageTwitterRound.

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **HIGH** | `twikit/connection.py:287,301,352,439,477,495,503,555,627` | **`time.sleep()` in async methods blocks the entire event loop.** TwikitConnection extends `Connection` (async), not `BaseSyncConnection`. All `time.sleep()` calls (rate limiting at line 287, random delays at 301, retry waits at 352/477/495) block the event loop thread. During tweet verification (lines 471-479), up to 10 retries x 3s = **30 seconds of total event loop blocking**. Should use `await asyncio.sleep()`. |
| **MEDIUM** | `twikit/connection.py:219` | `_handle_done_task` calls `task.result()` without try-except. Compare with mirror_db's version at line 252 which wraps in try-except. If `_get_response` raises unexpectedly, the exception propagates to asyncio's error handler. |
| **MEDIUM** | `twikit/connection.py:252` | `json.loads(srr_message.payload)` is **outside** the try-except at line 298 (BP14). If payload is malformed JSON (unlikely since skill constructs it), JSONDecodeError escapes `_get_response`, no response is sent, and the requesting behaviour times out. |

---

## 10. Twitter API (Tweepy)

| | |
|---|---|
| **Base URL** | Internal to tweepy library (Twitter API v2) |
| **Called from** | `tweepy/connection.py` (post, like, retweet, follow, get_me, get_user_tweets) |
| **Purpose** | Twitter posting, engagement, account info retrieval |

### Failure matrix

| Failure mode | All tweepy methods |
|---|---|
| **API error** | `TweepyException` → caught by `except Exception` in `_get_response` at line 271 → error dict returned |
| **Auth failure** | twitter client is `None` → returns error dict |
| **Network timeout** | **No timeout configured** — tweepy uses requests internally with no timeout parameter |

### FSM impact

Tweepy is called via SRR protocol. Errors return `{"error": ...}` dicts. Behaviours handle these.

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **MEDIUM** | `tweepy/connection.py` | **No timeout on tweepy HTTP calls** (CC5). `BaseSyncConnection` with `MAX_WORKER_THREADS=1` — a hanging tweepy call blocks all subsequent tweepy requests indefinitely. The behaviour waiting for the response times out after 300s, but the connection thread remains blocked. |
| **LOW** | `tweepy/connection.py:204` | `json.loads(srr_message.payload)` outside `_get_response`'s try-except. If it fails, exception escapes `on_send`. In BaseSyncConnection, `_task_done_callback` logs but does NOT send response — behaviour times out. (BP14) |

---

## 11. Google GenAI / Gemini

| | |
|---|---|
| **Base URL** | genai library (internal), or configurable `genai_x402_server_base_url` for x402 path |
| **Called from** | `genai/connection.py:283-302` (`_get_response`) |
| **Purpose** | LLM text generation (tweet ideas, persona updates) |

### Failure matrix

| Failure mode | `_get_response` |
|---|---|
| **API error** | `google.generativeai` exceptions → caught by `except Exception` at line 301 → error returned |
| **x402 API error** | `ValueError` from `_process_x402_request` → caught by `except Exception` → error returned |
| **Network timeout** | genai library uses requests internally — **no explicit timeout** |
| **x402 non-JSON response** | `response.json()` at line 223 in `_process_x402_request` → `JSONDecodeError` → caught by broad `except Exception` at line 301 |

### FSM impact

GenAI is called via SRR protocol from behaviours. Errors return `{"error": ...}` dicts.

### Bugs found

| Severity | Location | Bug |
|---|---|---|
| **MEDIUM** | `genai/connection.py:165` | **BP14: `json.loads(srr_message.payload)` outside `_get_response`'s try-except.** In `on_send()`, `json.loads(srr_message.payload)` at line 165 is before `_get_response`. If it fails, the exception escapes `on_send`. In `BaseSyncConnection`, `_run_in_pool` calls `on_send` via `executor.submit`. The `_task_done_callback` logs the exception but **no response envelope is sent**. The skill behaviour waiting for the response times out after 300s. |
| **LOW** | `genai/connection.py` | **No timeout on genai library HTTP calls** (CC5). `MAX_WORKER_THREADS=1` — a hanging genai call blocks all subsequent genai requests. |

---

## 12. Agent DB (Valory)

| | |
|---|---|
| **Base URL** | `https://afmdb.autonolas.tech` |
| **Called from** | Via `AgentDBBehaviour` parent class |
| **Purpose** | Store/retrieve agent performance data |

This endpoint uses the framework's `get_http_response` path and is managed by the `agent_db_abci` skill. Although `agent_db_abci` is a dev package (valory-authored, listed in `packages.json` under `dev`), its error handling follows standard framework patterns. Not analyzed in detail as it uses the standard `AgentDBBehaviour` base class.

---

## Summary: All Bugs Found

| # | Severity | Location | Bug |
|---|----------|----------|-----|
| 1 | **HIGH** | `base.py:966-972` | Fireworks API: missing early return on non-200 causes `json.loads` crash on non-JSON body (status 600 traceback, HTML error pages) |
| 2 | **HIGH** | `base.py:813` | Meme subgraph: `response_json["data"]["memeTokens"]["items"]` — triple direct key access crashes on null/missing keys (BP6) |
| 3 | **HIGH** | `base.py:739-740` | IPFS metadata: `json.loads` + `metadata["description"]` crash on non-JSON body or missing key (BP1/BP6) |
| 4 | **HIGH** | `twikit/connection.py:287+` | `time.sleep()` in async methods blocks entire event loop for up to 30s |
| 5 | **MEDIUM** | `handlers.py:1389` | `_estimate_gas` returns `False` instead of `None` — passes through `is None` check (BP11) |
| 6 | **MEDIUM** | `handlers.py:458` | Handler dispatch without global try-except — handler crash = no HTTP response (BP10) |
| 7 | **MEDIUM** | `genai/connection.py:165` | `json.loads` outside `_get_response` try-except — exception escapes `on_send` (BP14) |
| 8 | **MEDIUM** | `twikit/connection.py:219` | `_handle_done_task` doesn't catch exceptions from `task.result()` |
| 9 | **MEDIUM** | `twikit/connection.py:252` | `json.loads` outside try-except — exception escapes `_get_response` (BP14) |
| 10 | **MEDIUM** | `tweepy/connection.py` | No timeout on tweepy HTTP calls — can block connection thread indefinitely (CC5) |
| 11 | **LOW** | `mirror_db/connection.py:339` | `_raise_for_response` assumes error body is JSON |
| 12 | **LOW** | `tweepy/connection.py:204` | `json.loads` outside `_get_response` try-except (BP14) |
| 13 | **LOW** | `genai/connection.py` | No timeout on genai library HTTP calls (CC5) |
| 14 | **LOW** | `mirror_db/connection.py:181` | No explicit timeout on aiohttp session |

---

## Operational Impact Classification

### A. What can CRASH the agent process

| # | Trigger | Where exception escapes | How external failure causes it |
|---|---------|------------------------|-------------------------------|
| 1 | Fireworks API unreachable or returns non-JSON | `base.py:972` — `json.loads(response.body)` | status_code=600, body=traceback string → JSONDecodeError |
| 2 | Meme subgraph returns `{"data": null}` or missing keys | `base.py:813` — direct `[]` access | GraphQL error/schema change → KeyError/TypeError |
| 3 | IPFS gateway returns non-JSON or metadata missing "description" | `base.py:739-740` | CDN error page or malformed metadata → JSONDecodeError/KeyError |

**What happens after a behaviour crash:**

1. The generator raises an exception (JSONDecodeError/KeyError/TypeError)
2. **The agent process crashes.** An uncaught exception in a behaviour generator is fatal — the agent does not recover automatically.
3. The agent must be restarted externally (e.g., by a process supervisor, Docker restart policy, or manual intervention).
4. If the external service is still failing after restart, the agent crashes again immediately upon reaching the same behaviour.

**Net assessment:** Bug #2 (meme subgraph direct key access) is **most likely and impactful** because:
- The meme subgraph is a custom Railway-hosted service (not infrastructure-grade)
- It's called every period in `PullMemesRound`
- A schema change or transient error with `{"data": null}` would crash the agent immediately
- Triple-nested direct key access (`["data"]["memeTokens"]["items"]`) means three potential crash points
- **The agent enters a crash loop** if the subgraph continues returning bad data after restart

---

### B. What can get the agent STUCK

| # | Trigger | Mechanism | Duration | Recovery |
|---|---------|-----------|----------|----------|
| 1 | Twikit rate limiting / retries | `time.sleep()` blocks event loop | Up to 30s per verification (10 retries x 3s) | Automatic after sleep completes |
| 2 | Tweepy hanging HTTP call | No timeout, blocks connection thread | Indefinite | Requires agent restart |
| 3 | GenAI hanging HTTP call | No timeout, blocks connection thread | Indefinite | Requires agent restart |

**Net assessment:** Bug #4 (twikit `time.sleep()` blocking event loop) is **most impactful** because:
- It blocks ALL async operations (not just twikit) — healthcheck responses, other connections, Tendermint communication
- The 5-second minimum delay between calls (line 286-287) runs on EVERY request
- Tweet posting with verification can block for 30+ seconds
- `filter_suspended_users` iterates users with `time.sleep(randbelow(5))` per user — for N users, blocks N*2.5s average

---

### C. Agent keeps running with UNINTENDED SIDE-EFFECTS

| # | Trigger | Side-effect | Severity | Financial impact |
|---|---------|-------------|----------|-----------------|
| 1 | `_estimate_gas` returns `False` | `tx_data["gas"] = False` → `sign_transaction` fails with confusing error | Low | No — caught by broad except, but masks root cause |
| 2 | Handler crash (no try-except) | Client HTTP request hangs until timeout, no response | Low | No — UI only |
| 3 | Meme subgraph returns `[]` (empty) | Agent operates without meme token data | Low | Missed trading opportunities |

**Net assessment:** Side-effects are **low risk** in this codebase. The x402 fund management runs in a background thread and failures are isolated from the FSM. Meme token operations fail gracefully (empty lists).

---

## Cross-cutting Issues

### 1. No circuit breaker (CC2)

When an external service is down, the agent retries every period (300s round timeout → reset → new period → retry). No backoff between periods. Under sustained Fireworks/subgraph outage, the agent:
- Logs errors every 300s
- Wastes round timeout sleeping
- Never makes progress on Twitter engagement

**Recommendation:** Add a configurable backoff for external service failures that persists across periods.

### 2. Inconsistent retry strategies (CC1)

| Call site | Retries | Backoff | Max wait | Timeout |
|-----------|---------|---------|----------|---------|
| MirrorDB CRUD | 5 | Exponential (2x) | ~31s | None (aiohttp default) |
| Twikit post | 5 | None | Immediate | None |
| Twikit verify | 10 | Fixed 3s | 30s | None |
| Tweepy delete | 5 | Fixed 3s | 15s | None |
| IPFS media | 0 | N/A | N/A | 120s |
| LiFi quote | 0 | N/A | N/A | 30s |
| Fireworks | 0 | N/A | N/A | Framework default |
| Olas subgraph | 0 | N/A | N/A | Framework default |
| Meme subgraph | 0 | N/A | N/A | Framework default |

**Observation:** Twitter posting (financial impact: none) has 5 retries, while subgraph queries (needed for token operations) have zero retries.

### 3. Missing timeouts (CC5)

| Component | Missing timeout | Risk |
|-----------|----------------|------|
| Tweepy (BaseSyncConnection, 1 thread) | requests calls | Thread blocked indefinitely |
| GenAI (BaseSyncConnection, 1 thread) | genai library calls | Thread blocked indefinitely |
| MirrorDB (aiohttp) | Session-level timeout | 300s default |
| Twikit | twikit library calls | Event loop blocked |

### 4. `time.sleep()` in async context

The twikit connection uses blocking `time.sleep()` in 9 locations within async methods. This is the single biggest systemic issue — it blocks the entire event loop, affecting all connections and the agent's responsiveness. Every call should be `await asyncio.sleep()`.

---

## Combined Priority Matrix

| Priority | Issue | Category | Fix complexity | Fix description |
|----------|-------|----------|----------------|-----------------|
| **P0** | #1: Fireworks json.loads on non-200 | Crash | **Low** | Add `return None` after the error log at `base.py:969` |
| **P0** | #2: Meme subgraph direct key access | Crash | **Low** | Replace `response_json["data"]["memeTokens"]["items"]` with `.get()` chain + null guards at `base.py:813` |
| **P0** | #3: IPFS metadata crash | Crash | **Low** | Wrap `json.loads` in try-except and use `.get("description")` at `base.py:739-740` |
| **P0** | #4: Twikit time.sleep() blocks event loop | Stuck | **Medium** | Replace all `time.sleep()` with `await asyncio.sleep()` across 9 locations in `twikit/connection.py` |
| **P1** | #5: _estimate_gas returns False | Side-effect | **Low** | Change `return False` to `return None` at `handlers.py:1389` |
| **P2** | #6: Handler dispatch no try-except | Crash (handler) | **Low** | Wrap `handler(http_msg, http_dialogue, **kwargs)` in try-except at `handlers.py:458` |
| **P2** | #7: GenAI json.loads outside try | Stuck | **Low** | Move `json.loads` inside `_get_response` or add try-except in `on_send` at `genai/connection.py:165` |
| **P2** | #8: Twikit _handle_done_task | Stuck | **Low** | Add try-except around `task.result()` at `twikit/connection.py:219` |
| **P2** | #9: Twikit json.loads outside try | Stuck | **Low** | Move `json.loads` inside the try-except at `twikit/connection.py:252` |
| **P2** | #10: Tweepy no timeout | Stuck | **Low** | Pass `timeout=` to tweepy's internal requests calls (may need wrapper) |
| **P3** | CC1: Inconsistent retries | Systemic | **Medium** | Standardize retry strategy; add retries to subgraph queries |
| **P3** | CC2: No circuit breaker | Systemic | **High** | Implement circuit breaker for external services |
| **P3** | CC5: Missing timeouts | Systemic | **Medium** | Add timeouts to genai, tweepy, mirrordb sessions |
| **P4** | #11: MirrorDB error JSON assumption | Handler | **Low** | Add try-except around `response.json()` in `_raise_for_response` |
| **P4** | #12-13: Tweepy/GenAI json.loads | Defensive | **Low** | Move json.loads inside try-except in `on_send` |
| **P4** | #14: MirrorDB session timeout | Config | **Low** | Add `timeout=aiohttp.ClientTimeout(total=60)` to session |
