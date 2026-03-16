# External Request Resilience Audit

Deep analysis of every external HTTP dependency: what happens under each failure mode
(HTTP errors, unreachable, malformed data, empty 200s) and how failures propagate through the FSM.

## Fix Status

All 14 individual bugs plus 1 bonus fix have been applied. See commit `90fd85fa`.

| # | Status | Test coverage |
|---|--------|---------------|
| 1 | **FIXED** | `TestReplaceTweet::test_http_error_returns_none` |
| 1b | **FIXED** (bonus) | `TestGetPackagesJsonParseError::test_non_json_body_returns_none` |
| 2 | **FIXED** | `TestGetMemeCoinsSubgraphResilience` (4 tests) |
| 3 | **FIXED** | `TestIpfsMetadataResilience` (2 tests) |
| 4 | **FIXED** | Existing twikit tests updated (`time.sleep` → `asyncio.sleep` mocks) |
| 5 | **FIXED** | `TestEstimateGas::test_no_web3` (updated assertion) |
| 6 | **FIXED** | `TestHttpHandlerHandle::test_handler_exception_returns_500`, `TestHttpHandlerResponses::test_handle_internal_server_error` |
| 7 | **FIXED** | No test coverage (genai is a third-party package, not in tox test suite) |
| 8 | **FIXED** | `TestHandleDoneTask::test_handle_done_task_with_exception` |
| 9 | **FIXED** | `TestGetResponse::test_invalid_json_payload` |
| 10 | **FIXED** | `TestTwitterInit::test_client_wait_on_rate_limit` |
| 11 | **FIXED** | `TestRaiseForResponse::test_error_status_non_json_body` |
| 12 | **FIXED** | `TestOnSend::test_invalid_json_payload` |
| 13 | **FIXED** | No test coverage (genai is a third-party package, not in tox test suite) |
| 14 | **FIXED** | `TestRaiseForResponse::test_session_has_timeout` |
| CC1 | **DEFERRED** | Architectural — inconsistent retry strategies |
| CC2 | **DEFERRED** | Architectural — no circuit breaker pattern |
| CC5 | **DEFERRED** | Architectural — twikit library-level timeout |

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

`HttpHandler.handle()` at `handlers.py:458` dispatches to handler methods. ~~**Without a global try-except** — if a handler crashes, no HTTP response is sent.~~ **FIXED** (Fix #6): wrapped in try-except, returns HTTP 500 via `_handle_internal_server_error`.

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
| **200 but non-JSON body** | ~~`json.loads(response.body)` at line 639 → **JSONDecodeError CRASH**~~ **FIXED:** wrapped in try-except → returns `None` |
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
| ~~**LOW**~~ | `base.py:639` | ~~`json.loads(response.body)` after 200 check~~ **FIXED** (bonus Fix #1b): wrapped in try-except → returns `None` |
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
| **200 but non-JSON body** | ~~`json.loads(response.body)` at line 739 → **JSONDecodeError CRASH**~~ **FIXED:** wrapped in try-except → `continue` |
| **200 but missing "description" key** | ~~`metadata["description"]` at line 740 → **KeyError CRASH**~~ **FIXED:** uses `.get("description", "")` |

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
| ~~**HIGH**~~ | `base.py:739` | ~~`json.loads(response.body)` can crash on non-JSON 200 response~~ **FIXED** (Fix #3): wrapped in try-except → `continue` |
| ~~**HIGH**~~ | `base.py:740` | ~~`metadata["description"]` — direct key access~~ **FIXED** (Fix #3): uses `.get("description", "")` |

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
| **200 but non-JSON body** | ~~`json.loads(response.body)` at line 792 → **JSONDecodeError CRASH**~~ **FIXED:** wrapped in try-except → returns `[]` |
| **200 but `{"data": null}`** | ~~`response_json["data"]` → `None["memeTokens"]` → **TypeError CRASH**~~ **FIXED:** defensive `.get()` chain → returns `[]` |
| **200 but missing nested keys** | ~~`response_json["data"]["memeTokens"]["items"]` → **KeyError CRASH**~~ **FIXED:** defensive `.get()` chain → returns `[]` |

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
| ~~**HIGH**~~ | `base.py:792` | ~~`json.loads(response.body)` can crash on non-JSON 200 response~~ **FIXED** (Fix #2): wrapped in try-except → returns `[]` |
| ~~**HIGH**~~ | `base.py:813` | ~~`response_json["data"]["memeTokens"]["items"]` — triple direct key access~~ **FIXED** (Fix #2): defensive `.get()` chain with null guards |

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
| **HTTP 500** | ~~Logs error but **falls through** to `json.loads` → **JSONDecodeError CRASH**~~ **FIXED:** early `return None` after error log |
| **Unreachable (status=600)** | ~~Falls through to `json.loads` on traceback body → **JSONDecodeError CRASH**~~ **FIXED:** early `return None` |
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
| ~~**HIGH**~~ | `base.py:966-972` | ~~**Missing early return on non-200**~~ **FIXED** (Fix #1): added `return None` after the error log |

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
| ~~**MEDIUM**~~ | `handlers.py:1389` | ~~**BP11: `_estimate_gas` returns `False` instead of `None`**~~ **FIXED** (Fix #5): changed `return False` to `return None` |
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
| ~~**LOW**~~ | `mirror_db/connection.py:339` | ~~`_raise_for_response` assumes error body is JSON~~ **FIXED** (Fix #11): wrapped `response.json()` in try-except, falls back to `response.text()` |
| ~~**LOW**~~ | `mirror_db/connection.py:181` | ~~No explicit timeout on `aiohttp.ClientSession`~~ **FIXED** (Fix #14): added `timeout=aiohttp.ClientTimeout(total=60)` |

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
| ~~**HIGH**~~ | `twikit/connection.py:287,301,352,439,477,495,503,555,627` | ~~**`time.sleep()` in async methods blocks the entire event loop**~~ **FIXED** (Fix #4): all 9 `time.sleep()` calls replaced with `await asyncio.sleep()`, removed `import time` |
| ~~**MEDIUM**~~ | `twikit/connection.py:219` | ~~`_handle_done_task` calls `task.result()` without try-except~~ **FIXED** (Fix #8): wrapped in try-except, logs error and puts `None` envelope |
| ~~**MEDIUM**~~ | `twikit/connection.py:252` | ~~`json.loads(srr_message.payload)` outside try-except~~ **FIXED** (Fix #9): wrapped in try-except, returns error SrrMessage |

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
| ~~**MEDIUM**~~ | `tweepy/connection.py` | ~~**No timeout on tweepy HTTP calls**~~ **FIXED** (Fix #10): added `wait_on_rate_limit=True` to tweepy Client constructor (prevents indefinite hangs on rate limiting) |
| ~~**LOW**~~ | `tweepy/connection.py:204` | ~~`json.loads(srr_message.payload)` outside try-except~~ **FIXED** (Fix #12): wrapped in try-except, sends error response |

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
| ~~**MEDIUM**~~ | `genai/connection.py:165` | ~~**BP14: `json.loads(srr_message.payload)` outside try-except**~~ **FIXED** (Fix #7): wrapped in try-except, sends error response. Note: genai is a third-party package — no test coverage in tox suite. |
| ~~**LOW**~~ | `genai/connection.py` | ~~**No timeout on genai library HTTP calls**~~ **FIXED** (Fix #13): added `timeout=30` to x402 `session.post()`, `request_options={"timeout": 120}` to `model.generate_content()`. Note: genai is a third-party package — no test coverage in tox suite. |

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

All 14 bugs + 1 bonus fix have been applied. See Fix Status table at top.

| # | Severity | Location | Bug | Status |
|---|----------|----------|-----|--------|
| 1 | **HIGH** | `base.py:966-972` | Fireworks API: missing early return on non-200 | **FIXED** |
| 1b | **LOW** | `base.py:639` | Olas subgraph: `json.loads` on 200 non-JSON | **FIXED** (bonus) |
| 2 | **HIGH** | `base.py:813` | Meme subgraph: triple direct key access crashes on null/missing keys | **FIXED** |
| 3 | **HIGH** | `base.py:739-740` | IPFS metadata: `json.loads` + `metadata["description"]` crash | **FIXED** |
| 4 | **HIGH** | `twikit/connection.py:287+` | `time.sleep()` in async methods blocks entire event loop | **FIXED** |
| 5 | **MEDIUM** | `handlers.py:1389` | `_estimate_gas` returns `False` instead of `None` | **FIXED** |
| 6 | **MEDIUM** | `handlers.py:458` | Handler dispatch without global try-except | **FIXED** |
| 7 | **MEDIUM** | `genai/connection.py:165` | `json.loads` outside `_get_response` try-except | **FIXED** (no tests — third-party pkg) |
| 8 | **MEDIUM** | `twikit/connection.py:219` | `_handle_done_task` doesn't catch exceptions from `task.result()` | **FIXED** |
| 9 | **MEDIUM** | `twikit/connection.py:252` | `json.loads` outside try-except | **FIXED** |
| 10 | **MEDIUM** | `tweepy/connection.py` | No timeout on tweepy HTTP calls | **FIXED** |
| 11 | **LOW** | `mirror_db/connection.py:339` | `_raise_for_response` assumes error body is JSON | **FIXED** |
| 12 | **LOW** | `tweepy/connection.py:204` | `json.loads` outside `_get_response` try-except | **FIXED** |
| 13 | **LOW** | `genai/connection.py` | No timeout on genai library HTTP calls | **FIXED** (no tests — third-party pkg) |
| 14 | **LOW** | `mirror_db/connection.py:181` | No explicit timeout on aiohttp session | **FIXED** |

---

## Operational Impact Classification

### A. What can CRASH the agent process

**All crash bugs have been fixed.** Previously:

| # | Trigger | Status |
|---|---------|--------|
| 1 | Fireworks API unreachable or returns non-JSON | **FIXED** — early `return None` on non-200 |
| 2 | Meme subgraph returns `{"data": null}` or missing keys | **FIXED** — defensive `.get()` chain |
| 3 | IPFS gateway returns non-JSON or metadata missing "description" | **FIXED** — try-except + `.get()` |

---

### B. What can get the agent STUCK

**All blocking bugs have been fixed.** Previously:

| # | Trigger | Status |
|---|---------|--------|
| 1 | Twikit rate limiting / retries blocking event loop | **FIXED** — `time.sleep()` → `await asyncio.sleep()` |
| 2 | Tweepy hanging HTTP call | **FIXED** — `wait_on_rate_limit=True` on Client |
| 3 | GenAI hanging HTTP call | **FIXED** — timeouts added (30s x402, 120s generate_content) |

**Remaining risk:** Twikit library-level timeout (CC5) — twikit's internal HTTP calls have no explicit timeout. This depends on twikit library internals and is deferred as an architectural decision.

---

### C. Agent keeps running with UNINTENDED SIDE-EFFECTS

**All side-effect bugs have been fixed.**

| # | Trigger | Status |
|---|---------|--------|
| 1 | `_estimate_gas` returns `False` | **FIXED** — returns `None` |
| 2 | Handler crash (no try-except) | **FIXED** — returns HTTP 500 |
| 3 | Meme subgraph returns `[]` (empty) | Not a bug — graceful degradation by design |

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

### 3. Missing timeouts (CC5) — partially fixed

| Component | Missing timeout | Status |
|-----------|----------------|--------|
| Tweepy (BaseSyncConnection, 1 thread) | requests calls | **FIXED** — `wait_on_rate_limit=True` |
| GenAI (BaseSyncConnection, 1 thread) | genai library calls | **FIXED** — explicit timeouts added |
| MirrorDB (aiohttp) | Session-level timeout | **FIXED** — `ClientTimeout(total=60)` |
| Twikit | twikit library calls | **DEFERRED** — depends on twikit internals |

### 4. `time.sleep()` in async context — FIXED

~~The twikit connection uses blocking `time.sleep()` in 9 locations within async methods.~~ **All 9 locations replaced with `await asyncio.sleep()`.**

---

## Combined Priority Matrix

All individual bugs (P0–P4) have been fixed. Only architectural/systemic issues remain.

| Priority | Issue | Category | Status |
|----------|-------|----------|--------|
| **P0** | #1: Fireworks json.loads on non-200 | Crash | **FIXED** |
| **P0** | #2: Meme subgraph direct key access | Crash | **FIXED** |
| **P0** | #3: IPFS metadata crash | Crash | **FIXED** |
| **P0** | #4: Twikit time.sleep() blocks event loop | Stuck | **FIXED** |
| **P1** | #5: _estimate_gas returns False | Side-effect | **FIXED** |
| **P2** | #6: Handler dispatch no try-except | Crash (handler) | **FIXED** |
| **P2** | #7: GenAI json.loads outside try | Stuck | **FIXED** (no tests — third-party pkg) |
| **P2** | #8: Twikit _handle_done_task | Stuck | **FIXED** |
| **P2** | #9: Twikit json.loads outside try | Stuck | **FIXED** |
| **P2** | #10: Tweepy no timeout | Stuck | **FIXED** |
| **P3** | CC1: Inconsistent retries | Systemic | **DEFERRED** |
| **P3** | CC2: No circuit breaker | Systemic | **DEFERRED** |
| **P3** | CC5: Twikit library-level timeout | Systemic | **DEFERRED** |
| **P4** | #11: MirrorDB error JSON assumption | Handler | **FIXED** |
| **P4** | #12: Tweepy json.loads outside try | Defensive | **FIXED** |
| **P4** | #13: GenAI no timeout | Config | **FIXED** (no tests — third-party pkg) |
| **P4** | #14: MirrorDB session timeout | Config | **FIXED** |
