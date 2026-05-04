# How the benchmark calls the LLM (and multiego / 403 troubleshooting)

## Call chain

1. **`main.py` startup** reads the project-root **`.env`** (`_load_env_file`): a key is only written when it is **not** already in the process environment (it does not overwrite values you have already `export`-ed in your shell).

2. **API endpoint and key** (around lines 49–60):
   - `API_KEY` ← falls back to **`ANNOTATE_API_KEY`** if empty.
   - `API_BASE` ← falls back to **`ANNOTATE_API_BASE`** if empty; **if both are empty the code no longer falls back to any default URL**, so you must configure it explicitly in `.env` or the environment before running.
   - `MODEL_NAME` ← defaults to `gpt-5-mini`.
   - `TIMEOUT` ← defaults to 60 seconds when empty.

3. **Sending the request**: `OpenAICompatibleLLM.generate()` (around line 105+).
   - URL: `{API_BASE}/chat/completions` (`API_BASE` has trailing `/` stripped).
   - Method: `POST`.
   - Body: OpenAI-compatible JSON (`model`, `messages` = system role-prompt + user = `build_prompt(...)`, `temperature`: 0.7).
   - Headers: `Content-Type`, `Accept`, `Authorization: Bearer …`, plus **`User-Agent`** (see `openai_compat_http.py`).

4. **Implementation detail**: uses the stdlib **`urllib.request`**, **not** `requests` or the official OpenAI SDK.

## Why multiego sometimes returns 403 (error code 1010)

Common reasons (verify against multiego ops / docs):

| Possible cause | Notes |
|----------------|-------|
| **Default User-Agent** | The old code only sent `Python-urllib/3.x`; some CDNs (Cloudflare) reject this with **403** outright. The default is now a **browser-style User-Agent**, and you can still override via **`LLM_USER_AGENT`**. |
| **Egress IP** | Data-centre / overseas / non-campus IPs are blocked; "the same key works from my laptop but fails from CI / cloud" usually falls in this bucket. |
| **Key or permission** | Invalid key or no access to the model; some gateways return a generic 403 in either case. |
| **Extra headers / mTLS** | A handful of intranet gateways require `CF-Access-*`, `Referer`, etc.; if their docs demand it, extend `openai_compat_http.py` (or file a request). |

## Recommended setup (multiego only)

Write these explicitly in `.env` (so you never accidentally hit a different fallback):

```env
API_KEY=your_multiego_key
API_BASE=https://llm-sjtu.multiego.me/v1
MODEL_NAME=gpt-5.4-mini
# optional: custom UA
# LLM_USER_AGENT=Mozilla/5.0 ...
```

Local verification (same path and header style the benchmark uses):

```bash
curl -sS -o /dev/stderr -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" \
  -d '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
  "$API_BASE/chat/completions"
```

If `curl` also returns 403, the issue is unrelated to this repo — investigate **network / gateway policy / the key itself**.
