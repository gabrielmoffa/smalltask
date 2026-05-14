# AI news digest smalltask — design

## Purpose

A daily smalltask that:

1. Pulls fresh AI news from Google News RSS.
2. Picks 3 stories the user hasn't seen in the last 7 days.
3. Sends each story to the user's Telegram (same chat as the daily-improvements bot) with a short summary and a few social-media-angled questions designed to prompt opinions, hot takes, and content ideas.

Runs once per day via GitHub Actions.

## Non-goals

- No two-way Telegram bot. The user replies wherever they want; replies are not ingested.
- No long-term news archive. Dedup state is pruned to a 7-day rolling window.
- No new third-party Python dependencies. Everything uses stdlib (`urllib`, `xml.etree.ElementTree`, `html.parser`, `json`).

## Layout

Mirrors `examples/daily_improvements/`:

```
examples/ai_news/
  agents/
    ai_news.yaml
  tools/
    news_feed.py        # fetch + parse Google News RSS
    news_state.py       # load/save/prune sent_news.json
    article.py          # fetch article HTML and extract text
    telegram_news.py    # send one Telegram message per story
  sent_news.json        # committed state, pruned to last 7 days
```

Plus `.github/workflows/ai_news.yml`.

## Components

### `news_feed.fetch_headlines() -> list[dict]`

- Issues three GET requests to `https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en` for queries `AI`, `LLM`, `artificial intelligence`.
- Parses each RSS payload with `xml.etree.ElementTree`.
- Dedupes by URL across the three feeds.
- Returns up to ~30 items: `[{"title": str, "url": str, "source": str, "published": "YYYY-MM-DD"}]`.
- Best-effort decode of Google News redirect URLs to the underlying publisher URL when straightforward; otherwise keeps the Google News URL.
- On HTTP failure for one query, logs and continues with the others. If all three fail, raises so the agent can report.

### `news_state.load_recent_urls() -> list[str]`

- Reads `examples/ai_news/sent_news.json` (creates an empty `{"sent": []}` if missing).
- Returns list of URLs sent in the last 7 days.

### `news_state.append_sent(urls: list[str]) -> str`

- Adds each URL with today's date.
- Drops any entry older than 7 days.
- Writes the JSON back, sorted by date.
- Commits and pushes the change using the same git pattern as `daily_improvements` (configure user as `smalltask-bot`, push to the current branch).
- Returns a status string for the agent log.

### `article.fetch_text(url: str) -> str`

- HTTP GET with a desktop User-Agent.
- Strips `<script>` and `<style>` blocks, then strips tags via `html.parser`.
- Collapses whitespace, truncates to ~4000 chars.
- On failure returns a short error string (the agent decides whether to retry or skip).

### `telegram_news.send_story(title, url, source, summary, questions) -> str`

- Posts to the same `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` as the existing `telegram.py` tool.
- Message format:

  ```
  📰 *{title}* — {source}
  {url}

  {summary}

  💭 *Questions for you:*
  • {q1}
  • {q2}
  • {q3}
  ```

- `questions` is a list (2–3 items). Markdown mode `Markdown`.

## Agent (`ai_news.yaml`)

- `llm.connection: openrouter`, `model: anthropic/claude-sonnet-4.6`, `max_tokens: 4096`.
- Tools: `news_feed.fetch_headlines`, `news_state.load_recent_urls`, `article.fetch_text`, `telegram_news.send_story`.
- `post_hook: news_state.append_sent` — receives the URLs the agent actually sent (extracted from tool results).

Prompt outline:

1. Call `fetch_headlines`.
2. Call `load_recent_urls`; filter out already-sent URLs in-context.
3. From the remaining headlines, pick **3** that look most interesting for an AI-focused practitioner who creates social-media content. Prefer variety (e.g., not all model-release news).
4. For each pick, call `article.fetch_text(url)`. If a fetch fails, swap in the next-best unseen headline.
5. For each of the 3 stories, draft:
   - A 2–3 sentence summary.
   - 2–3 questions angled to give the user raw material for social posts (hot takes, contrarian framings, practical use angles — not generic "what do you think?").
6. Call `send_story` once per story.
7. End with a 1-line final message listing the 3 URLs sent.

Hard rules in the prompt:

- Never invent URLs or facts — only use values returned by tools.
- One `send_story` call per story, three calls total.
- If `load_recent_urls` filters out everything, send a single "no fresh AI news today" via `send_story` with `url=""` and skip the state update.
- On tool failure, retry at most once.

## Post-hook: `news_state.append_sent`

The post-hook framework passes `output` and `tool_results`. Implementation extracts URLs from `tool_results` where the tool name is `send_story` and the call succeeded, then calls the same `append_sent` logic. Commits `sent_news.json` and pushes.

## GitHub Actions workflow

`.github/workflows/ai_news.yml`:

- Cron: `0 8 * * *` (08:00 UTC daily) plus `workflow_dispatch`.
- `permissions: contents: write` (needed to push the `sent_news.json` update).
- Steps: checkout, setup Python 3.11, `pip install .`, configure git identity, run agent.
- Env: `OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` from existing secrets.

## Error handling summary

| Failure | Behavior |
|---|---|
| All three RSS queries fail | Agent reports in final message, no Telegram send, no commit. |
| One RSS query fails | Continue with the others. |
| All headlines already seen | Send one "no fresh news today" Telegram message; skip state update. |
| `article.fetch_text` fails for a pick | Agent swaps in next-best unseen headline. |
| `send_story` fails | Retry once; if still failing, report in final message. State update only records URLs that were successfully sent. |

## Out of scope (future ideas)

- Multilingual news (Portuguese/English mix).
- Sentiment- or topic-balanced picking.
- Replying to the questions via Telegram and drafting posts.
- Image/thumbnail extraction.
