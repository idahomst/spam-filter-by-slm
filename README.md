# AI Mail Sentry — Hybrid SLM-based Spam Filter (v2.0)

A lightweight, privacy-focused spam filter that uses a two-stage hybrid approach combining pattern matching and ChromaDB vector analysis with LLM fallback for uncertain cases.

All processing happens **locally** — no data ever leaves your machine.

## Overview

Unlike rule-based filters (Rspamd, SpamAssassin) that rely on global signatures, this project uses a **hybrid classification pipeline**:

- **Stage 1 (Fast Path):** Pattern matching + ChromaDB distance analysis for confident decisions
- **Stage 2 (LLM Fallback):** Structured feature extraction + SLM reasoning only for uncertain cases

This approach dramatically reduces LLM call volume (~60-70% fewer calls) while improving accuracy by using explicit signals instead of relying on the SLM to guess from raw email text alone.

### How it works

```
Unseen INBOX emails
         |
         v
  Stage 1: Pattern + Distance Analysis ───┐
    ├── Keyword/urgency detection          │
    ├── ChromaDB distance to spam examples │ → High confidence? Fast return
    └── Combined scoring                   │
                         │                 │
                    Uncertain?              │
                         │                  │
                         v                  |
  Stage 2: LLM Fallback (structured features)
         |
    SPAM? move to Junk
    HAM?  leave unread in INBOX
```

**Key design decisions:**
- **Junk folder only:** Spam examples from your junk folder are used as reference patterns. HAM examples were removed because they added noise without meaningful classification signal on small models.
- **Pattern detection enabled by default:** Explicit keyword, urgency, sender domain, and structure analysis runs instantly without LLM overhead.
- **ChromaDB distances as features:** Vector similarity isn't just for retrieval — it's a confidence signal for the decision pipeline.

The vector DB is updated **incrementally** — only newly arrived emails are fetched from IMAP on each cron run, so startup overhead stays minimal even with large folders.

## Prerequisites

| Component | Version |
|-----------|---------|
| OS | Debian Bookworm (or any Linux / macOS) |
| RAM | 3 GB min (gemma2:2b), 4-8 GB recommended; 4+ CPU threads |
| Disk | 500MB for Python venv and ollama + 1.6 GB for gemma2:2b model |
| Ollama | latest |
| Python | 3.11+ |

## Installation

**1. Install Ollama and pull the model:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b
```

> **Recommended default:** `gemma2:2b` — best binary classification quality, fastest inference, uses ~1.6 GB RAM
> 
> **Alternative models** (see [Model selection](#model-selection) below for details):
> ```bash
> ollama pull qwen2.5:3b   # better Czech/multilingual support, ~2-3 GB RAM
> ollama pull llama3.2:3b  # good general-purpose option, ~2 GB RAM
> ```

**2. Clone the repository:**

```bash
git clone https://github.com/idahomst/spam-filter-by-slm.git
cd spam-filter-by-slm
```

**3. Create a virtual environment and install dependencies:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Configure credentials:**

```bash
cp .env.example .env
$EDITOR .env   # fill in IMAP_SERVER, EMAIL_USER, EMAIL_PASS
```

**5. Build the vector DB for the first time:**

```bash
source venv/bin/activate
python spam_filter.py --rebuild-db
```

## Configuration

All settings live in `.env`. The only required values are the three IMAP credentials.

### Required Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAP_SERVER` | *(required)* | Hostname of your IMAP server |
| `EMAIL_USER` | *(required)* | Your email address |
| `EMAIL_PASS` | *(required)* | Your IMAP password or app password |

### Basic Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INBOX_FOLDER` | `INBOX` | Folder to scan for new mail |
| `JUNK_FOLDER` | `Junk` | Folder used as spam reference data in ChromaDB |
| `MODEL_NAME` | `gemma2:2b` | Ollama model (must be pulled first) |
| `DB_PATH` | `./spam_memory` | Path for the ChromaDB vector store |
| `MAX_EMAIL_CHARS` | `1000` | Max characters read from each email body |
| `SIMILAR_RESULTS` | `5` | Spam examples retrieved for distance analysis |

### Indexing Settings (Incremental DB)

These control how many emails are indexed per folder. HAM folders are **indexed** but no longer used for classification — they exist only to keep the index up-to-date if you want them in future versions.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_JUNK_EMAILS` | `300` | How many recent junk emails to keep in memory |
| `HAM_FOLDERS` | `Sent` | Comma-separated folders indexed for completeness |
| `MAX_HAM_EMAILS` | `200` | How many HAM emails to index per folder (not used in classification) |

### Hybrid Classifier Thresholds

Fine-tune the two-stage pipeline. These values are tuned for gemma2:2b on limited resources.

| Variable | Default | Description |
|----------|---------|-------------|
| `SPAM_DISTANCE_THRESHOLD` | `0.6` | Avg ChromaDB distance below = confident SPAM |
| `HAM_DISTANCE_THRESHOLD` | `0.4` | Min distance threshold for confident decisions |
| `CONFIDENCE_MARGIN` | `0.15` | Gap between decision thresholds |

### Pattern-Based Detection

Explicit keyword and structure analysis that runs without LLM calls.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SPAM_PATTERNS` | `true` | Enable pattern-based detection (recommended) |
| `SPAM_KEYWORD_WEIGHT` | `0.3` | Weight for spam keyword matches |
| `SPAM_URGENCY_WEIGHT` | `0.25` | Weight for urgency language detection |
| `SPAM_SENDER_WEIGHT` | `0.25` | Weight for sender domain analysis |
| `SPAM_STRUCTURE_WEIGHT` | `0.2` | Weight for content structure anomalies |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ACTIVE_LEARNING` | `true` | Enable active learning feedback loop (future feature) |

---

## Running

**Normal run** (incremental — only new emails are fetched from IMAP):

```bash
source venv/bin/activate
python spam_filter.py
```

Output includes a classification summary at the end:
```
Classification summary: 12 emails — 3 spam, 9 ham, 4 LLM fallbacks (33%)
```

**Full DB rebuild** (re-indexes everything from scratch):

```bash
python spam_filter.py --rebuild-db
```

Use `--rebuild-db` when:
- Setting up for the first time
- Significantly increasing `MAX_*_EMAILS`
- Purging stale entries after bulk-deleting old junk

Output is written both to the console and to syslog (`journalctl -t mail-filter`).

## Automation (cron)

```
# Classify new mail every 15 minutes (incremental DB sync)
# `timeout 720` kills the process after 12 minutes — before the next tick fires.
# This prevents process pile-up if Ollama or IMAP stalls.
# The built-in lock (spam_filter.lock) also ensures only one instance runs at a time.
*/15 * * * * timeout 720 /path/to/spam-filter-by-slm/venv/bin/python /path/to/spam-filter-by-slm/spam_filter.py > /dev/null 2>&1

# Full DB rebuild every Sunday at 02:00 (picks up bulk-deleted junk, refreshes index)
# Allow up to 3 hours for a full rebuild.
0 2 * * 0  timeout 10800 /path/to/spam-filter-by-slm/venv/bin/python /path/to/spam-filter-by-slm/spam_filter.py --rebuild-db > /dev/null 2>&1
```

## Model selection

| Model | RAM | Czech / multilingual | Speed | Notes |
|-------|-----|----------------------|-------|-------|
| `gemma2:2b` | ~1.6 GB | ★★★ | **Fastest** | **Default — best binary classification, low RAM** |
| `qwen2.5:3b` | ~2-3 GB | ★★★★ | Fast | Better multilingual (incl. Czech); needs more memory |
| `llama3.2:3b` | ~2 GB | ★★★ | Fast | General-purpose option; previous default |
| `qwen2.5:7b` | ~4.7 GB | ★★★★★ | Medium | Best accuracy; only if you have 8+ GB RAM free |

Change the model in `.env`:
```
MODEL_NAME=gemma2:2b
```

## CPU limit for Ollama

On a shared server, prevent Ollama from consuming all CPU threads:

```bash
sudo systemctl edit ollama.service
```

Add:

```ini
[Service]
# limit to 400% (4 threads)
CPUQuota=400%
# Lower the process priority
Nice=10
```

Then:

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

## Safety notes

- Only **UNSEEN** (unread) messages are processed.
- `BODY.PEEK[]` is used so the script never marks emails as read — your mail client
  only sees the result (spam disappearing from the inbox).
- If the model makes a mistake, move the misclassified email back to the correct
  folder manually. On the next run, the Junk memory updates automatically via
  the incremental sync.
- Classification statistics are logged after each batch — watch for high LLM fallback rates (>40%) which may indicate threshold tuning is needed.
- Credentials are read from `.env`, which is excluded from git via `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
