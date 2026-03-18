# AI Mail Sentry — SLM-based Spam Filter

A lightweight, privacy-focused spam filter that uses a Small Language Model (SLM) and
vector memory (RAG) to classify emails based on your personal mail history.

All processing happens **locally** — no data ever leaves your machine.

## Overview

Unlike rule-based filters (Rspamd, SpamAssassin) that rely on global signatures, this
project uses **SLM** (for example `qwen2.5:3b` via Ollama) to understand the *context and intent* of
emails in any language. It learns from **both sides**:

- **What you don't want** — your Junk folder (spam examples)
- **What you do want** — your Sent folder and other trusted folders (HAM examples)

Teaching the model both sides dramatically reduces false positives compared to
spam-only training.

### How it works

```
Junk folder (last 300 emails)          Sent / Archive (last 200 emails/folder)
        |                                              |
        v                                              v
  ChromaDB: junk_folder_patterns    ChromaDB: ham_folder_patterns
        |                                              |
        +---------------+---------------+
                        |
              top-5 spam + top-5 ham examples
                        |
                        v
             qwen2.5:3b (via Ollama)
             classifies each unseen INBOX email
                        |
               SPAM? move to Junk
               HAM?  leave unread in INBOX
```

The vector DB is updated **incrementally** — only newly arrived emails are fetched
from IMAP on each cron run, so startup overhead stays minimal even with large folders.

## Prerequisites

| Component | Version |
|-----------|---------|
| OS | Debian Bookworm (or any Linux) |
| RAM / CPU | 4 GB RAM min, 8 GB recommended; 4+ CPU threads |
| Disk | 500MB for Python venv and ollama + 2-5 GB for SLM |
| Ollama | latest |
| Python | 3.11+ |

## Installation

**1. Install Ollama and pull the model:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

> **Alternative models** (see [Model selection](#model-selection) below for details):
> ```bash
> ollama pull qwen2.5:7b   # better Czech/multilingual, ~5 GB RAM
> ollama pull gemma2:2b    # fastest, good classification quality
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
$EDITOR .env   # fill in IMAP_SERVER, EMAIL_USER, EMAIL_PASS, HAM_FOLDERS
```

**5. Build the vector DB for the first time:**

```bash
source venv/bin/activate
python spam_filter.py --rebuild-db
```

## Configuration

All settings live in `.env`. The only required values are the three IMAP credentials.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAP_SERVER` | *(required)* | Hostname of your IMAP server |
| `EMAIL_USER` | *(required)* | Your email address |
| `EMAIL_PASS` | *(required)* | Your IMAP password or app password |
| `INBOX_FOLDER` | `INBOX` | Folder to scan for new mail |
| `JUNK_FOLDER` | `Junk` | Folder used as spam training data |
| `HAM_FOLDERS` | `Sent` | Comma-separated folders used as HAM training data |
| `MODEL_NAME` | `qwen2.5:3b` | Any model available in Ollama |
| `DB_PATH` | `./spam_memory` | Path for the ChromaDB vector store |
| `MAX_JUNK_EMAILS` | `300` | How many recent junk emails to keep in memory |
| `MAX_HAM_EMAILS` | `200` | How many recent HAM emails to keep per folder |
| `MAX_EMAIL_CHARS` | `1000` | Max characters read from each email body |
| `SIMILAR_RESULTS` | `5` | Examples of each type passed to the LLM per classification |

### HAM_FOLDERS tips

The more relevant your HAM folders, the fewer false positives:

```
HAM_FOLDERS=Sent                        # single folder
HAM_FOLDERS=Sent,Archive                # multiple folders
HAM_FOLDERS=Sent Items,INBOX.Archive    # names with spaces or IMAP hierarchy
```

**Sent** is the most powerful choice — every domain and contact you reply to
becomes near-impossible to false-positive on.

## Running

**Normal run** (incremental — only new emails are fetched from IMAP):

```bash
source venv/bin/activate
python spam_filter.py
```

**Full DB rebuild** (re-indexes everything from scratch):

```bash
python spam_filter.py --rebuild-db
```

Use `--rebuild-db` when:
- Setting up for the first time
- Changing `HAM_FOLDERS` or significantly increasing `MAX_*_EMAILS`
- Purging stale entries after bulk-deleting old junk

Output is written both to the console and to syslog (`journalctl -t mail-filter`).

## Automation (cron)

```
# Classify new mail every 15 minutes (incremental DB sync)
# `timeout 720` kills the process after 12 minutes — before the next tick fires.
# This prevents process pile-up if Ollama or IMAP stalls.
# The built-in lock (spam_filter.lock) also ensures only one instance runs at a time.
*/15 * * * * timeout 720 /path/to/spam-filter-by-slm/venv/bin/python /path/to/spam-filter-by-slm/spam_filter.py > /dev/null 2>&1

# Full DB rebuild every Sunday at 02:00 (picks up bulk-deleted junk, refreshes HAM)
# Allow up to 3 hours for a full rebuild.
0 2 * * 0  timeout 10800 /path/to/spam-filter-by-slm/venv/bin/python /path/to/spam-filter-by-slm/spam_filter.py --rebuild-db > /dev/null 2>&1
```

## Model selection

| Model | RAM | Czech / multilingual | Speed | Notes |
|-------|-----|----------------------|-------|-------|
| `qwen2.5:3b` | ~2 GB | ★★★★ | Fast | **Default — best small multilingual model** |
| `gemma2:2b` | ~1.6 GB | ★★★ | Fastest | Excellent binary classification |
| `llama3.2:3b` | ~2 GB | ★★★ | Fast | Previous default |
| `qwen2.5:7b` | ~4.7 GB | ★★★★★ | Medium | Best Czech quality; recommended if RAM allows |
| `llama3.1:8b` | ~5 GB | ★★★★ | Medium | Large context, good multilingual |

Change the model in `.env`:
```
MODEL_NAME=qwen2.5:7b
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
  folder manually. On the next run, the Junk/HAM memory updates automatically via
  the incremental sync.
- Credentials are read from `.env`, which is excluded from git via `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
