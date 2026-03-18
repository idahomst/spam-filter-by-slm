#!/usr/bin/env python3
"""
AI Mail Sentry — SLM-based Spam Filter
Classifies incoming emails using an SLM (e.g. qwen2.5:3b) + ChromaDB (RAG).

The filter learns both sides of the coin:
  • SPAM  — from the Junk folder (what you don't want)
  • HAM   — from Sent / Archive / any folder you trust (what you do want)

Usage:
    cp .env.example .env                # fill in your credentials
    python spam_filter.py               # classify new mail (incremental DB sync)
    python spam_filter.py --rebuild-db  # full DB rebuild, then classify
"""

import argparse
import email
import fcntl
import logging
import os
import sys
import tempfile
from email.policy import default
from logging.handlers import SysLogHandler

import chromadb
import ollama
from dotenv import load_dotenv
from imapclient import IMAPClient

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — all values can be overridden via environment variables
# ---------------------------------------------------------------------------
IMAP_SERVER   = os.environ.get("IMAP_SERVER", "")
EMAIL_USER    = os.environ.get("EMAIL_USER", "")
EMAIL_PASS    = os.environ.get("EMAIL_PASS", "")
INBOX_FOLDER  = os.getenv("INBOX_FOLDER", "INBOX")
JUNK_FOLDER   = os.getenv("JUNK_FOLDER", "Junk")

_ham_folders_raw = os.getenv("HAM_FOLDERS", "Sent")
HAM_FOLDERS = [f.strip() for f in _ham_folders_raw.split(",") if f.strip()]

DB_PATH          = os.getenv("DB_PATH", "./spam_memory")
MODEL_NAME       = os.getenv("MODEL_NAME", "qwen2.5:3b")
MAX_JUNK_EMAILS  = int(os.getenv("MAX_JUNK_EMAILS", "300"))
MAX_HAM_EMAILS   = int(os.getenv("MAX_HAM_EMAILS", "200"))
MAX_EMAIL_CHARS  = int(os.getenv("MAX_EMAIL_CHARS", "1000"))
SIMILAR_RESULTS  = int(os.getenv("SIMILAR_RESULTS", "5"))

# ---------------------------------------------------------------------------
# Logging — syslog for cron/systemd, console for manual runs
# ---------------------------------------------------------------------------
logger = logging.getLogger("mail-filter")
logger.setLevel(logging.INFO)

try:
    syslog_handler = SysLogHandler(address="/dev/log")
except OSError:
    syslog_handler = SysLogHandler()  # fallback: UDP localhost:514
syslog_handler.setFormatter(
    logging.Formatter("%(name)s [%(process)d]: %(levelname)s %(message)s")
)
logger.addHandler(syslog_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_text_from_msg(msg) -> str:
    """Extract plain text body from an email message, with HTML fallback."""
    parts_to_try = ("text/plain", "text/html")

    if msg.is_multipart():
        for mime_type in parts_to_try:
            for part in msg.walk():
                if part.get_content_type() == mime_type:
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(errors="ignore")[:MAX_EMAIL_CHARS]
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(errors="ignore")[:MAX_EMAIL_CHARS]

    return ""


def build_content(msg) -> str:
    """Build a combined subject + body string for embedding and classification."""
    subject = str(msg.get("Subject", "(No Subject)"))
    body = get_text_from_msg(msg)
    return f"Subject: {subject}\nBody: {body}"


def _doc_id(folder: str, uid: int) -> str:
    """Stable, collision-free ChromaDB document ID that includes the folder name.

    Different folders can have identical UID numbers; namespacing them prevents
    one folder's entries from overwriting another's.
    """
    safe_folder = folder.replace("/", "_").replace("\\", "_")
    return f"{safe_folder}/{uid}"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def sync_folder(
    client: IMAPClient,
    collection,
    folder: str,
    max_emails: int,
    label: str,
    full_rebuild: bool = False,
) -> None:
    """Index emails from *folder* into *collection*, incrementally or fully.

    Incremental mode (default):
        Only emails whose document IDs are not yet in ChromaDB are fetched
        from IMAP.  Normal cron runs touch IMAP only for genuinely new mail.

    Full-rebuild mode (--rebuild-db):
        All UIDs in the window are re-fetched and upserted.  Use this after
        wiping the collection or changing MAX_*_EMAILS significantly.
    """
    try:
        client.select_folder(folder, readonly=True)
    except Exception as exc:
        logger.warning(f"Cannot open folder {folder!r}: {exc}")
        return

    uids = client.search(["ALL"])
    if not uids:
        logger.info(f"{label}: folder is empty — nothing to index.")
        return

    recent_uids = uids[-max_emails:]
    doc_ids = [_doc_id(folder, u) for u in recent_uids]

    if full_rebuild:
        to_fetch_pairs = list(zip(recent_uids, doc_ids))
    else:
        existing = set(collection.get(ids=doc_ids)["ids"])
        to_fetch_pairs = [
            (uid, did) for uid, did in zip(recent_uids, doc_ids)
            if did not in existing
        ]

    if not to_fetch_pairs:
        logger.info(f"{label}: memory up to date — nothing new to index.")
        return

    fetch_uids = [uid for uid, _ in to_fetch_pairs]
    uid_to_docid = {uid: did for uid, did in to_fetch_pairs}

    logger.info(f"{label}: fetching {len(fetch_uids)} new email(s) from IMAP...")
    fetch_data = client.fetch(fetch_uids, ["RFC822"])

    indexed = 0
    for uid, data in fetch_data.items():
        doc_id = uid_to_docid.get(uid)
        if not doc_id:
            continue
        try:
            msg = email.message_from_bytes(data[b"RFC822"], policy=default)
            content = build_content(msg)
            collection.upsert(ids=[doc_id], documents=[content])
            indexed += 1
        except Exception as exc:
            logger.warning(f"Skipping UID {uid} from {folder!r}: {exc}")

    logger.info(f"{label}: indexed {indexed}/{len(fetch_uids)} email(s).")


def classify_email(content: str, junk_collection, ham_collection) -> str:
    """Return 'SPAM' or 'HAM' using both junk and ham examples as context.

    Providing HAM examples alongside SPAM examples gives the model a clear
    contrast, which dramatically reduces false positives on legitimate mail
    that superficially resembles spam (newsletters, Czech business email, etc.).
    """
    junk_count = junk_collection.count()

    if junk_count == 0:
        logger.warning("Junk memory is empty — defaulting to HAM.")
        return "HAM"

    # Retrieve the most similar SPAM examples
    n_spam = min(SIMILAR_RESULTS, junk_count)
    spam_results = junk_collection.query(query_texts=[content], n_results=n_spam)
    spam_examples = (
        "\n---\n".join(spam_results["documents"][0])
        if spam_results["documents"][0]
        else "(no spam examples available)"
    )

    # Retrieve the most similar HAM examples (optional — skipped if DB empty)
    ham_section = ""
    ham_count = ham_collection.count()
    if ham_count > 0:
        n_ham = min(SIMILAR_RESULTS, ham_count)
        ham_results = ham_collection.query(query_texts=[content], n_results=n_ham)
        ham_docs = ham_results["documents"][0] if ham_results["documents"][0] else []
        if ham_docs:
            ham_section = (
                "HAM examples (legitimate emails this user sends/receives):\n"
                + "\n---\n".join(ham_docs)
                + "\n\n---\n\n"
            )

    prompt = f"""You are a personal spam filter.

SPAM examples (emails previously marked as junk by this user):
{spam_examples}

---

{ham_section}Classify the following email.
Reply with ONLY one word: SPAM or HAM — nothing else.

EMAIL TO CLASSIFY:
{content}"""

    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    raw = response.get("response", "").strip()

    # Take only the first word to be robust against verbose model output
    first_word = raw.split()[0].upper() if raw else "HAM"
    return "SPAM" if first_word == "SPAM" else "HAM"


def classify_and_move(client: IMAPClient, junk_collection, ham_collection) -> None:
    """Scan INBOX for unseen messages, classify each, and move spam to Junk."""
    client.select_folder(INBOX_FOLDER)
    uids = client.search(["UNSEEN"])

    if not uids:
        logger.info("No new emails to scan.")
        return

    logger.info(f"Scanning {len(uids)} unseen message(s)...")
    fetch_data = client.fetch(uids, ["BODY.PEEK[]"])

    for uid, data in fetch_data.items():
        try:
            raw_email = data[b"BODY[]"]
            msg = email.message_from_bytes(raw_email, policy=default)
            subject = str(msg.get("Subject", "(No Subject)"))
            content = build_content(msg)

            verdict = classify_email(content, junk_collection, ham_collection)

            if verdict == "SPAM":
                logger.info(f"[SPAM] Moving to Junk: {subject!r}")
                client.move([uid], JUNK_FOLDER)
            else:
                logger.info(f"[HAM]  Leaving in inbox: {subject!r}")

        except Exception as exc:
            logger.error(f"Error processing UID {uid}: {exc}")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Exit early with a clear message if required config is missing."""
    missing = [
        var for var in ("IMAP_SERVER", "EMAIL_USER", "EMAIL_PASS")
        if not os.environ.get(var)
    ]
    if missing:
        for var in missing:
            logger.error(f"Missing required environment variable: {var}")
        logger.error("Copy .env.example to .env and fill in your credentials.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Prevent concurrent cron runs: acquire an exclusive lock before doing any work.
    # If another instance is already running, exit immediately and silently.
    lock_path = os.path.join(tempfile.gettempdir(), "spam_filter.lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info("Another instance is already running — exiting.")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="AI Mail Sentry — SLM-based spam filter"
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help=(
            "Wipe and fully rebuild the vector DB from all configured folders, "
            "then classify new mail. Use after changing HAM_FOLDERS, after "
            "significantly increasing MAX_*_EMAILS, or to purge stale entries."
        ),
    )
    args = parser.parse_args()

    validate_config()

    # Initialize ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)

        if args.rebuild_db:
            logger.info("--rebuild-db: wiping existing collections...")
            for name in ("junk_folder_patterns", "ham_folder_patterns"):
                try:
                    chroma_client.delete_collection(name)
                    logger.info(f"  Deleted collection: {name}")
                except Exception:
                    pass  # collection may not exist yet

        junk_collection = chroma_client.get_or_create_collection(
            name="junk_folder_patterns"
        )
        ham_collection = chroma_client.get_or_create_collection(
            name="ham_folder_patterns"
        )
    except Exception as exc:
        logger.error(f"Failed to initialize ChromaDB at {DB_PATH!r}: {exc}")
        sys.exit(1)

    # Connect to IMAP and run the filter
    try:
        with IMAPClient(IMAP_SERVER, use_uid=True) as client:
            try:
                client.login(EMAIL_USER, EMAIL_PASS)
            except Exception as exc:
                logger.error(f"IMAP login failed for {EMAIL_USER!r}: {exc}")
                sys.exit(1)

            full = args.rebuild_db

            # Sync spam examples
            sync_folder(
                client, junk_collection, JUNK_FOLDER, MAX_JUNK_EMAILS,
                "Junk", full_rebuild=full,
            )

            # Sync HAM examples from each configured folder
            for folder in HAM_FOLDERS:
                sync_folder(
                    client, ham_collection, folder, MAX_HAM_EMAILS,
                    f"HAM({folder})", full_rebuild=full,
                )

            # Classify incoming mail
            classify_and_move(client, junk_collection, ham_collection)

    except Exception as exc:
        logger.error(f"IMAP connection error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
