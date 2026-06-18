#!/usr/bin/env python3
"""
AI Mail Sentry — Hybrid SLM-based Spam Filter (v2.0)

Two-stage classification pipeline:
  Stage 1 (Fast): Pattern matching + ChromaDB distance analysis
                 → confident SPAM/HAM decisions without LLM calls
  
  Stage 2 (Fallback): Structured feature extraction + LLM reasoning
                     → used only for uncertain cases

Key improvements over v1.x:
- Removed HAM example retrieval (added noise, reduced accuracy)
- Added pattern-based spam detection (keywords, urgency, sender analysis)
- Uses ChromaDB distances as confidence signals, not just retrieval
- Falls back to LLM ONLY when Stage 1 cannot decide confidently
- Default model changed from qwen2.5:3b → gemma2:2b (better classification, less RAM)

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

DB_PATH          = os.getenv("DB_PATH", "./spam_memory")
MODEL_NAME       = os.getenv("MODEL_NAME", "gemma2:2b")
MAX_JUNK_EMAILS  = int(os.getenv("MAX_JUNK_EMAILS", "300"))
MAX_EMAIL_CHARS  = int(os.getenv("MAX_EMAIL_CHARS", "1000"))
SIMILAR_RESULTS  = int(os.getenv("SIMILAR_RESULTS", "5"))

# Hybrid classifier thresholds (tuned for gemma2:2b on limited resources)
SPAM_DISTANCE_THRESHOLD = float(os.getenv("SPAM_DISTANCE_THRESHOLD", "0.6"))
HAM_DISTANCE_THRESHOLD  = float(os.getenv("HAM_DISTANCE_THRESHOLD", "0.4"))
CONFIDENCE_MARGIN       = float(os.getenv("CONFIDENCE_MARGIN", "0.15"))

# Pattern-based detection flags
ENABLE_SPAM_PATTERNS   = os.getenv("ENABLE_SPAM_PATTERNS", "true").lower() == "true"
SPAM_KEYWORD_WEIGHT    = float(os.getenv("SPAM_KEYWORD_WEIGHT", "0.3"))
SPAM_URGENCY_WEIGHT    = float(os.getenv("SPAM_URGENCY_WEIGHT", "0.25"))
SPAM_SENDER_WEIGHT     = float(os.getenv("SPAM_SENDER_WEIGHT", "0.25"))
SPAM_STRUCTURE_WEIGHT  = float(os.getenv("SPAM_STRUCTURE_WEIGHT", "0.2"))

# Active learning: record misclassified emails for training feedback
ENABLE_ACTIVE_LEARNING = os.getenv("ENABLE_ACTIVE_LEARNING", "true").lower() == "true"

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


# ---------------------------------------------------------------------------
# Hybrid Classification Functions
# ---------------------------------------------------------------------------

def pattern_based_spam_check(content: str, msg) -> dict:
    """Analyze email content for explicit spam indicators.

    Returns a score dictionary with component scores and overall confidence.
    This runs without LLM calls — fast and deterministic.
    """
    if not ENABLE_SPAM_PATTERNS:
        return {"score": 0.5, "details": {}, "reason": "patterns disabled"}

    spam_keywords = [
        # Financial urgency
        "verify your account", "confirm immediately", "urgent action required",
        "click here now", "limited time offer", "act fast",
        "you have been selected", "winner notification", "prize claim",
        # Suspicious requests
        "send verification code", "password reset needed", "update payment info",
        "wire transfer", "bitcoin", "cryptocurrency", "investment opportunity",
        # Common spam patterns
        "Dear valued customer", "Congratulations!", "Free gift",
        "no obligation", "risk free trial", "cancel anytime",
    ]

    urgency_keywords = [
        "URGENT", "IMMEDIATELY", "ASAP", "NOW", "EXPIRES",
        "WARNING", "ALERT", "NOTICE", "ACTION REQUIRED",
    ]

    suspicious_domains = [
        "@gmail.com", "@yahoo.com", "@hotmail.com",  # Free email for business
        "@tempmail.org", "@guerrillamail.com",
    ]

    # Score component 1: Spam keyword density
    content_lower = content.lower()
    keyword_matches = sum(1 for kw in spam_keywords if kw in content_lower)
    keyword_score = min(keyword_matches / 5.0, 1.0)  # Normalize to 0-1

    # Score component 2: Urgency language
    text_upper = content.upper()
    urgency_count = sum(1 for kw in urgency_keywords if kw in text_upper)
    urgency_score = min(urgency_count / 3.0, 1.0)

    # Score component 3: Sender analysis (if available from msg)
    sender = str(msg.get("From", "")) if hasattr(msg, "get") else ""
    sender_is_suspicious = any(domain in sender for domain in suspicious_domains)
    # Also check if sender doesn't match any known contact patterns
    sender_score = 0.5 if sender_is_suspicious else 0.1

    # Score component 4: Content structure anomalies
    # All caps ratio, excessive punctuation, etc.
    alpha_count = sum(1 for c in content if c.isalpha())
    all_caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
    exclamation_count = content.count("!")
    structure_score = min((all_caps_ratio * 2 + exclamation_count / 10), 1.0)

    # Weighted combination
    overall = (
        keyword_score * SPAM_KEYWORD_WEIGHT +
        urgency_score * SPAM_URGENCY_WEIGHT +
        sender_score * SPAM_SENDER_WEIGHT +
        structure_score * SPAM_STRUCTURE_WEIGHT
    )

    return {
        "score": overall,
        "details": {
            "keyword_matches": keyword_matches,
            "urgency_count": urgency_count,
            "sender_suspicious": sender_is_suspicious,
            "all_caps_ratio": round(all_caps_ratio, 3),
            "exclamation_count": exclamation_count,
        },
        "reason": f"keywords={keyword_matches}, urgency={urgency_count}",
    }


def analyze_chromadb_distances(spam_results) -> dict:
    """Analyze ChromaDB query distances to determine confidence level.

    Returns distance metrics and whether we should use LLM or skip it.
    """
    if not spam_results["distances"][0]:
        return {"avg_distance": float('inf'), "min_distance": 0, "max_distance": 0}

    distances = spam_results["distances"][0]
    avg_dist = sum(distances) / len(distances)
    min_dist = min(distances)
    max_dist = max(distances)

    # Distance interpretation:
    # - Low distance to spam = confident SPAM (no LLM needed)
    # - High distance from everything = uncertain (LLM recommended)
    is_confident_spam = avg_dist < SPAM_DISTANCE_THRESHOLD and min_dist < HAM_DISTANCE_THRESHOLD
    is_confident_ham  = avg_dist > (1.0 - HAM_DISTANCE_THRESHOLD)

    return {
        "avg_distance": avg_dist,
        "min_distance": min_dist,
        "max_distance": max_dist,
        "is_confident_spam": is_confident_spam,
        "is_confident_ham": is_confident_ham,
    }


def classify_with_llm(content: str, msg, junk_collection, combined_score: float, distance_analysis: dict) -> tuple:
    """LLM fallback for uncertain cases with structured features.

    Only called when Stage 1 (pattern + distance) cannot make a confident decision.
    The prompt includes explicit feature values so the model doesn't need to reason
    from raw email text alone.
    """
    # Retrieve spam examples for context
    junk_count = junk_collection.count()
    n_spam = min(SIMILAR_RESULTS, junk_count)
    spam_results = junk_collection.query(query_texts=[content], n_results=n_spam)
    spam_examples = (
        "\n---\n".join(spam_results["documents"][0])
        if spam_results["documents"][0]
        else "(no spam examples available)"
    )

    # Extract sender for the prompt
    sender = str(msg.get("From", "Unknown")) if hasattr(msg, "get") else "Unknown"

    prompt = f"""You are an expert spam classifier. Analyze this email using structured criteria.

RECENT SPAM EXAMPLES (from user's junk folder):
{spam_examples}

CLASSIFICATION CRITERIA:
1. Sender authenticity — is the sender who they claim to be?
2. Urgency pressure — does it demand immediate action?
3. Content legitimacy — is there genuine value or misleading promises?
4. Pattern matching — does it follow known spam templates?

STRUCTURED INPUT (extracted features):
- Combined confidence score: {combined_score:.3f}
- ChromaDB avg distance to spam: {distance_analysis['avg_distance']:.3f}
- Is confident SPAM by distance: {distance_analysis['is_confident_spam']}

EMAIL TO CLASSIFY:
From: {sender}
Content: {content}

Reply with ONLY one word: SPAM or HAM — nothing else."""

    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    raw = response.get("response", "").strip()

    # Take only the first word to be robust against verbose model output
    first_word = raw.split()[0].upper() if raw else "HAM"
    verdict = "SPAM" if first_word == "SPAM" else "HAM"

    return verdict, {
        "confidence": "medium",
        "method": "llm_fallback",
        "score": round(combined_score, 3),
    }


def classify_email_hybrid(content: str, msg, junk_collection) -> tuple:
    """Two-stage hybrid classifier: pattern matching + distance analysis → LLM fallback.

    Stage 1 (fast): Pattern matching + ChromaDB distances for confident cases
    Stage 2 (slow): LLM reasoning only for uncertain cases

    Returns: (verdict, metadata) where metadata contains confidence info
    """
    # -----------------------------------------------------------------------
    # Stage 1: Fast path - pattern matching and distance analysis
    # -----------------------------------------------------------------------
    
    # Pattern-based scoring (no LLM needed)
    pattern_result = pattern_based_spam_check(content, msg)

    # Get ChromaDB distances to spam examples
    junk_count = junk_collection.count()
    if junk_count == 0:
        logger.warning("Junk memory is empty — defaulting to HAM.")
        return "HAM", {"confidence": "low", "method": "empty_db_default"}

    n_spam = min(SIMILAR_RESULTS, junk_count)
    spam_results = junk_collection.query(query_texts=[content], n_results=n_spam)
    
    # Analyze distance patterns
    distance_analysis = analyze_chromadb_distances(spam_results)

    # Combine pattern score with distance signal
    # Weighted: 40% patterns, 60% vector similarity (inverted since lower distance = more similar)
    combined_score = (
        pattern_result["score"] * 0.4 + 
        max(0, 1 - distance_analysis["avg_distance"]) * 0.6
    )

    # Decision logic based on confidence thresholds
    if combined_score > 0.7 and distance_analysis["is_confident_spam"]:
        return "SPAM", {
            "confidence": "high", 
            "method": "pattern+distance_fast",
            "score": round(combined_score, 3),
        }

    if combined_score < 0.3 or distance_analysis["is_confident_ham"]:
        return "HAM", {
            "confidence": "high", 
            "method": "pattern+distance_fast",
            "score": round(combined_score, 3),
        }

    # -----------------------------------------------------------------------
    # Stage 2: LLM fallback for uncertain cases with structured features
    # -----------------------------------------------------------------------
    
    logger.info(f"Stage 2: Using LLM for uncertain case (score={combined_score:.2f})")
    
    return classify_with_llm(content, msg, junk_collection, combined_score, distance_analysis)


def classify_and_move(client: IMAPClient, junk_collection) -> None:
    """Scan INBOX for unseen messages, classify each, and move spam to Junk.

    Uses the hybrid classifier which combines pattern matching + vector distances
    with LLM fallback only for uncertain cases. Only the Junk folder is used as
    reference data — HAM folders are no longer synced or consulted.
    """
    client.select_folder(INBOX_FOLDER)
    uids = client.search(["UNSEEN"])

    if not uids:
        logger.info("No new emails to scan.")
        return

    # Track classification statistics
    stats = {"spam": 0, "ham": 0, "llm_fallback": 0, "fast_path": 0}

    logger.info(f"Scanning {len(uids)} unseen message(s)...")
    fetch_data = client.fetch(uids, ["BODY.PEEK[]"])

    for uid, data in fetch_data.items():
        try:
            raw_email = data[b"BODY[]"]
            msg = email.message_from_bytes(raw_email, policy=default)
            subject = str(msg.get("Subject", "(No Subject)"))
            content = build_content(msg)

            # Use hybrid classifier directly (pass None for msg since we have it)
            verdict, metadata = classify_email_hybrid(content, msg, junk_collection)

            stats[verdict.lower()] += 1
            if metadata["method"] == "llm_fallback":
                stats["llm_fallback"] += 1
            else:
                stats["fast_path"] += 1

            if verdict == "SPAM":
                logger.info(f"[SPAM] Moving to Junk: {subject!r} (method={metadata['method']}, score={metadata.get('score', '?')})")
                client.move([uid], JUNK_FOLDER)
            else:
                logger.info(f"[HAM]  Leaving in inbox: {subject!r}")

        except Exception as exc:
            logger.error(f"Error processing UID {uid}: {exc}")

    # Log summary statistics
    total = stats["spam"] + stats["ham"]
    if total > 0:
        llm_pct = (stats["llm_fallback"] / total * 100) if total else 0
        logger.info(f"Classification summary: {total} emails — "
                   f"{stats['spam']} spam, {stats['ham']} ham, "
                   f"{stats['llm_fallback']} LLM fallbacks ({llm_pct:.0f}%)")


# ---------------------------------------------------------------------------
# Startup validation


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
            "then classify new mail. Use after significantly increasing MAX_*_EMAILS "
            "or to purge stale entries."
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

            # Classify incoming mail
            classify_and_move(client, junk_collection)

    except Exception as exc:
        logger.error(f"IMAP connection error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
