"""Export Coco chat sessions as collaboration traces.

Usage:
    python export_coco_traces.py --list
    python export_coco_traces.py --sessions <id> <id> [-o traces/coco_traces.json]

Session ids can be shortened to any unique prefix shown by --list.
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

MIN_USER_TURNS = 5


def default_store_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.environ["APPDATA"])
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "coco" / "chat-conversations.json"


def iso(ms):
    return datetime.fromtimestamp(ms / 1000).isoformat(timespec="seconds") if ms else None


def is_user_turn(message) -> bool:
    return message.get("role") == "user" and message.get("text", "").strip() != ""


def num_user_turns(conversation) -> int:
    return sum(is_user_turn(m) for m in conversation.get("messages", []))


def tutor_model(conversation):
    for m in conversation.get("messages", []):
        model = (m.get("tutorMetrics") or {}).get("model")
        if model:
            return model
    return None


def export_turn(message, include_screen_context: bool):
    if message.get("role") == "user":
        turn = {"role": "user", "text": message.get("text", "")}
        if message.get("images"):
            turn["num_images_omitted"] = len(message["images"])
        if include_screen_context and message.get("screenContext"):
            turn["screen_context"] = message["screenContext"]
    else:
        turn = {"role": "assistant", "text": message.get("text", "")}
        tool_calls = [
            {"name": a["call"].get("name"), "arguments": a["call"].get("arguments")}
            for a in message.get("activity") or []
            if a.get("kind") == "tool" and a.get("call")
        ]
        if tool_calls:
            turn["tool_calls"] = tool_calls
        if message.get("suggestion"):
            turn["proactive_suggestion"] = message["suggestion"]
        if message.get("stopped"):
            turn["stopped_by_user"] = True
    if message.get("ts"):
        turn["timestamp"] = iso(message["ts"])
    return turn


def export_conversation(conversation, include_screen_context: bool):
    turns = [
        export_turn(m, include_screen_context)
        for m in conversation.get("messages", [])
        # Failed or unfinished replies are UI retry state, not part of the collaboration.
        if not m.get("isError") and not m.get("isStreaming")
    ]
    return {
        "session_id": conversation["sessionId"],
        "title": conversation.get("title"),
        "problem": conversation.get("problem"),
        "model": tutor_model(conversation),
        "created_at": iso(conversation.get("createdAt")),
        "num_user_turns": sum(t["role"] == "user" for t in turns),
        "turns": turns,
    }


def resolve(conversations, prefix):
    matches = [c for c in conversations if c["sessionId"].startswith(prefix)]
    if len(matches) != 1:
        problem = "No session" if not matches else f"{len(matches)} sessions"
        sys.exit(f"{problem} match id '{prefix}'. Run with --list to see session ids.")
    return matches[0]


def list_conversations(conversations):
    print(f"{'session id':<10}  {'created':<19}  {'user turns':>10}  title")
    for c in sorted(conversations, key=lambda c: c.get("createdAt", 0), reverse=True):
        turns = num_user_turns(c)
        flag = "" if turns >= MIN_USER_TURNS else "  (too short)"
        print(
            f"{c['sessionId'][:8]:<10}  {iso(c.get('createdAt')) or '':<19}  {turns:>10}  "
            f"{(c.get('title') or '')[:60]}{flag}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list saved Coco sessions and exit")
    parser.add_argument("--sessions", nargs="+", metavar="ID", help="session ids (or unique prefixes) to export")
    parser.add_argument("-o", "--output", default="traces/coco_traces.json", help="output JSON file")
    parser.add_argument("--store", type=Path, default=default_store_path(), help="path to Coco's chat-conversations.json")
    parser.add_argument(
        "--include-screen-context",
        action="store_true",
        help="also export screen observations attached to user turns (only present if screen sensing was on)",
    )
    args = parser.parse_args()

    if not args.store.exists():
        sys.exit(f"Could not find {args.store}. Have you chatted with Coco yet? Use --store to point at the file.")
    with open(args.store) as f:
        conversations = json.load(f)

    if args.list or not args.sessions:
        list_conversations(conversations)
        if not args.list:
            print("\nPick sessions to export with --sessions <id> <id>.")
        return

    traces = []
    for prefix in args.sessions:
        trace = export_conversation(resolve(conversations, prefix), args.include_screen_context)
        if trace["num_user_turns"] < MIN_USER_TURNS:
            print(
                f"Warning: session {trace['session_id'][:8]} has {trace['num_user_turns']} user turns; "
                f"the assignment requires at least {MIN_USER_TURNS}."
            )
        traces.append(trace)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(traces)} trace(s) to {output}")


if __name__ == "__main__":
    main()
