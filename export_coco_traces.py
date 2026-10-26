"""Export Coco chat sessions as collaboration traces.

Usage:
    python export_coco_traces.py --list
    python export_coco_traces.py --sessions <id> <id> [-o traces/coco_traces.json]
    python export_coco_traces.py --simulation-log traces/simulation_log.jsonl [-o traces/simulated_traces.json]

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
CARD_TOOL = "ask_user_question"


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


def num_user_turns(conversation) -> int:
    return sum(t["role"] == "user" for m in conversation.get("messages", []) for t in export_message(m))


def tutor_model(conversation):
    for m in conversation.get("messages", []):
        model = (m.get("tutorMetrics") or {}).get("model")
        if model:
            return model
    return None


def card_questions(call):
    return (call.get("result") or {}).get("questions") or (call.get("arguments") or {}).get("questions") or []


def card_answers(call):
    """The answers submitted in a question card, or None if it is still open or was skipped."""
    result = call.get("result") or {}
    if call.get("status") != "completed" or result.get("cancelled") or not result.get("answers"):
        return None
    return [
        {"question": a["question"], "answer": ", ".join([*(a.get("selected") or []), *([a["answer"]] if a.get("answer") else [])])}
        for a in sorted(result["answers"], key=lambda a: a["questionIndex"])
    ]


def card_status(call):
    if card_answers(call):
        return "answered"
    return "waiting" if call.get("status") == "running" else "skipped"


def format_question(question):
    options = "; ".join(
        o["label"] + (f" ({o['description']})" if o.get("description") else "") for o in question.get("options", [])
    )
    multi = " (pick any number)" if question.get("multiSelect") else ""
    return f"{question['question']}\nOptions{multi}: {options}"


def format_card(intro, questions):
    return "\n\n".join([*([intro] if intro else []), *(format_question(q) for q in questions)])


def export_message(message, include_screen_context: bool = False):
    """Export one stored message as a list of turns.

    A Coco reply that asks a question card becomes several turns: Coco's turn asking
    the questions, the user's turn answering them, and Coco's turn after the answers.
    """
    if message.get("role") == "user":
        if not message.get("text", "").strip():
            return []
        turn = {"role": "user", "text": message["text"]}
        if message.get("images"):
            turn["num_images_omitted"] = len(message["images"])
        if include_screen_context and message.get("screenContext"):
            turn["screen_context"] = message["screenContext"]
        turns = [turn]
    else:
        turns, narration, tool_calls = [], [], []
        for a in message.get("activity") or []:
            if a.get("kind") == "message":
                narration.append(a["text"])
            elif a.get("kind") == "tool" and a.get("call"):
                call = a["call"]
                if call.get("name") == CARD_TOOL and card_questions(call):
                    # Coco's messages before the card are what the user saw when answering it
                    intro, questions = "\n\n".join(narration), card_questions(call)
                    turn = {
                        "role": "assistant",
                        "text": format_card(intro, questions),
                        "question_card": {"intro": intro, "questions": questions, "status": card_status(call)},
                    }
                    if tool_calls:
                        turn["tool_calls"] = tool_calls
                    turns.append(turn)
                    answers = card_answers(call)
                    if answers:
                        turns.append(
                            {
                                "role": "user",
                                "text": "\n".join(f"{a['question']} {a['answer']}" for a in answers),
                                "question_card_answers": answers,
                            }
                        )
                    narration, tool_calls = [], []
                else:
                    tool_calls.append({"name": call.get("name"), "arguments": call.get("arguments")})
        turn = {"role": "assistant", "text": message.get("text", "")}
        if tool_calls:
            turn["tool_calls"] = tool_calls
        if message.get("suggestion"):
            turn["proactive_suggestion"] = message["suggestion"]
        if message.get("stopped"):
            turn["stopped_by_user"] = True
        if not turns or turn["text"].strip() or len(turn) > 2:
            turns.append(turn)
    if message.get("ts"):
        for turn in turns:
            turn["timestamp"] = iso(message["ts"])
    return turns


def export_conversation(conversation, include_screen_context: bool):
    turns = [
        turn
        for m in conversation.get("messages", [])
        # Failed or unfinished replies are UI retry state, not part of the collaboration.
        if not m.get("isError") and not m.get("isStreaming")
        for turn in export_message(m, include_screen_context)
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


def export_simulated(conversations, log_path, include_screen_context: bool):
    with open(log_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    by_id = {c["sessionId"]: c for c in conversations}
    traces = []
    for score in sorted((r for r in records if r["type"] == "session_score"), key=lambda r: r["index"]):
        start = next(r for r in records if r["type"] == "session_start" and r["index"] == score["index"])
        if start["session_id"] not in by_id:
            print(f"Warning: Coco session {start['session_id'][:8]} is missing from Coco's history; skipping it.")
            continue
        turns = {r["turn_number"]: r for r in records if r["type"] == "turn" and r["index"] == score["index"]}
        trace = export_conversation(by_id[start["session_id"]], include_screen_context)
        user_turn = 0
        for turn in trace["turns"]:
            if turn["role"] != "user":
                continue
            user_turn += 1
            logged = turns.get(user_turn)
            turn["simulated"] = logged is not None
            if logged:
                turn["realism"] = logged["realism"]
                turn["realism_tags"] = logged["tags"]
                # The student changed the simulated text before sending it
                turn["edited"] = not logged["matched"]
        trace.update(
            {
                "simulated": True,
                "opener": start["opener"],
                "coco_score": score["score"],
                "coco_score_reason": score["reason"],
            }
        )
        traces.append(trace)
    return traces


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
    parser.add_argument("--simulation-log", type=Path, help="export the sessions recorded by simulate_turns.py")
    parser.add_argument(
        "-o", "--output", help="output JSON file (default: traces/coco_traces.json, or traces/simulated_traces.json with --simulation-log)"
    )
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

    if args.simulation_log:
        traces = export_simulated(conversations, args.simulation_log, args.include_screen_context)
        write_traces(traces, Path(args.output or "traces/simulated_traces.json"))
        return

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

    write_traces(traces, Path(args.output or "traces/coco_traces.json"))


def write_traces(traces, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(traces)} trace(s) to {output}")


if __name__ == "__main__":
    main()
