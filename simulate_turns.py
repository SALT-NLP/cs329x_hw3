"""Simulate human-agent collaboration with Coco, using Persimmon as the user.

For each opener in simulation_config.json, you start a new Coco chat with the opener.
After every Coco reply, this script asks Persimmon for your next turn, copies it to
your clipboard, and asks you to rate how realistic it is. You paste it into Coco
unchanged. If Coco asks a question card instead, Persimmon answers each question and
you paste the answers into the card. At the end of each session you rate how well
Coco collaborated.

Usage:
    python simulate_turns.py [--config simulation_config.json] [--log traces/simulation_log.jsonl]

The script can be stopped at any time (Ctrl+C) and resumes where it left off.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from humansand import Persimmon

from export_coco_traces import CARD_TOOL, card_answers, default_store_path, export_message, format_card, format_question

PERSIMMON_MODEL = "persimmon-v0.1"
AGENT_NAME = "Coco"
POLL_SECONDS = 1.0
SLOW_REPLY_SECONDS = 120
TAGS = ["too agreeable", "too verbose", "reveals too much", "off-topic", "breaks character", "repetitive"]


def normalize(text):
    return " ".join(text.split())


def read_conversations(store):
    # Coco rewrites the file atomically, but retry in case we catch it mid-write
    for _ in range(5):
        try:
            with open(store) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.2)
    return []


def live_turns(conversation):
    # Unlike the export, this keeps the reply in flight so that an open question card shows up
    return [t for m in conversation.get("messages", []) if not m.get("isError") for t in export_message(m)]


def user_turns(conversation):
    return [t for t in live_turns(conversation) if t["role"] == "user"]


def open_card(conversation):
    last = conversation["messages"][-1] if conversation.get("messages") else {}
    if last.get("role") != "tutor" or not last.get("isStreaming"):
        return None
    for a in last.get("activity") or []:
        call = a.get("call") or {}
        if call.get("name") == CARD_TOOL and call.get("status") == "running" and (call.get("result") or {}).get("waitingForAnswer"):
            return call
    return None


def find_card(conversation, call_id):
    for m in conversation.get("messages", []):
        for a in m.get("activity") or []:
            if (a.get("call") or {}).get("id") == call_id:
                return a["call"]
    return None


def persimmon_turns(turns, user_name):
    """Turns as Persimmon sees them; question cards become one question and one answer at a time."""
    out, questions = [], []
    for t in turns:
        card = t.get("question_card")
        if card and card["status"] != "skipped":
            questions = card["questions"]
            out.append((AGENT_NAME, format_card(card["intro"], questions[:1])))
        elif t.get("question_card_answers"):
            for i, a in enumerate(t["question_card_answers"]):
                if i:
                    out.append((AGENT_NAME, format_question(questions[i])))
                out.append((user_name, a["answer"]))
        elif t["text"].strip():
            out.append((user_name if t["role"] == "user" else AGENT_NAME, t["text"]))
    return out


def find_conversation(store, session_id):
    return next((c for c in read_conversations(store) if c["sessionId"] == session_id), None)


def copy_to_clipboard(text):
    for cmd, encoding in [(["pbcopy"], "utf-8"), (["clip"], "utf-16"), (["wl-copy"], "utf-8"), (["xclip", "-selection", "clipboard"], "utf-8")]:
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=text.encode(encoding), check=True)
            return True
    return False


def ask_int(prompt, low, high):
    while True:
        answer = input(prompt).strip()
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        print(f"  Please enter a number from {low} to {high}.")


def show(label, text):
    bar = "-" * 80
    print(f"\n{bar}\n{label}\n{bar}\n{text}\n{bar}")


class SimulationLog:
    def __init__(self, path):
        self.path = Path(path)
        self.records = []
        if self.path.exists():
            with open(self.path) as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def append(self, record):
        record["ts"] = time.time()
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def session(self, index):
        start = next((r for r in self.records if r["type"] == "session_start" and r["index"] == index), None)
        turns = {r["turn_number"]: r for r in self.records if r["type"] == "turn" and r["index"] == index}
        score = next((r for r in self.records if r["type"] == "session_score" and r["index"] == index), None)
        return start, turns, score

    def used_session_ids(self):
        return {r["session_id"] for r in self.records if r["type"] == "session_start"}


class Simulator:
    def __init__(self, config, store, log):
        self.config = config
        self.store = store
        self.log = log
        self.user_name = config["user_name"]
        self.max_user_turns = config["max_user_turns"]
        self.client = Persimmon(api_key=os.environ["PERSIMMON_API_KEY"])

    def wait_for_new_chat(self, opener, started_at):
        target = normalize(opener)
        used = self.log.used_session_ids()
        print("Waiting for you to start a new Coco chat with this opener...")
        while True:
            for c in sorted(read_conversations(self.store), key=lambda c: c.get("createdAt", 0), reverse=True):
                texts = [t["text"] for t in user_turns(c)]
                if (
                    c["sessionId"] not in used
                    and c.get("createdAt", 0) >= (started_at - 5) * 1000
                    and texts
                    and normalize(texts[0]) == target
                ):
                    return c["sessionId"]
            time.sleep(POLL_SECONDS)

    def wait_for_coco(self, session_id, num_user_turns, ignored_cards=()):
        """Wait until Coco finishes its reply or asks a question card. Returns (conversation, card)."""
        print("Waiting for Coco...")
        waited, hinted = 0.0, False
        while True:
            c = find_conversation(self.store, session_id)
            if c and c.get("messages") and len(user_turns(c)) == num_user_turns:
                card = open_card(c)
                if card and card["id"] not in ignored_cards:
                    return c, card
                last = c["messages"][-1]
                if last["role"] == "tutor" and not last.get("isStreaming") and not last.get("isError"):
                    return c, None
            time.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            if waited > SLOW_REPLY_SECONDS and not hinted:
                print("  Still waiting. If Coco shows an error, click Retry.")
                hinted = True

    def wait_for_pasted_turn(self, session_id, num_user_turns, simulated):
        print("Paste the simulated turn into Coco and send it.")
        while True:
            c = find_conversation(self.store, session_id)
            texts = [t["text"] for t in user_turns(c)] if c else []
            if len(texts) >= num_user_turns:
                matched = normalize(texts[num_user_turns - 1]) == normalize(simulated)
                if not matched:
                    print(
                        "  Warning: the message Coco received doesn't match the simulated turn. "
                        "Please paste simulated turns unchanged; this turn will be flagged as edited."
                    )
                return matched
            time.sleep(POLL_SECONDS)

    def wait_for_card_answers(self, session_id, call_id, simulated):
        """Returns whether the submitted answers match, or None if the card was skipped."""
        print("Click Submit answers in the card when all answers are pasted.")
        while True:
            call = find_card(find_conversation(self.store, session_id) or {}, call_id) or {}
            if call.get("status") not in (None, "running"):
                answers = card_answers(call)
                if answers is None:
                    print("  The card was skipped, so this turn is dropped. Please answer cards with the simulated answers.")
                    return None
                matched = [normalize(a["answer"]) for a in answers] == [normalize(a) for a in simulated]
                if not matched:
                    print(
                        "  Warning: the answers Coco received don't match the simulated ones. "
                        "Please paste simulated answers unchanged; this turn will be flagged as edited."
                    )
                return matched
            time.sleep(POLL_SECONDS)

    def next_user_turn(self, history, setting):
        persimmon = self.client.conversations.create(
            model=PERSIMMON_MODEL,
            participants=[
                {"name": self.user_name, "description": self.config["persona"]},
                {"name": AGENT_NAME, "description": self.config["agent_description"], "generate_content": False},
            ],
            setting=setting,
        )
        for name, content in history:
            persimmon.append_turn({"name": name, "content": content})
        return persimmon.get_turn(speaker=self.user_name).content

    def simulate_turn(self, conversation, setting):
        return self.next_user_turn(persimmon_turns(live_turns(conversation), self.user_name), setting)

    def simulate_card(self, conversation, card, setting):
        # The history ends with Coco asking the first question; ask the rest one at a time
        history = persimmon_turns(live_turns(conversation), self.user_name)
        answers = []
        for i, question in enumerate(card["result"]["questions"]):
            if i:
                history.append((AGENT_NAME, format_question(question)))
            answers.append(self.next_user_turn(history, setting))
            history.append((self.user_name, answers[-1]))
        return answers

    def rate_turn(self):
        realism = ask_int("Realism (1-5): could you plausibly have written this at this point? ", 1, 5)
        print("Optional tags (comma-separated numbers, Enter to skip):")
        for i, tag in enumerate(TAGS, 1):
            print(f"  {i}. {tag}")
        picked = input("Tags: ").strip()
        tags = [TAGS[int(x) - 1] for x in picked.replace(" ", "").split(",") if x.isdigit() and 1 <= int(x) <= len(TAGS)]
        return realism, tags

    def message_turn(self, index, session_id, turn_number, conversation, setting, logged):
        simulated = logged["simulated_text"] if logged else self.simulate_turn(conversation, setting)
        show(f"Simulated turn {turn_number}/{self.max_user_turns} (copied to clipboard)", simulated)
        if not copy_to_clipboard(simulated):
            print("(Could not access the clipboard; copy the turn above manually.)")
        if logged:
            print("This turn was already rated before the script stopped.")
        else:
            realism, tags = self.rate_turn()
        matched = self.wait_for_pasted_turn(session_id, turn_number, simulated)
        if not logged:
            self.log.append(
                {
                    "type": "turn",
                    "index": index,
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "simulated_text": simulated,
                    "realism": realism,
                    "tags": tags,
                    "matched": matched,
                }
            )

    def card_turn(self, index, session_id, turn_number, conversation, card, setting, logged):
        questions = card["result"]["questions"]
        answers = logged["answers"] if logged else self.simulate_card(conversation, card, setting)
        print(f"\nCoco asked {len(questions)} question(s) in a question card. Simulated turn {turn_number}/{self.max_user_turns}:")
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            show(f"Question {i}: {question['question']}", f"Answer: {answer}")
        if logged:
            print("This turn was already rated before the script stopped.")
        else:
            realism, tags = self.rate_turn()
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            if not copy_to_clipboard(answer):
                print(f"(Could not access the clipboard; copy answer {i} above manually.)")
            input(
                f"Answer {i} is copied. Paste it into the text box under \"{question['question']}\" "
                "(don't click an option), then press Enter here. "
            )
        matched = self.wait_for_card_answers(session_id, card["id"], answers)
        if matched is not None and not logged:
            self.log.append(
                {
                    "type": "turn",
                    "index": index,
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "tool_call_id": card["id"],
                    "answers": answers,
                    "simulated_text": "\n".join(f"{q['question']} {a}" for q, a in zip(questions, answers)),
                    "realism": realism,
                    "tags": tags,
                    "matched": matched,
                }
            )
        return matched

    def run_session(self, index, session):
        opener, setting = session["opener"], session["setting"]
        start, turns, score = self.log.session(index)
        if score:
            return
        print(f"\n================ Session {index + 1}/{len(self.config['sessions'])} ================")

        if start:
            session_id = start["session_id"]
            print(f"Resuming Coco chat {session_id[:8]}.")
        else:
            show("Opener (copied to clipboard)", opener)
            if not copy_to_clipboard(opener):
                print("(Could not access the clipboard; copy the opener above manually.)")
            print("Start a new session in Coco, paste the opener, and send it.")
            session_id = self.wait_for_new_chat(opener, time.time())
            self.log.append({"type": "session_start", "index": index, "session_id": session_id, "opener": opener})

        conversation = find_conversation(self.store, session_id)
        num_user_turns = max(1, len(user_turns(conversation)) if conversation else 1)
        skipped_cards = set()
        while True:
            conversation, card = self.wait_for_coco(session_id, num_user_turns, skipped_cards)
            if num_user_turns >= self.max_user_turns:
                if card:
                    print("Coco asked a question card after the last turn of this session. Click Skip so Coco finishes its reply.")
                    skipped_cards.add(card["id"])
                    continue
                break
            turn_number = num_user_turns + 1
            logged = turns.get(turn_number)
            if logged and (logged.get("tool_call_id") != (card or {}).get("id")):
                logged = None  # Coco's reply changed since this turn was logged (e.g. after Retry)
            if card:
                matched = self.card_turn(index, session_id, turn_number, conversation, card, setting, logged)
                if matched is None:
                    continue
            else:
                self.message_turn(index, session_id, turn_number, conversation, setting, logged)
            num_user_turns = turn_number

        print("\nSession complete. Now rate Coco, thinking about the whole session.")
        coco_score = ask_int("Coco's collaboration quality in this session (1-5): ", 1, 5)
        reason = input("In one line, why? ").strip()
        self.log.append({"type": "session_score", "index": index, "session_id": session_id, "score": coco_score, "reason": reason})

    def run(self):
        for index, session in enumerate(self.config["sessions"]):
            self.run_session(index, session)
        print("\nAll sessions are done. Export them with:")
        print("  python export_coco_traces.py --simulation-log traces/simulation_log.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="simulation_config.json")
    parser.add_argument("--log", default="traces/simulation_log.jsonl")
    parser.add_argument("--store", type=Path, default=default_store_path(), help="path to Coco's chat-conversations.json")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("PERSIMMON_API_KEY"):
        sys.exit("PERSIMMON_API_KEY is not set. Add it to your .env file.")
    with open(args.config) as f:
        config = json.load(f)

    try:
        Simulator(config, args.store, SimulationLog(args.log)).run()
    except KeyboardInterrupt:
        print("\nStopped. Run the script again to resume where you left off.")


if __name__ == "__main__":
    main()
