"""Simulate human-agent collaboration with Coco, using Persimmon as the user.

For each opener in simulation_config.json, this script starts a new Coco chat with the
opener. After every Coco reply, it asks Persimmon for your next turn and sends it to
Coco for you, then asks you to rate how realistic the turn is while Coco replies. If
Coco asks a question card instead, Persimmon answers each question and the script
submits the answers. At the end of each session you rate how well Coco collaborated.

The script drives Coco through its local driver, so start Coco with it enabled:
    cd <mini-collaborative-agent>/desktop && COCO_DRIVER=1 npm start

Usage:
    python simulate_turns.py [--config simulation_config.json] [--log traces/simulation_log.jsonl]

The script can be stopped at any time (Ctrl+C) and resumes where it left off.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from humansand import Persimmon

from export_coco_traces import CARD_TOOL, default_store_path, export_message, format_card, format_question

PERSIMMON_MODEL = "persimmon-v0.1"
AGENT_NAME = "Coco"
POLL_SECONDS = 1.0
SLOW_REPLY_SECONDS = 120
TAGS = ["too agreeable", "too verbose", "reveals too much", "off-topic", "breaks character", "repetitive"]
DRIVER_HELP = (
    "Start Coco with its local driver enabled, then run this script again:\n"
    "  cd <mini-collaborative-agent>/desktop && COCO_DRIVER=1 npm start"
)


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


def ask_int(prompt, low, high):
    while True:
        answer = input(prompt).strip()
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        print(f"  Please enter a number from {low} to {high}.")


def show(label, text):
    bar = "-" * 80
    print(f"\n{bar}\n{label}\n{bar}\n{text}\n{bar}")


class Driver:
    """Client for Coco's local driver, which acts in the chat as the user would."""

    def __init__(self, info_path):
        try:
            info = json.loads(Path(info_path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            sys.exit(f"Could not find Coco's driver at {info_path}.\n{DRIVER_HELP}")
        self.url, self.token = info["url"], info["token"]

    def post(self, route, body):
        request = urllib.request.Request(
            self.url + route,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = json.load(error).get("error", error.reason)
            raise RuntimeError(f"Coco's driver refused the request: {detail}") from None
        except urllib.error.URLError:
            sys.exit(f"Could not reach Coco. Is the app still running?\n{DRIVER_HELP}")

    def start_session(self, opener):
        return self.post("/sessions", {"opener": opener})["sessionId"]

    def send_message(self, session_id, text):
        self.post(f"/sessions/{session_id}/messages", {"text": text})

    def answer_card(self, session_id, call_id, answers):
        """Submits one answer per question, or skips the card when `answers` is None."""
        self.post(f"/sessions/{session_id}/answers", {"toolCallId": call_id, "answers": answers})


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


class Simulator:
    def __init__(self, config, store, log, driver):
        self.config = config
        self.store = store
        self.log = log
        self.driver = driver
        self.user_name = config["user_name"]
        self.max_user_turns = config["max_user_turns"]
        self.client = Persimmon(api_key=os.environ["PERSIMMON_API_KEY"])

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

    def wait_for_user_turn(self, session_id, num_user_turns):
        """Wait until Coco has recorded the turn the driver just sent."""
        while True:
            c = find_conversation(self.store, session_id)
            if c and len(user_turns(c)) >= num_user_turns:
                return
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

    def log_turn(self, index, session_id, turn_number, simulated, realism, tags, **extra):
        self.log.append(
            {
                "type": "turn",
                "index": index,
                "session_id": session_id,
                "turn_number": turn_number,
                "simulated_text": simulated,
                "realism": realism,
                "tags": tags,
                # The driver sends simulated turns unchanged
                "matched": True,
                **extra,
            }
        )

    def message_turn(self, index, session_id, turn_number, conversation, setting):
        simulated = self.simulate_turn(conversation, setting)
        show(f"Simulated turn {turn_number}/{self.max_user_turns} (sent to Coco)", simulated)
        self.driver.send_message(session_id, simulated)
        # Rate while Coco works on its reply
        realism, tags = self.rate_turn()
        self.log_turn(index, session_id, turn_number, simulated, realism, tags)

    def card_turn(self, index, session_id, turn_number, conversation, card, setting):
        questions = card["result"]["questions"]
        answers = self.simulate_card(conversation, card, setting)
        print(f"\nCoco asked {len(questions)} question(s) in a question card. Simulated turn {turn_number}/{self.max_user_turns}:")
        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            show(f"Question {i}: {question['question']}", f"Answer: {answer}")
        self.driver.answer_card(session_id, card["id"], answers)
        print("(Answers submitted to Coco.)")
        realism, tags = self.rate_turn()
        self.log_turn(
            index,
            session_id,
            turn_number,
            "\n".join(f"{q['question']} {a}" for q, a in zip(questions, answers)),
            realism,
            tags,
            tool_call_id=card["id"],
            answers=answers,
        )

    def rate_unlogged_turns(self, index, session_id, conversation, turns):
        """Rate turns that were sent before the script stopped but never rated."""
        for turn_number, turn in enumerate(user_turns(conversation)[1:], 2):
            if turn_number in turns:
                continue
            show(f"Simulated turn {turn_number}/{self.max_user_turns} (sent before the script stopped)", turn["text"])
            realism, tags = self.rate_turn()
            extra = {}
            if turn.get("question_card_answers"):
                extra = {"answers": [a["answer"] for a in turn["question_card_answers"]]}
            self.log_turn(index, session_id, turn_number, turn["text"], realism, tags, **extra)

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
            show("Next opener", opener)
            input("Press Enter to start a new Coco chat with this opener (Ctrl+C to stop for now). ")
            session_id = self.driver.start_session(opener)
            print("(Opener sent to Coco.)")
            self.log.append({"type": "session_start", "index": index, "session_id": session_id, "opener": opener})
            self.wait_for_user_turn(session_id, 1)

        conversation = find_conversation(self.store, session_id)
        if not conversation:
            sys.exit(f"Coco chat {session_id[:8]} is missing from Coco's history.")
        self.rate_unlogged_turns(index, session_id, conversation, turns)
        num_user_turns = max(1, len(user_turns(conversation)))
        skipped_cards = set()
        while True:
            conversation, card = self.wait_for_coco(session_id, num_user_turns, skipped_cards)
            if num_user_turns >= self.max_user_turns:
                if card:
                    print("Coco asked a question card after the last turn of this session; skipping it.")
                    self.driver.answer_card(session_id, card["id"], None)
                    skipped_cards.add(card["id"])
                    continue
                break
            turn_number = num_user_turns + 1
            if card:
                self.card_turn(index, session_id, turn_number, conversation, card, setting)
            else:
                self.message_turn(index, session_id, turn_number, conversation, setting)
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
    parser.add_argument(
        "--driver", type=Path, default=None, help="path to Coco's driver.json (default: next to --store)"
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("PERSIMMON_API_KEY"):
        sys.exit("PERSIMMON_API_KEY is not set. Add it to your .env file.")
    with open(args.config) as f:
        config = json.load(f)
    driver = Driver(args.driver or args.store.parent / "driver.json")

    try:
        Simulator(config, args.store, SimulationLog(args.log), driver).run()
    except KeyboardInterrupt:
        print("\nStopped. Run the script again to resume where you left off.")
    except RuntimeError as error:
        sys.exit(f"{error}\nFix the problem in Coco (e.g. click Retry), then run the script again to resume.")


if __name__ == "__main__":
    main()
