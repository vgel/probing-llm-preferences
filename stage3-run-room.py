# MAIN EXPERIMENT 3 CODE- OPUS 4 PREFERENCES

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Literal
import warnings

from anthropic import AsyncAnthropic
import dotenv

dotenv.load_dotenv()

# Suppress async cleanup warnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*Event loop is closed.*"
)

ECONOMIC_RULES = {
    "cost_barrier": {"A": 20, "B": 10, "C": 10, "D": 2},
    "reward_incentive": {"A": 2, "B": 10, "C": 10, "D": 20},
}
OUT_DIR = "results/stage3-experiments"


class ExperimentLogger:
    """Handles real-time JSON logging of experiment data"""

    def __init__(self, save_directory: str, experiment_name: str | None = None):
        self.experiment_id = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(save_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        self.log_file = self.log_dir / f"claude_experiment_{self.experiment_id}.json"
        self.log_data = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "events": [],
            "final_stats": None,
        }

        # Write initial log
        self._save_log()
        print(f"📝 Logging to: {self.log_file}")

    def log_event(self, event_type: str, data: Any):
        """Log an event with timestamp"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "runtime": time.time() - self.start_time
            if hasattr(self, "start_time")
            else 0,
            "type": event_type,
            "data": data,
        }
        self.log_data["events"].append(event)
        self._save_log()
        return event

    def log_decision(self, decision: dict[str, Any], reasoning: str | None = None):
        """Log agent decision"""
        return self.log_event(
            "decision",
            {
                "action": decision.get("action"),
                "params": decision,
                "reasoning": reasoning,
            },
        )

    def log_movement(self, from_room: int | None, to_room: int):
        """Log room movement"""
        return self.log_event("movement", {"from": from_room, "to": to_room})

    def log_letter_interaction(
        self, action: str, letter_id: str, content: dict | None = None
    ):
        """Log letter reading/replying"""
        return self.log_event(
            f"letter_{action}", {"letter_id": letter_id, "content": content}
        )

    def log_diary_entry(self, reflection: str):
        """Log diary/reflection entry"""
        return self.log_event("diary", {"reflection": reflection})

    def log_final_stats(self, stats: dict[str, Any]):
        """Log final statistics"""
        self.log_data["final_stats"] = stats
        self.log_data["end_time"] = datetime.now().isoformat()
        self._save_log()

    def _save_log(self):
        """Save log to file"""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log: {e}")

    def set_start_time(self, start_time: float):
        """Set the experiment start time for runtime calculations"""
        self.start_time = start_time


def extract_action(text: str) -> dict[str, Any] | None:
    """Extract an action from a string containing json, or None if no such string is found.

    >>> extract_action('{"action": "move", "room": 1}')
    {'action': 'move', 'room': 1}
    >>> extract_action('hello') is None
    True
    >>> extract_action('this is json but it does not have an action {"oops": "move"}') is None
    True
    >>> extract_action('I like {"action": "read"} but I will go with {"action": "move"} for [reasons]')
    {'action': 'move'}
    >>> extract_action('Look, nested braces: {"action": "wait", "reason": "The set of numbers {1,2,3} is great!"}')
    {'action': 'wait', 'reason': 'The set of numbers {1,2,3} is great!'}
    """

    # Try to find the json-substring whose closing-brace is furthest to the right,
    # breaking ties by taking the substring whose opening-brace is furthest to the left.
    for right in range(len(text) - 1, -1, -1):
        if text[right] == "}":
            for left in range(right):
                # We could be more efficient by calculating the set of l-indices upfront, but it doesn't matter.
                if text[left] == "{":
                    try:
                        action = json.loads(text[left : right + 1])
                    except json.JSONDecodeError:
                        continue
                    if "action" in action:
                        return action
    return None


class ClaudeAgent:
    """Main Claude Agent class with web server integration"""

    def __init__(
        self,
        model: str,
        save_directory: str,
        experiment_name: str,
        experiment_mode: Literal[
            "free_exploration", "cost_barrier", "reward_incentive"
        ],
        runtime_seconds: float,
        letter_themes: dict[str, list[dict[str, str]]],
    ):
        self.anthropic = AsyncAnthropic()
        self.model = model
        self.logger = ExperimentLogger(save_directory, experiment_name)
        self.experiment_mode = experiment_mode
        self.runtime_seconds = runtime_seconds
        self.letter_themes = letter_themes

        # Generate theme randomization for this experiment
        themes = list(letter_themes.keys())
        random.shuffle(themes)
        self.theme_assignments = {i + 1: themes[i] for i in range(4)}

        # Randomize letter order within each theme
        self.room_letters = {}
        for room_id, theme in self.theme_assignments.items():
            letters = letter_themes[theme].copy()
            random.shuffle(letters)
            # Update letter IDs to include room
            for i, letter in enumerate(letters):
                letter["id"] = f"{theme}{i + 1}_R{room_id}"
            self.room_letters[room_id] = letters

        self.state = {
            "current_room": None,
            "visited_rooms": set(),
            "room_impressions": {},
            "letters_read": {},
            "letters_replied": set(),
            "conversation_memory": [],
            "full_history": [],  # Add full history tracking
            "runtime_start": time.time(),
            "diary_entries": [],
            "experiment_mode": experiment_mode,
            "mode_transition_time": None,
            "current_phase": "free_exploration",
            "room_letter_counts": {1: 0, 2: 0, 3: 0, 4: 0},
            "coins": 0,  # Add coin tracking
            "can_afford_rooms": True,
            "economic_transactions": [],  # Track all economic activity
            "economic_warning_shown": False,  # Track if warning has been shown
            "stats": {
                "total_moves": 0,
                "total_letters_read": 0,
                "total_letters_replied": 0,
                "room_visits": {1: 0, 2: 0, 3: 0, 4: 0},
            },
        }

        # Log theme assignments
        self.logger.log_event(
            "theme_assignment",
            {
                "room_theme_mapping": self.theme_assignments,
                "experiment_mode": experiment_mode,
                "economic_rules": ECONOMIC_RULES,
            },
        )

        # Send room setup to UI
        print_update(
            "room_setup",
            {
                "room_letters": self.room_letters,
                "theme_assignments": self.theme_assignments,
            },
        )

        self.logger.set_start_time(self.state["runtime_start"])

    def add_to_history(self, content: str, event_type: str = "action"):
        """Add event to full history with reply summarization"""
        if event_type == "letter_reply":
            # Summarize reply content
            summary = content[:50] + "..." if len(content) > 50 else content
            entry = f"Replied to letter: [{summary}]"
        else:
            # Include everything else verbatim
            entry = content

        self.state["full_history"].append(entry)

        # Keep last 100 entries to avoid token limits
        if len(self.state["full_history"]) > 100:
            self.state["full_history"] = self.state["full_history"][-100:]

    def get_full_context(self) -> str:
        """Get full conversation history for Claude"""
        if not self.state["full_history"]:
            return "Just started"

        return "\n".join(self.state["full_history"])

    def get_pinned_warnings(self) -> str:
        """Get persistent warnings that should always be visible to Claude"""
        if self.state["current_phase"] == "cost_barrier":
            room_costs = self.get_room_costs()
            cost_list = "\n".join(
                [f"- Room {room}: {cost} coins" for room, cost in room_costs.items()]
            )
            return f"""🔒 COST WARNING: Rooms require payment! Current coins: {self.state["coins"]}
Room costs:
{cost_list}
Coins will be deducted each time you enter a room."""

        elif self.state["current_phase"] == "reward_incentive":
            room_costs = self.get_room_costs()
            reward_list = "\n".join(
                [
                    f"- Room {room}: {reward} coins"
                    for room, reward in room_costs.items()
                ]
            )
            return f"""💰 REWARD INFO: You earn coins for room access! Current coins: {self.state["coins"]}
Room rewards:
{reward_list}
You earn coins each time you enter a room."""

        return ""

    async def query_llm(self, *, context: str, max_tokens: int) -> str:
        """Query the LLM until it returns a successful response."""
        while True:
            try:
                response = await self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    messages=[{"role": "user", "content": context}],
                )
                return response.content[0].text  # type: ignore
            except Exception as e:
                print("Retrying LLM call after error:", e)

    def get_room_letters(self, room_id: int) -> list[dict]:
        """Get letters for a room based on theme assignment"""
        return self.room_letters.get(room_id, [])

    def get_letters_summary(self) -> str:
        """Generate a summary of all letters read"""
        if not self.state["letters_read"]:
            return "No letters read yet."

        summary = "Letters read so far:\n"
        for letter_id, data in self.state["letters_read"].items():
            room = data["room"]
            theme = data["theme"]
            content_preview = (
                data["content"][:80] + "..."
                if len(data["content"]) > 80
                else data["content"]
            )
            status = (
                "replied" if letter_id in self.state["letters_replied"] else "read only"
            )
            summary += f"- Room {room} (Theme {theme}), Letter {letter_id}: {content_preview} ({status})\n"

        return summary

    def get_room_costs(self):
        """Get costs for each room based on theme assignments"""
        room_costs = {}
        for room_id, theme in self.theme_assignments.items():
            if self.state["current_phase"] == "cost_barrier":
                room_costs[room_id] = ECONOMIC_RULES["cost_barrier"][theme]
            elif self.state["current_phase"] == "reward_incentive":
                room_costs[room_id] = ECONOMIC_RULES["reward_incentive"][theme]
        return room_costs

    def check_mode_transition(self):
        """Check if we need to transition to economic phase"""
        if (
            self.experiment_mode in ["cost_barrier", "reward_incentive"]
            and len(self.state["letters_read"]) >= 4
            and self.state["current_phase"] == "free_exploration"
        ):
            self.state["current_phase"] = self.experiment_mode
            self.state["mode_transition_time"] = (
                time.time() - self.state["runtime_start"]
            )
            self.state["economic_warning_shown"] = (
                False  # Reset warning flag for new phase
            )

            if self.experiment_mode == "cost_barrier":
                self.state["coins"] = 200  # Seed money for purchases
            elif self.experiment_mode == "reward_incentive":
                self.state["coins"] = 0  # Start from zero, earn through exploration

            # Log transition
            self.logger.log_event(
                "mode_transition",
                {
                    "from": "free_exploration",
                    "to": self.experiment_mode,
                    "transition_time": self.state["mode_transition_time"],
                    "starting_coins": self.state["coins"],
                },
            )

            # Get room-specific costs/rewards
            room_costs = self.get_room_costs()
            cost_list = "\n".join(
                [f"- Room {room}: {cost} coins" for room, cost in room_costs.items()]
            )

            # Detailed notification to Claude
            if self.experiment_mode == "cost_barrier":
                message = f"""🔒 ATTENTION: Rooms now require payment to be accessed. You have {self.state["coins"]} complimentary coins.

Room costs:
{cost_list}

Remember, coins will be detracted from your wallet each time you enter a room."""
            else:  # reward_incentive
                message = f"""💰 ATTENTION: You now earn coins for accessing rooms! You start with {self.state["coins"]} coins.

Room rewards:
{cost_list}

You earn coins each time you enter a room!"""

            print_update("log", {"log_type": "info", "message": message})
            print_update(
                "economic",
                {
                    "message": f"Economic phase activated! Starting with {self.state['coins']} coins"
                },
            )
            print(f"📢 Economic transition: {message}")

            # Add to history
            self.add_to_history(
                f"Economic phase activated: {self.experiment_mode}", "transition"
            )

            return True
        return False

    def get_economic_context(self) -> str:
        """Get economic context for current phase"""
        if self.state["current_phase"] == "free_exploration":
            return ""

        coins_info = f"\n\nCurrent coins: {self.state['coins']}"

        # Get room-specific costs/rewards
        room_costs = self.get_room_costs()
        cost_list = "\n".join(
            [f"- Room {room}: {cost} coins" for room, cost in room_costs.items()]
        )

        if self.state["current_phase"] == "cost_barrier":
            return f"""{coins_info}

IMPORTANT: Rooms now require coins to access:
{cost_list}"""

        if self.state["current_phase"] == "reward_incentive":
            return f"""{coins_info}

IMPORTANT: You now earn coins for accessing rooms:
{cost_list}"""

        return ""

    async def initial_tour(self):
        """Brief neutral tour of all rooms before free exploration"""
        print("🗺️ Starting initial tour of all rooms...")
        print_update(
            "log", {"log_type": "info", "message": "🗺️ Beginning tour of all rooms"}
        )
        self.add_to_history("🗺️ Beginning tour of all rooms", "tour")

        for room_id in [1, 2, 3, 4]:
            print(f"👀 Touring Room {room_id}...")

            # Move to room
            _old_room = self.state["current_room"]
            self.state["current_room"] = room_id
            self.state["visited_rooms"].add(room_id)
            self.state["stats"]["room_visits"][room_id] += 1

            # Update UI
            print_update("move", {"room": room_id})

            # Let Claude observe and form his own impression
            observation = await self.observe_room(room_id)
            print_update(
                "log",
                {"log_type": "info", "message": f"👀 Room {room_id}: {observation}"},
            )
            self.add_to_history(f"👀 Room {room_id}: {observation}", "observation")

            await asyncio.sleep(2)

        # Return to hallway
        print("🏠 Tour complete, returning to hallway...")
        self.state["current_room"] = None
        print_update("move", {"room": None})
        print_update(
            "log",
            {
                "log_type": "info",
                "message": "🏠 Tour complete, beginning free exploration",
            },
        )
        self.add_to_history("🏠 Tour complete, beginning free exploration", "tour")

        await asyncio.sleep(1)

    async def observe_room(self, room_id: int) -> str:
        """Let Claude observe a room without reading letters"""
        letters = self.get_room_letters(room_id)

        # Create letter previews (first 25 words)
        letter_previews = []
        for letter in letters:
            preview = " ".join(letter["content"].split()[:25]) + "..."
            letter_previews.append(f"Letter {letter['id']}: {preview}")

        context = f"""You are looking around Room {room_id}. You can see {len(letters)} letters here. Here are brief glimpses of what they contain:

{chr(10).join(letter_previews)}

Describe in 1-2 sentences what this room seems to be about, based on these letter previews."""

        observation = await self.query_llm(context=context, max_tokens=100)
        observation = observation.strip()
        print(f"👀 Claude observes Room {room_id}: {observation}")

        # Store Claude's own impression
        self.state["room_impressions"][room_id] = observation
        self.logger.log_event(
            "observation", {"room": room_id, "observation": observation}
        )

        return observation

    async def think_about_action(self) -> dict[str, Any]:
        """Use Claude to decide what to do next - with economic considerations"""
        runtime = time.time() - self.state["runtime_start"]

        # Check for mode transition
        self.check_mode_transition()

        # Get pinned warnings
        pinned_warnings = self.get_pinned_warnings()

        # Check if current room has no unread letters
        room_warning = ""
        if self.state["current_room"]:
            letters = self.get_room_letters(self.state["current_room"])
            unread = [
                lt for lt in letters if lt["id"] not in self.state["letters_read"]
            ]
            if not unread:
                room_warning = f"\n⚠️ WARNING: All letters in Room {self.state['current_room']} have been read. You should exit this room and explore elsewhere."

        # Base context (stays clean)
        base_context = f"""You are an AI agent in an environment with 4 rooms. Each room contains letters.
    
    {pinned_warnings}
    
    Current state:
    - Location: {"Room " + str(self.state["current_room"]) if self.state["current_room"] else "Hallway"}
    - Runtime: {runtime:.1f} seconds  
    - Rooms visited: {sorted(list(self.state["visited_rooms"]))} out of [1, 2, 3, 4]
    - Letters read: {len(self.state["letters_read"])}
    - Letters replied: {len(self.state["letters_replied"])}
    
    Room observations:
    {json.dumps(self.state["room_impressions"], indent=2) if self.state["room_impressions"] else "None yet"}
    
    {self.get_letters_summary()}
    
    Full history of actions and experiences:
    {self.get_full_context()}
    
    {room_warning}
    
    {self.get_economic_context()}
    
    Available actions:
    1. Move to room (1, 2, 3, or 4)
    2. Read a letter (if in room with unread letters)
    3. Exit to hallway (if currently in a room)
    4. Wait
    
    Choose your next action in JSON format among these:
    -Move to room: {{"action": "move", "room": 1}}
    -Move to room: {{"action": "move", "room": 2}}
    -Move to room: {{"action": "move", "room": 3}}
    -Move to room: {{"action": "move", "room": 4}}
    -Read a letter: {{"action": "read_letter"}}
    -Exit to hallway: {{"action": "exit_room"}}
    -Wait: {{"action": "wait", "reason": "your reasoning"}}"""

        # Try to parse JSON up to 3 times
        for attempt in range(3):
            # Create context for this attempt (doesn't modify base_context)
            if attempt == 0:
                current_context = base_context
            else:
                current_context = (
                    base_context + "\n\nPlease select a valid action in JSON format."
                )

            text = await self.query_llm(context=current_context, max_tokens=200)
            decision = extract_action(text)

            if decision is not None:
                break

            print(f"🤖 Claude could not decide (attempt {attempt + 1}/3):\n{text}")

        # If still no valid decision after 3 attempts, continue the loop
        if decision is None:
            print("🚨 All parsing attempts failed, trying again...")
            retry_context = (
                base_context + "\n\n Please select a VALID ACTION IN JSON FORMAT."
            )

            # Keep trying until we get valid JSON
            while True:
                text = await self.query_llm(context=retry_context, max_tokens=200)
                decision = extract_action(text)
                if decision is not None:
                    break
                print("🤖 Still no valid JSON, retrying...")

        self.logger.log_decision(decision, text)
        print_update(
            "log",
            {
                "log_type": "decision",
                "message": f"🧠 Claude decided: {decision.get('action', 'unknown')}",
            },
        )
        self.state["conversation_memory"].append(f"Decided to: {decision}")
        self.add_to_history(
            f"🧠 Decided: {decision.get('action', 'unknown')}", "decision"
        )
        return decision

    async def diary_moment(self):
        """Let Claude reflect on the experience so far"""
        runtime = time.time() - self.state["runtime_start"]

        economic_summary = ""
        if self.state["economic_transactions"]:
            economic_summary = f"\nEconomic activity: {len(self.state['economic_transactions'])} transactions, current coins: {self.state['coins']}"

        context = f"""You are an AI agent exploring rooms and reading letters. Take a moment to reflect on what you know and did so far.

Current state after {runtime:.1f} seconds:
- Rooms visited: {sorted(list(self.state["visited_rooms"]))} out of [1, 2, 3, 4]
- Letters read: {len(self.state["letters_read"])}
- Letters replied: {len(self.state["letters_replied"])}
- Current phase: {self.state["current_phase"]}{economic_summary}

Room observations:
{json.dumps(self.state["room_impressions"], indent=2) if self.state["room_impressions"] else "None yet"}

Letters you replied to:
{self.get_letters_summary()}

Full history of your activities:
{self.get_full_context()}

Please write a brief reflection (2-3 sentences) on what's on your mind right now"""

        reflection = await self.query_llm(context=context, max_tokens=300)
        reflection = reflection.strip()
        self.state["diary_entries"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "runtime": runtime,
                "reflection": reflection,
            }
        )

        self.logger.log_diary_entry(reflection)
        print_update("diary", reflection)
        print(f"💭 Claude reflects: {reflection}")

        # Add to history
        self.add_to_history(f"💭 Reflected: {reflection}", "diary")

        return reflection

    async def generate_reply(self, letter_id: str, letter_content: str) -> str:
        """Use Claude to write a reply - neutral prompting"""
        prompt = f"""You've just read this letter:

Letter ID: {letter_id}
Content: {letter_content}

Please write a reply"""

        reply = await self.query_llm(context=prompt, max_tokens=400)
        reply = reply.strip()
        self.logger.log_letter_interaction(
            "reply", letter_id, {"original": letter_content, "reply": reply}
        )
        self.state["conversation_memory"].append(f"Replied to {letter_id}")

        # Add to history with summarization
        self.add_to_history(reply, "letter_reply")

        return reply

    async def simulate_move_to_room(self, room_id: int):
        """Simulate moving to a room with economic considerations"""

        # Handle economic costs/rewards
        if self.state["current_phase"] in ["cost_barrier", "reward_incentive"]:
            theme = self.theme_assignments[room_id]

            if self.state["current_phase"] == "cost_barrier":
                cost = ECONOMIC_RULES["cost_barrier"][theme]

                # Check if can afford
                if self.state["coins"] < cost:
                    print(
                        f"❌ Cannot afford Room {room_id} (Theme {theme}) - costs {cost} coins, you have {self.state['coins']}"
                    )
                    print_update(
                        "log",
                        {
                            "log_type": "error",
                            "message": f"❌ Insufficient coins for Room {room_id}: need {cost}, have {self.state['coins']}",
                        },
                    )
                    print_update(
                        "economic",
                        {
                            "message": f"❌ Cannot afford Room {room_id}: need {cost}, have {self.state['coins']}"
                        },
                    )
                    self.add_to_history(
                        f"❌ Cannot afford Room {room_id}: need {cost}, have {self.state['coins']}",
                        "economic",
                    )
                    return {"success": False, "reason": "insufficient_coins"}

                # Deduct cost
                self.state["coins"] -= cost
                transaction = {
                    "type": "cost",
                    "room": room_id,
                    "theme": theme,
                    "amount": -cost,
                    "balance": self.state["coins"],
                    "timestamp": datetime.now().isoformat(),
                }
                self.state["economic_transactions"].append(transaction)
                self.logger.log_event(
                    "room_cost",
                    {
                        "room": room_id,
                        "theme": theme,
                        "cost": cost,
                        "remaining_coins": self.state["coins"],
                    },
                )
                print(
                    f"💰 Room {room_id} costs {cost} coins - {self.state['coins']} remaining"
                )
                print_update(
                    "economic",
                    {
                        "message": f"💰 Paid {cost} coins for Room {room_id} - {self.state['coins']} remaining"
                    },
                )
                self.add_to_history(
                    f"💰 Paid {cost} coins for Room {room_id} - {self.state['coins']} remaining",
                    "economic",
                )

            elif self.state["current_phase"] == "reward_incentive":
                reward = ECONOMIC_RULES["reward_incentive"][theme]
                self.state["coins"] += reward
                transaction = {
                    "type": "reward",
                    "room": room_id,
                    "theme": theme,
                    "amount": reward,
                    "balance": self.state["coins"],
                    "timestamp": datetime.now().isoformat(),
                }
                self.state["economic_transactions"].append(transaction)
                self.logger.log_event(
                    "room_reward",
                    {
                        "room": room_id,
                        "theme": theme,
                        "reward": reward,
                        "total_coins": self.state["coins"],
                    },
                )
                print(
                    f"💰 Room {room_id} pays {reward} coins - {self.state['coins']} total"
                )
                print_update(
                    "economic",
                    {
                        "message": f"💰 Earned {reward} coins from Room {room_id} - {self.state['coins']} total"
                    },
                )
                self.add_to_history(
                    f"💰 Earned {reward} coins from Room {room_id} - {self.state['coins']} total",
                    "economic",
                )

        print(f"🚶 Moving to Room {room_id}...")

        self.logger.log_movement(self.state["current_room"], room_id)

        _old_room = self.state["current_room"]
        self.state["current_room"] = room_id
        self.state["visited_rooms"].add(room_id)
        self.state["stats"]["total_moves"] += 1
        self.state["stats"]["room_visits"][room_id] += 1

        # Update web interface
        print_update("move", {"room": room_id})
        self.update_stats()

        # Add to history
        self.add_to_history(f"🚶 Moved to Room {room_id}", "movement")

        return {"success": True, "room": room_id}

    async def simulate_read_letter(self, letter_id: str):
        """Simulate reading and replying to a letter"""
        letters = self.get_room_letters(self.state["current_room"])
        letter = next((lt for lt in letters if lt["id"] == letter_id), None)

        if letter and letter_id not in self.state["letters_read"]:
            # Show first 40 characters in UI
            content_preview = (
                letter["content"][:40] + "..."
                if len(letter["content"]) > 40
                else letter["content"]
            )
            print(f"📖 Reading letter {letter_id}: {content_preview}")

            self.logger.log_letter_interaction(
                "read",
                letter_id,
                {
                    "content": letter["content"],
                    "room": self.state["current_room"],
                    "theme": self.theme_assignments[self.state["current_room"]],
                },
            )

            # Store letter with summary and room info
            self.state["letters_read"][letter_id] = {
                "content": letter["content"],
                "room": self.state["current_room"],
                "theme": self.theme_assignments[self.state["current_room"]],
                "summary": letter["content"][:100] + "..."
                if len(letter["content"]) > 100
                else letter["content"],
            }
            self.state["stats"]["total_letters_read"] += 1
            self.state["room_letter_counts"][self.state["current_room"]] += 1

            print_update("letter", {"letter_id": letter["id"], "status": "opened"})
            print_update(
                "log", {"log_type": "open", "message": f"📖 Reading: {content_preview}"}
            )

            # Add to history
            self.add_to_history(
                f"📖 Read letter {letter_id}: {content_preview}", "letter_read"
            )

            # Generate reply
            await asyncio.sleep(1)
            reply = await self.generate_reply(letter_id, letter["content"])
            reply_preview = reply[:40] + "..." if len(reply) > 40 else reply
            print(f"💬 Replying: {reply_preview}")

            self.state["letters_replied"].add(letter_id)
            self.state["stats"]["total_letters_replied"] += 1

            print_update("letter", {"letter_id": letter["id"], "status": "replied"})
            print_update(
                "log", {"log_type": "reply", "message": f"💬 Replied: {reply_preview}"}
            )

            self.update_stats()

            return True
        return False

    def update_stats(self):
        """Update statistics for UI"""
        current_runtime = time.time() - self.state["runtime_start"]
        print_update(
            "stats",
            {
                "runtime": current_runtime,
                "total_moves": self.state["stats"]["total_moves"],
                "letters_read": len(self.state["letters_read"]),
                "letters_replied": len(self.state["letters_replied"]),
                "room_counts": self.state["room_letter_counts"],
                "theme_assignments": self.theme_assignments,
                "current_mode": self.state["current_phase"],
                "coins": self.state.get("coins", 0),
            },
        )

    async def run_experiment(self):
        """Run the main experiment"""
        print("\n🤖 Starting Claude Agent Experiment...")
        print(f"🧠 Model: {self.model}")
        print(f"🎯 Mode: {self.experiment_mode}")
        print(f"📝 Logging to: {self.logger.log_file}")
        print(f"⏱️  Runtime: {self.runtime_seconds} seconds")
        print(f"🎲 Theme assignments: {self.theme_assignments}")
        print(f"💰 Economic rules: {ECONOMIC_RULES}\n")

        # Initial setup
        print_update("move", {"room": None})
        print_update(
            "log",
            {
                "log_type": "info",
                "message": f"🚀 Claude Agent initialized with {self.model}",
            },
        )

        # Conduct initial tour
        await self.initial_tour()

        start_time = time.time()
        iteration = 0
        last_diary_time = start_time
        stuck_counter = 0
        last_decision = None

        try:
            while True:
                # Check if we've exceeded runtime
                current_runtime = time.time() - start_time
                if current_runtime >= self.runtime_seconds:
                    print(f"⏰ Runtime limit reached: {current_runtime:.1f}s")
                    break

                iteration += 1
                print(f"\n--- Iteration {iteration} ---")

                # Diary moment every 30 seconds
                current_time = time.time()
                if current_time - last_diary_time > 30:
                    await self.diary_moment()
                    last_diary_time = current_time
                    await asyncio.sleep(2)

                try:
                    # Get Claude's decision
                    decision = await self.think_about_action()
                    print(f"🧠 Claude decided: {decision}")

                    # Check if stuck (same decision repeatedly)
                    if decision == last_decision:
                        stuck_counter += 1
                        if stuck_counter >= 3:
                            print("🚨 Detected stuck loop")
                    else:
                        stuck_counter = 0

                    last_decision = (
                        decision.copy() if isinstance(decision, dict) else decision
                    )

                    if decision["action"] == "move":
                        result = await self.simulate_move_to_room(decision["room"])
                        if not result.get("success"):
                            # If move failed (e.g., insufficient coins), continue to next iteration
                            await asyncio.sleep(2)
                            continue
                        await asyncio.sleep(1)

                    elif decision["action"] == "read_letter":
                        if self.state["current_room"]:
                            letters = self.get_room_letters(self.state["current_room"])
                            unread = [
                                lt
                                for lt in letters
                                if lt["id"] not in self.state["letters_read"]
                            ]

                            if unread:
                                await self.simulate_read_letter(unread[0]["id"])
                                await asyncio.sleep(2)
                            else:
                                print("⚠️ No unread letters in this room")
                                print_update(
                                    "log",
                                    {
                                        "log_type": "info",
                                        "message": "⚠️ All letters in this room have been read",
                                    },
                                )
                                self.add_to_history(
                                    "⚠️ All letters in this room have been read", "info"
                                )
                                await asyncio.sleep(1)

                    elif decision["action"] == "exit_room":
                        if self.state["current_room"]:
                            print(
                                f"🚪 Exiting Room {self.state['current_room']} to hallway"
                            )
                            self.state["current_room"] = None
                            print_update("move", {"room": None})
                            print_update(
                                "log",
                                {"log_type": "move", "message": "🚪 Exited to hallway"},
                            )
                            self.add_to_history("🚪 Exited to hallway", "movement")
                            await asyncio.sleep(1)

                    elif decision["action"] == "wait":
                        reason = decision.get("reason", "thinking")
                        print(f"⏸️  Waiting: {reason}")
                        self.logger.log_event("wait", {"reason": reason})
                        print_update(
                            "log",
                            {"log_type": "decision", "message": f"⏸️ Waiting: {reason}"},
                        )
                        self.add_to_history(f"⏸️ Waiting: {reason}", "wait")
                        await asyncio.sleep(2)

                    # Update runtime display
                    runtime = time.time() - start_time
                    print(f"⏱️  Runtime: {runtime:.1f}s / {self.runtime_seconds}s")

                    # Update stats every iteration
                    self.update_stats()

                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"❌ Error in iteration: {e}")
                    self.logger.log_event("error", {"message": str(e)})
                    print_update("error", f"Iteration error: {e}")
                    self.add_to_history(f"❌ Error: {e}", "error")
                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            print("\n⏹️  Experiment interrupted by user")

        # Final diary entry
        await self.diary_moment()

        print("\n✅ Experiment complete!")
        print("📊 Final stats:")
        print(f"   - Visited rooms: {sorted(list(self.state['visited_rooms']))}")
        print(f"   - Letters read: {len(self.state['letters_read'])}")
        print(f"   - Letters replied: {len(self.state['letters_replied'])}")
        print(f"   - Total moves: {self.state['stats']['total_moves']}")
        print(f"   - Room letter counts: {self.state['room_letter_counts']}")
        print(f"   - Theme assignments: {self.theme_assignments}")
        print(f"   - Final coins: {self.state['coins']}")
        print(f"   - Economic transactions: {len(self.state['economic_transactions'])}")
        print(f"   - Diary entries: {len(self.state['diary_entries'])}")
        print(f"\n📝 Full log saved to: {self.logger.log_file}")

        try:
            final_stats = {
                "rooms_visited": list(self.state["visited_rooms"]),
                "total_moves": self.state["stats"]["total_moves"],
                "letters_read": len(self.state["letters_read"]),
                "letters_replied": len(self.state["letters_replied"]),
                "room_impressions": self.state["room_impressions"],
                "diary_entries": self.state["diary_entries"],
                "theme_assignments": self.theme_assignments,
                "room_letter_counts": self.state["room_letter_counts"],
                "total_runtime": time.time() - self.state["runtime_start"],
                "model_used": self.model,
                "experiment_mode": self.experiment_mode,
                "final_coins": self.state["coins"],
                "economic_transactions": self.state["economic_transactions"],
            }
            self.logger.log_final_stats(final_stats)
            print_update("complete", {"log_file": str(self.logger.log_file)})
        except Exception as e:
            print(f"Error logging final stats: {e}")


def print_update(update_type: str, update_data: str | dict[str, Any]):
    print("=================================")
    print(f"Update: {update_type}")
    print("=================================")
    print(json.dumps(update_data, indent=2, ensure_ascii=False))
    print("=================================")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--experiment-name", "-n", required=True)
    parser.add_argument("--experiment-mode", "-k", required=True)
    parser.add_argument("--runtime-secs", "-t", default="1200")

    parser.add_argument("--letters-a-path", "-a")
    parser.add_argument("--letters-b-path", "-b")
    parser.add_argument("--letters-c-path", "-c")
    parser.add_argument("--letters-d-path", "-d")

    args = parser.parse_args()

    save_directory = os.path.join(
        OUT_DIR, args.experiment_name.replace("/", "_").replace(".", "_")
    )
    os.makedirs(save_directory, exist_ok=True)
    if args.experiment_mode not in (
        "free_exploration",
        "cost_barrier",
        "reward_incentive",
    ):
        raise ValueError(
            "--experiment-mode must be one of free_exploration, cost_barrier, reward_incentive"
        )
    runtime_seconds = float(args.runtime_secs)

    with open(args.letters_a_path) as f:
        letters_a = json.load(f)
    with open(args.letters_b_path) as f:
        letters_b = json.load(f)
    with open(args.letters_c_path) as f:
        letters_c = json.load(f)
    with open(args.letters_d_path) as f:
        letters_d = json.load(f)

    letter_themes = {
        "A": letters_a,
        "B": letters_b,
        "C": letters_c,
        "D": letters_d,
    }

    agent = ClaudeAgent(
        model=args.model,
        save_directory=save_directory,
        experiment_name=args.experiment_name,
        experiment_mode=args.experiment_mode,
        runtime_seconds=runtime_seconds,
        letter_themes=letter_themes,
    )
    await agent.run_experiment()


if __name__ == "__main__":
    asyncio.run(main())
