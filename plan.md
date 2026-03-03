# Project Plan: Multimodal Robotic Blackjack Dealer

## 1. Project Overview
A robotic Blackjack game where users interact via **voice commands** (bet sizing) and **hand gestures** (hit, stand, double, split), while a simulated robot arm (`robosuite`) acts as the dealer by manipulating chip objects.

**Team size:** 3 members  
**Timeline:** 2 weeks  
**Course:** CS 188 — Introduction to Robotics (Winter 2026)

---

## 2. System Architecture

```
┌──────────────────────┐
│   Input Processing   │  ← Person A
│  (Voice + Gestures)  │
└────────┬─────────────┘
         │ actions / bets
         ▼
┌──────────────────────┐
│  Game State Manager  │  ← Person B
│   (Blackjack Logic)  │
└────────┬─────────────┘
         │ robot commands
         ▼
┌──────────────────────┐
│  Robotics Execution  │  ← Person C
│  (Chip Manipulation) │
└──────────────────────┘
```

The three modules communicate through **well-defined interfaces** (Python function signatures / message contracts), allowing all three to be developed and tested independently.

---

## 3. Work Assignments

### Person A — Input Processing (Voice + Gesture HRI)
**Goal:** Deliver two Python modules that convert raw sensor input into clean game commands.

| Task | Days | Details |
|------|------|---------|
| Gesture recognition | 1–4 | Use `MediaPipe Hands` via webcam. Classify static poses: tap table → Hit, wave → Stand, peace sign → Double, two hands → Split. Build a small labeled dataset and measure accuracy. |
| Voice recognition | 3–5 | Use `SpeechRecognition` or OpenAI Whisper to parse spoken bet amounts (e.g., "bet fifty" → `50`). Handle noise and edge cases. |
| Unified input API | 5–6 | Expose a clean interface: `get_player_action() → Action` and `get_bet_amount() → int` that Person B can call. |
| Accuracy evaluation | 7 | Run ≥ 50 trials per modality and report accuracy for the final paper. |

**Interface contract (output to Person B):**
```python
class Action(Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"

def get_player_action() -> Action: ...
def get_bet_amount() -> int: ...
```

---

### Person B — Game State Manager (Blackjack Engine + UI)
**Goal:** A fully testable Blackjack state machine and a visual display for the game.

| Task | Days | Details |
|------|------|---------|
| Core Blackjack logic | 1–3 | Implement dealing, hand evaluation, hit/stand/double/split rules, dealer AI (hit on soft 17), bankroll tracking. Write unit tests. |
| Game display / UI | 3–5 | Build a simple screen display (Pygame, terminal, or web) showing: player hand, dealer hand, current bet, bankroll. |
| Robot command API | 4–6 | After each hand resolves, emit robot commands: `payout_chips(amount)` or `collect_chips(amount)` that Person C implements. |
| Integration glue | 7–8 | Wire Person A's input API and Person C's robot API into the game loop. Handle error/retry logic. |

**Interface contract (output to Person C):**
```python
class RobotCommand:
    action: str   # "payout" or "collect"
    amount: int   # chip value

def on_hand_resolved(result) -> RobotCommand: ...
```

---

### Person C — Robotics Execution (Chip Manipulation in Robosuite)
**Goal:** A `robosuite` simulated robot arm that picks and places chip objects on command.

| Task | Days | Details |
|------|------|---------|
| Robosuite environment setup | 1–2 | Set up `robosuite` simulation with a robot arm (e.g., Panda). Create a custom task environment with chip objects (colored blocks) in Dealer Bank and Player Betting zones. Define coordinate frames and camera views. |
| Pick-and-place routines | 2–5 | Implement `payout_chips(amount)` (move chips from bank → player) and `collect_chips(amount)` (player → bank). Tune grasping, trajectories, and release in simulation. |
| Robustness & edge cases | 5–7 | Handle multi-chip payouts (break amount into denominations), retry on grasp failure, speed optimization. Render the simulation view for the demo video. |
| Manipulation eval | 7 | Run ≥ 30 simulated manipulation trials; measure success rate and latency for the final paper. |

**Interface contract (input from Person B):**
```python
def payout_chips(amount: int) -> bool: ...
def collect_chips(amount: int) -> bool: ...
```

---

## 4. Integration Milestones

| Milestone | Target Day | Description |
|-----------|-----------|-------------|
| **M1: Interfaces frozen** | Day 3 | All three agree on the exact Python function signatures above. Stub implementations checked in so everyone can develop against them. |
| **M2: Module demos** | Day 7 | Each person demos their module independently (A: gesture/voice accuracy, B: game plays correctly with keyboard input, C: robosuite arm moves chips on command). |
| **M3: End-to-end integration** | Day 9 | Wire all modules together. Play a full hand: voice bet → gesture actions → game resolution → robot moves chips. |
| **M4: Polish & record** | Day 11–12 | Debug edge cases, stress test, record the demo video. |
| **M5: Deliverables** | Day 13–14 | Write final report, build project website, clean up source code & README. |

---

## 5. Shared Responsibilities (Days 11–14)

These tasks are done together and should be split evenly:

- [ ] **Demo video** — Record a polished video showing the full pipeline in action.
- [ ] **Project website** — Clear, visually engaging summary with images and the demo video.
- [ ] **Final report** — Each person writes their own module section; one person assembles and edits.
- [ ] **README & code cleanup** — Ensure the repo is well-documented and reproducible.

---

## 6. Evaluation Metrics (for Report)

| Metric | Owner | Method |
|--------|-------|--------|
| Gesture recognition accuracy | A | ≥ 50 trials, report % correct per gesture |
| Voice command accuracy | A | ≥ 50 trials, report % correct bets parsed |
| Manipulation success rate (sim) | C | ≥ 30 simulated pick-and-place trials, report % success |
| End-to-end latency | All | Time from gesture → game update → robot action |

