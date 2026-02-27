# Project Plan: Multimodal Robotic Blackjack Dealer

## 1. Project Overview & Feasibility Assessment
The goal is to create a robotic Blackjack game where a user interacts via voice commands (bet sizing) and hand gestures (hit, stand, double, split), while a robot acts as the dealer. 

Given the 1-2 week timeline of a college-level introductory robotics course (CS188), we need to carefully scope the project to ensure success.

### 🛑 Out of Scope (Too Difficult for 1-2 Weeks)
- **Physical Card Manipulation (Dealing, Shuffling, Picking Up):** Standard playing cards are thin, deformable, and flat against the table. Grasping them with a standard parallel-jaw gripper is notoriously difficult and prone to failure without building specialized end-effectors (like suction cups) or motion primitives. 
- **Two Physical Robots:** Managing two robots (a dealer and a player robot) introduces significant integration overhead and potential collision issues. 

### ✅ In Scope (Feasible Approach)
- **Virtual Cards / Human Cards:** The card game state will be managed virtually (displayed on a screen) while the robot focuses on the physical interaction of **managing bets and chips**. 
- **Physical Chip Manipulation:** The robot dealer will physically take chips (represented by graspable colored blocks or thick 3D-printed chips) from the user when they lose, and push/place chips to the user when they win.
- **Multimodal Control:** Using highly reliable off-the-shelf ML models to process Voice (for betting) and Gestures (for playing). 
- **Robosuite Simulation (Fallback):** If physical hardware access or calibration takes too long, this entire setup can be implemented in a simulated `robosuite` environment, which perfectly aligns with the course's "Multimodal Control" example project.

---

## 2. System Architecture

The project will consist of three main modules:

### A. Input Processing Module (Human-Robot Interaction)
1. **Voice Recognition (Bets):** Use `SpeechRecognition` library or OpenAI's Whisper API to listen for numbers (e.g., "Bet fifty").
2. **Gesture Recognition (Actions):** Use `MediaPipe Hands` via a webcam to classify static gestures:
   - *Fist* = Stand
   - *Open Palm* = Hit
   - *Two Fingers (Peace Sign)* = Split
   - *Thumbs Up / Pointing* = Double Down

### B. Game State Manager
- A Python state machine that runs the rules of Blackjack.
- Tracks the player's bankroll, current bet, player's hand, and dealer's hand.
- Outputs the physical actions required by the robot (e.g., "Collect 50 chips", "Payout 100 chips").

### C. Robotics Execution Module
- **Environment:** A physical robot arm (or a simulated `robosuite` arm).
- **Objects:** Colored wooden blocks representing chip denominations (e.g., Red = $25, Blue = $50).
- **Actions:** Pick-and-place routines to move chips between the "Dealer Bank" area and the "Player Betting" area based on the Game State Manager's output.

---

## 3. Implementation Timeline (2 Weeks)

### Week 1: Core Logic & Perception
- **Day 1-2:** Implement the Python Blackjack Game State Manager (no robotics yet). 
- **Day 3-4:** Integrate `MediaPipe` for hand gesture recognition and map them to game actions (Hit, Stand, etc.). Test accuracy.
- **Day 5:** Integrate voice recognition for bet sizing. Combine Voice + Gestures + Game State into a playable text/UI-based game.
- **Weekend:** Define the workspace. Decide definitively between a Physical Robot or Robosuite simulation. Set up the environment with "chip" objects.

### Week 2: Robotics Integration & Finalization
- **Day 8-9:** Implement robot kinematics/trajectories for chip manipulation. Write functions for `payout_chips(amount)` and `collect_chips()`.
- **Day 10-11:** End-to-end integration. Connect the HRI (Voice/Camera) to the robot so that completing a hand automatically triggers the robot to move chips.
- **Day 12:** Debugging, edge cases, and robustness testing. 
- **Day 13:** Record the demo video (showing the UI, the user making gestures/speaking, and the robot moving chips).
- **Day 14:** Write the final project report (addressing method, evaluation, and reflections).

---

## 4. Evaluation Metrics for Final Report
To meet the course grading criteria, we will evaluate:
1. **Perception Accuracy:** Success rate of gesture recognition (e.g., over 50 trials, how often did MediaPipe correctly map the gesture to Hit/Stand/Double).
2. **Command NLP Accuracy:** Success rate of parsing voice bets.
3. **Manipulation Success Rate:** How often the robot successfully executes the pick-and-place operation to collect or payout chips without dropping them. 
4. **Latency:** The average time from issuing a gesture to the game state updating and the robot taking action.
