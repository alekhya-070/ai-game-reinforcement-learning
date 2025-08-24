# 🎮 AI Grid World Game with Reinforcement Learning (Q-Learning)

This project implements a simple **grid-based game** where an **AI agent** learns to move around the environment and reach the **food** using **Reinforcement Learning (Q-Learning)**.  

The game is intentionally minimal:
- The agent is a **single block** that can move in four directions.
- The goal is to **reach food placed randomly on the grid**.
- If the agent hits the wall, the game is over.

---

## 🧠 Concepts Explained

### 🎲 The Game
- The game board is a **2D grid**.  
- The agent (a block) can move in **four directions**: up, down, left, right.  
- Food is placed at random locations.  
- The challenge is for the agent to **find the shortest path** to the food without colliding with walls.  

---

### 🤖 Machine Learning in Games
Machine Learning lets programs **learn behaviors automatically** instead of being explicitly coded.  

Here:
- **Environment**: the grid world.  
- **Agent**: the moving block (AI player).  
- **Goal**: maximize score by reaching food quickly.  

---

### 🔁 Reinforcement Learning (RL)
Reinforcement Learning is **learning by trial and error**:
- The agent makes a move (action).  
- The environment responds with a new state and a reward.  
- Over time, the agent improves its strategy to get higher rewards.  

For this game:
- **State**: Agent’s position + Food’s position.  
- **Actions**: Move up, down, left, right.  
- **Rewards**:
  - **+10** → if food is reached.  
  - **-10** → if the agent hits a wall.  
  - **-1** → for each move (to encourage efficiency).  

---

### 🏆 Q-Learning
Q-Learning is the **algorithm** used for training the agent.  

It learns a function `Q(s, a)` = expected reward of taking action `a` in state `s`.  

**Update rule**:
Q(s, a) ← Q(s, a) + α [ r + γ max(Q(s’, a’)) - Q(s, a) ]

markdown
Copy
Edit

- `s`: current state  
- `a`: action taken  
- `r`: reward received  
- `s’`: new state after the action  
- `α`: learning rate (speed of learning)  
- `γ`: discount factor (importance of future rewards)  

After enough training, the agent chooses the **best moves** to reach food quickly and avoid walls.

---

## 🛠️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/alekhya-070/ai-game-reinforcement-learning.git
   cd ai-game-reinforcement-learning
Create a virtual environment (recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ Running the Game
1. Play manually
bash
Copy
Edit
python game.py
Use arrow keys to move the agent.

2. Train the AI agent
bash
Copy
Edit
python train.py
The agent learns with Q-learning.

The trained model is saved to model.pkl.

3. Watch the AI play
bash
Copy
Edit
python agent.py
The agent loads the trained model and plays automatically.

📊 Results
At first, the agent moves randomly.

Over time, it learns to reach food efficiently.

It avoids hitting walls and finds shorter paths.

🔮 Future Improvements
Upgrade to Deep Q-Learning (DQN) for larger grids.

Add multiple food items.

Introduce obstacles for extra difficulty.

Compare performance with other RL algorithms.

✨ Key Learnings
How to represent a simple game as an RL environment.

How Q-Learning works step by step.

How rewards guide learning.

How an AI agent can learn strategies without explicit programming.
