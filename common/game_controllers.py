import common.game_constants as game_constants
import common.game_state as game_state
import pygame
import math
import random
import numpy as np


class KeyboardController:
    def GetAction(self, state: game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down
        return action


# ------------------- Q-Learning Controller (no PyTorch) -------------------

class AIController:
    """
    Tabular Q-learning controller with episodic training +
    stronger reward shaping and sensible epsilon schedule.

    State encoding (compact, but informative):
      - Player grid cell (px, py)
      - Goal   grid cell (gx, gy)
      - Nearest enemy relative grid delta (dex, dey), clipped to [-3, 3]

    Action space: 5 (No_action, Up, Down, Left, Right)
    """

    def __init__(self) -> None:
        # Discretization/grid (pixels per cell)
        self.PLAYER_GOAL_GRID = 40   # 800x600 -> (20 x 15) grid
        self.ENEMY_REL_GRID = 40     # coarse relative enemy buckets
        self.ENEMY_REL_CLIP = 3      # clip relative enemy buckets to [-3, 3]

        self.num_actions = 5

        # Q-learning hyperparameters (tuned)
        self.alpha = 0.15         # learning rate (a bit higher to speed up learning)
        self.gamma = 0.97         # discount (value long-term)
        self.epsilon = 1.0        # start exploring
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995  # per-episode decay

        # Q-table as dict: key = state tuple, value = np.array(5)
        self.Q = {}

    # ------------------- State encoding helpers -------------------

    def _center(self, rect):
        return (rect.x + rect.width * 0.5, rect.y + rect.height * 0.5)

    def _nearest_enemy_delta_and_dist(self, state: game_state.GameState):
        px, py = self._center(state.PlayerEntity.entity)
        best_d2 = float("inf")
        best_dx, best_dy = 0.0, 0.0
        for e in state.EnemyCollection:
            ex, ey = self._center(e.entity)
            dx, dy = ex - px, ey - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_dx, best_dy = dx, dy
        return best_dx, best_dy, math.sqrt(best_d2)

    def _encode_state(self, state: game_state.GameState):
        # Player & goal grid cells
        px = int(state.PlayerEntity.entity.x // self.PLAYER_GOAL_GRID)
        py = int(state.PlayerEntity.entity.y // self.PLAYER_GOAL_GRID)
        gx = int(state.GoalLocation.x // self.PLAYER_GOAL_GRID)
        gy = int(state.GoalLocation.y // self.PLAYER_GOAL_GRID)

        # Nearest enemy relative position (bucketized & clipped)
        dx, dy, _ = self._nearest_enemy_delta_and_dist(state)
        dex = int(np.clip(dx // self.ENEMY_REL_GRID, -self.ENEMY_REL_CLIP, self.ENEMY_REL_CLIP))
        dey = int(np.clip(dy // self.ENEMY_REL_GRID, -self.ENEMY_REL_CLIP, self.ENEMY_REL_CLIP))

        return (px, py, gx, gy, dex, dey)

    def _ensure_state(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros(self.num_actions, dtype=np.float32)

    # ------------------- Policy & update -------------------

    def _choose_action_index(self, s_encoded, greedy: bool = False) -> int:
        self._ensure_state(s_encoded)
        if (not greedy) and (random.random() < self.epsilon):
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.Q[s_encoded]))

    def _update_q(self, s_enc, a_idx, r, s2_enc):
        self._ensure_state(s_enc)
        self._ensure_state(s2_enc)
        best_next = np.max(self.Q[s2_enc])
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s_enc][a_idx]
        self.Q[s_enc][a_idx] += self.alpha * td_error

    # ------------------- Reward helpers -------------------

    def _dist_to_goal(self, st: game_state.GameState) -> float:
        (px, py) = self._center(st.PlayerEntity.entity)
        (gx, gy) = self._center(st.GoalLocation)
        return math.hypot(px - gx, py - gy)

    # ------------------- Public API -------------------

    def GetAction(self, state: game_state.GameState) -> game_state.GameActions:
        """
        Used by the game loop to act step-by-step (after training).
        During evaluation, we force greedy actions in EvaluateModel().
        """
        s_enc = self._encode_state(state)
        a_idx = self._choose_action_index(s_enc, greedy=False)
        return game_state.GameActions(a_idx)

    def _GetGreedyAction(self, state: game_state.GameState) -> game_state.GameActions:
        s_enc = self._encode_state(state)
        a_idx = self._choose_action_index(s_enc, greedy=True)
        return game_state.GameActions(a_idx)

    # ------------------- Training & Evaluation -------------------

    def TrainModel(self, episodes: int = 3000, max_steps_per_episode: int = 600):
        """
        Train with clear EPISODES:
          - Each episode starts with a fresh GameState()
          - Episode ends on goal or enemy hit, or when max steps reached
          - Epsilon decays per episode (not every step)
        This dramatically stabilizes learning.
        """
        for _ in range(episodes):
            state = game_state.GameState()
            prev_dist_goal = self._dist_to_goal(state)

            for _t in range(max_steps_per_episode):
                s_enc = self._encode_state(state)
                a_idx = self._choose_action_index(s_enc, greedy=False)
                action = game_state.GameActions(a_idx)

                obs = state.Update(action)
                s2_enc = self._encode_state(state)

                # Base reward per step (slight negative to encourage shorter paths)
                reward = -0.05

                # Move toward goal shaping
                new_dist_goal = self._dist_to_goal(state)
                if new_dist_goal < prev_dist_goal:
                    reward += 0.4
                else:
                    reward -= 0.3
                prev_dist_goal = new_dist_goal

                # Enemy proximity penalty (stay away from nearest enemy)
                _, _, nearest_enemy_dist = self._nearest_enemy_delta_and_dist(state)
                if nearest_enemy_dist < 60:       # very close
                    reward -= 1.0
                elif nearest_enemy_dist < 100:    # moderately close
                    reward -= 0.5

                done = False
                if obs == game_state.GameObservation.Reached_Goal:
                    reward += 100.0
                    done = True
                elif obs == game_state.GameObservation.Enemy_Attacked:
                    reward -= 150.0
                    done = True

                # Q-learning update
                self._update_q(s_enc, a_idx, reward, s2_enc)

                if done:
                    break

            # Epsilon decay per episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                if self.epsilon < self.epsilon_min:
                    self.epsilon = self.epsilon_min

    def EvaluateModel(self):
        """
        Greedy acting (no exploration) for 100,000 steps,
        counting how many times we hit enemies or goals.
        """
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()

        # Force greedy policy during evaluation
        saved_eps = self.epsilon
        self.epsilon = 0.0

        for _ in range(100000):
            action = self._GetGreedyAction(state)
            obs = state.Update(action)
            if obs == game_state.GameObservation.Enemy_Attacked:
                attacked += 1
            elif obs == game_state.GameObservation.Reached_Goal:
                reached_goal += 1

        # restore epsilon
        self.epsilon = saved_eps
        return (attacked, reached_goal)
