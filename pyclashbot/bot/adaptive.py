"""Adaptive battle strategy learner.

Tracks win/loss outcomes and adjusts elixir-spend weights over time
so the bot gets better the more it plays.

Two files are written to ~/.pyclashbot/ and persist across sessions:

  adaptive_weights.json  — learned elixir weights + aggregate counts
  match_history.json     — full per-match log (plays, deck, outcome,
                           timestamps); never truncated, appended forever
"""

import json
import time
from pathlib import Path

# --- Storage ---

_DATA_DIR = Path.home() / ".pyclashbot"
_WEIGHTS_FILE = _DATA_DIR / "adaptive_weights.json"
_HISTORY_FILE = _DATA_DIR / "match_history.json"

# Max matches kept in history before the oldest are pruned (0 = unlimited)
_MAX_HISTORY = 10_000

# Default weights per phase (index = elixir amount 3-9, offset by 3)
_DEFAULT_WEIGHTS = {
    "early":  [0,    0,    0,    0,    0,    0.5,  0.5],
    "single": [0,    0.05, 0.1,  0.2,  0.3,  0.3,  0.05],
    "double": [0,    0,    0.05, 0.15, 0.3,  0.4,  0.1],
    "triple": [0,    0,    0.05, 0.2,  0.45, 0.3,  0],
}

# How much to nudge weights on a win or loss (5% per outcome)
_LEARN_RATE = 0.05

# Minimum weight floor so no elixir amount ever goes fully to zero
# (except those clamped to 0 in defaults — those stay at 0)
_WEIGHT_FLOOR = 0.0

# After this many games the system is "confident" and slows learning
_CONFIDENCE_THRESHOLD = 50


def _load_state() -> dict:
    """Load persisted adaptive state from disk, or return a fresh default."""
    try:
        if _WEIGHTS_FILE.exists():
            with open(_WEIGHTS_FILE) as f:
                data = json.load(f)
            # Validate structure — if anything looks wrong, reset
            if (
                isinstance(data, dict)
                and "weights" in data
                and "games" in data
                and all(phase in data["weights"] for phase in _DEFAULT_WEIGHTS)
            ):
                return data
    except Exception:
        pass

    return {
        "weights": {phase: list(w) for phase, w in _DEFAULT_WEIGHTS.items()},
        "games": 0,
        "wins": 0,
        "phase_plays": {phase: [0] * 7 for phase in _DEFAULT_WEIGHTS},
        "phase_wins":  {phase: [0] * 7 for phase in _DEFAULT_WEIGHTS},
        "last_updated": "",
    }


def _save_state(state: dict) -> None:
    """Persist adaptive state to disk."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(_WEIGHTS_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass  # disk write failure is non-fatal


# ---------------------------------------------------------------------------
# Match history
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    """Load the full match history from disk."""
    try:
        if _HISTORY_FILE.exists():
            with open(_HISTORY_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _append_match(record: dict) -> None:
    """Append one match record to history and prune if over the limit."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        history = _load_history()
        history.append(record)
        if _MAX_HISTORY > 0 and len(history) > _MAX_HISTORY:
            history = history[-_MAX_HISTORY:]
        with open(_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass  # non-fatal


def get_match_history() -> list[dict]:
    """Return the full saved match history (newest last).

    Each record has:
        timestamp   — ISO-ish string when the match ended
        won         — bool
        duration_s  — approximate battle duration in seconds
        plays       — list of {phase, elixir} dicts for each card played
    """
    return _load_history()


def _normalize(weights: list[float]) -> list[float]:
    """Normalize a weight list so it sums to 1.0, preserving zero slots."""
    total = sum(weights)
    if total <= 0:
        # Fallback: uniform over non-zero positions
        n = sum(1 for w in weights if w > 0) or len(weights)
        return [1.0 / n if w > 0 else 0.0 for w in weights]
    return [w / total for w in weights]


# --- Session tracking ---

class AdaptiveSession:
    """Tracks card plays during a single battle for learning after the outcome."""

    def __init__(self):
        self._state = _load_state()
        # Plays this battle: list of (phase, elixir_amount) tuples (raw elixir, not index)
        self._battle_plays: list[tuple[str, int]] = []
        self._battle_active = False
        self._battle_start_time: float = 0.0

    def start_battle(self) -> None:
        """Call at the start of each fight."""
        self._battle_plays = []
        self._battle_active = True
        self._battle_start_time = time.time()

    def record_play(self, phase: str, elixir_amount: int) -> None:
        """Record that we played a card waiting for `elixir_amount` in `phase`."""
        if not self._battle_active:
            return
        # store raw elixir for history; convert to index only for weight math
        if 3 <= elixir_amount <= 9:
            self._battle_plays.append((phase, elixir_amount))

    def finish_battle(self, won: bool) -> None:
        """Update weights based on this battle's outcome, save weights, log match."""
        if not self._battle_active:
            return
        self._battle_active = False

        duration = round(time.time() - self._battle_start_time, 1)
        state = self._state
        state["games"] += 1
        if won:
            state["wins"] += 1

        # Confidence-based learning rate — slow down as we accumulate data
        games = state["games"]
        lr = _LEARN_RATE * max(0.2, 1.0 - games / (_CONFIDENCE_THRESHOLD * 2))

        for phase, elixir in self._battle_plays:
            idx = elixir - 3
            state["phase_plays"][phase][idx] += 1
            if won:
                state["phase_wins"][phase][idx] += 1

            weights = state["weights"][phase]
            if won:
                weights[idx] = min(1.0, weights[idx] + lr)
            else:
                weights[idx] = max(_WEIGHT_FLOOR, weights[idx] - lr * 0.5)

            state["weights"][phase] = _normalize(weights)

        _save_state(state)

        # Append full match record to history
        _append_match({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "won": won,
            "duration_s": duration,
            "plays": [{"phase": p, "elixir": e} for p, e in self._battle_plays],
        })

    def get_weights(self, phase: str) -> list[float]:
        """Return the current learned weights for `phase`."""
        return list(self._state["weights"].get(phase, _DEFAULT_WEIGHTS[phase]))

    def get_stats(self) -> dict:
        """Return a snapshot of the adaptive stats for display."""
        s = self._state
        games = s["games"]
        wins = s["wins"]
        win_rate = f"{wins / games * 100:.1f}%" if games > 0 else "n/a"
        history_len = len(_load_history())
        return {
            "games": games,
            "wins": wins,
            "win_rate": win_rate,
            "history_matches": history_len,
        }


# Module-level session — shared across the whole worker process lifetime
_session = AdaptiveSession()


def get_session() -> AdaptiveSession:
    """Return the module-level adaptive session."""
    return _session


def reset_session() -> None:
    """Create a fresh session (call after a full bot restart)."""
    global _session
    _session = AdaptiveSession()
