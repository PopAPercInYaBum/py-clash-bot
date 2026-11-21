"""random module for randomizing fight plays"""

import collections
import random
import time
from typing import Literal

from pyclashbot.bot.card_detection import (
    calculate_play_coords,
    check_which_cards_are_available,
    create_default_bridge_iar,
    get_card_group,
    get_play_coords_for_card,
    identify_hand_cards,
    is_hero_champion_ability_visible,
    switch_side,
    trigger_hero_champion_ability,
)
from pyclashbot.bot.coords import (
    CLOSE_BATTLE_LOG_BUTTON,
    EMOTE_BUTTON_COORD,
    EMOTE_ICON_COORDS,
    HAND_CARDS_COORDS,
    PLAYABLE_PLAY_REGION_LTRB,
    QUICKMATCH_POPUP_BUTTON_COORD,
    START_FIGHT_BUTTON_COORD,
)
from pyclashbot.bot.nav import (
    check_for_in_battle_with_delay,
    get_to_activity_log,
    get_to_main_after_fight,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from pyclashbot.bot.recorder import (
    finish_fight_recording,
    is_recording,
    log_play,
    start_fight_recording,
    stop_fight_capture,
)
from pyclashbot.bot.state_detect import (
    check_if_battle_has_ended,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    check_pixels_for_win_in_battle_log,
    count_elixir,
)
from pyclashbot.bot.adaptive import get_session as _get_adaptive_session
from pyclashbot.utils.logger import Logger
from pyclashbot.utils.versioning import __version__

ELIXIR_WAIT_TIMEOUT = 40  # too high but someone got errors with that so idk

# ---------------------------------------------------------------------------
# Counter-push / combat state machine
# ---------------------------------------------------------------------------

# Card groups that are primarily defensive responses to an opponent push
_DEFENSIVE_GROUPS: frozenset[str] = frozenset({
    "defense_building",
    "reactive_spell",
    "spirit",
})
# Back-support cards used as defensive answers (minions, bats, skeleton army, etc.)
_DEFENSIVE_BACK_SUPPORT: frozenset[str] = frozenset({
    "bats", "evo_bats", "minions", "minion_horde", "evo_minion_horde",
    "skeleton_army", "evo_skeleton_army", "skeletons", "evo_skeletons",
    "goblins", "hero_goblins", "goblin_gang", "archers", "evo_archers",
    "valkyrie", "evo_valkyrie", "ice_wizard", "electro_wizard",
    "hunter", "evo_hunter", "wizard", "evo_wizard", "hero_wizard",
    "baby_dragon", "evo_baby_dragon", "bomber", "evo_bomber",
    "tombstone", "hero_tombstone", "elixir_collector",
    "battle_healer", "dart_goblin", "evo_dart_goblin",
    "flying_machine", "musketeer", "evo_musketeer", "hero_musketeer",
})

# Card groups that form the front of a push (tanks, win-cons at the bridge)
_PUSH_GROUPS: frozenset[str] = frozenset({
    "bridge_line",
    "bridge_rush",
    "miner",
    "goblin_barrel",
    "goblin_drill",
    "graveyard",
    "siege_building",
})

# Groups that provide support behind a push
_SUPPORT_GROUPS: frozenset[str] = frozenset({
    "back_support",
    "king_lane",
    "princess",
})

# Pure spell groups (don't change combat phase)
_SPELL_GROUPS: frozenset[str] = frozenset({
    "lane_spell",
    "large_spell",
    "center_spell",
    "rocket",
    "tornado",
})


def _get_card_role(card_id: str) -> str:
    """Classify a card_id into a broad combat role used by CombatStateTracker.

    Returns one of: "push", "defend", "spell", "support".
    """
    group = get_card_group(card_id)
    if group in _PUSH_GROUPS:
        return "push"
    if group in _DEFENSIVE_GROUPS:
        return "defend"
    if group in _SPELL_GROUPS:
        return "spell"
    if group == "back_support" and card_id in _DEFENSIVE_BACK_SUPPORT:
        return "defend"
    return "support"


class CombatStateTracker:
    """Tracks the current combat phase and guides card selection.

    Phases
    ------
    defend       — opponent is pushing; play defensive cards on the pushed lane.
    counter_push — we just stopped a push; tank + support on the same lane.
    push         — quiet field; take the initiative with tanks and win-cons.
    """

    # action_offset from switch_side() above this → opponent is pushing hard.
    # Calibrated from real game logs: normal background noise is 15k–23k,
    # so a genuine heavy push registers well above that.
    _PUSH_THRESHOLD = 28000
    # below this after defending → field has quieted enough to counter-push
    _SAFE_THRESHOLD = 18000
    # how many consecutive defensive plays before we consider the push stopped
    _DEFENSE_PLAYS_NEEDED = 2
    # how many push plays before we move from counter_push → push (sustained pressure)
    _COUNTER_PUSH_PLAYS = 3
    # max consecutive defensive plays before forcing a counter-push regardless of offset.
    # prevents the bot from defending forever if the threshold never dips.
    _MAX_DEFEND_PLAYS = 5

    def __init__(self) -> None:
        # Start aggressive — take the initiative, don't wait to react
        self.mode: str = "push"
        self.active_side: str = "left"
        self._defense_count: int = 0
        self._push_count: int = 0

    def update(self, action_offset: float, side: str) -> None:
        """Call at the start of each play decision with the latest switch_side() output."""
        if action_offset >= self._PUSH_THRESHOLD:
            # Opponent is actively pushing — switch to defend if not already
            if self.mode != "defend":
                self.mode = "defend"
                self._defense_count = 0
                self._push_count = 0
            self.active_side = side
        elif (
            self.mode == "defend"
            and (
                action_offset <= self._SAFE_THRESHOLD
                or self._defense_count >= self._MAX_DEFEND_PLAYS
            )
            and self._defense_count >= self._DEFENSE_PLAYS_NEEDED
        ):
            # Field quieted (or we've defended long enough) — time to counter-push
            self.mode = "counter_push"
            self.active_side = side  # push the lane we just defended
            self._push_count = 0

    def record_play(self, role: str) -> None:
        """Update internal counters after a card is played."""
        if role == "defend":
            self._defense_count += 1
            self._push_count = 0
        elif role in ("push", "support"):
            self._push_count += 1
            self._defense_count = 0
            if self.mode == "counter_push" and self._push_count >= self._COUNTER_PUSH_PLAYS:
                self.mode = "push"
        # spells: no phase change

    def preferred_roles(self) -> list[str]:
        """Return card roles in preference order for the current phase."""
        if self.mode == "defend":
            return ["defend", "spell", "support", "push"]
        if self.mode == "counter_push":
            return ["push", "support", "defend", "spell"]
        # push
        return ["push", "support", "spell", "defend"]


def select_card_smart(
    emulator,
    card_indices: list[int],
    combat_tracker: CombatStateTracker,
    last_played: "collections.deque[int]",
) -> tuple[int, str]:
    """Pick a card index biased toward the roles preferred in the current combat phase.

    Identifies all available cards (one color-fingerprint call per slot using the
    already-cached battle_iar), then selects the best role match while also avoiding
    recently played slots for variety.

    Starvation prevention: any slot that has been skipped _STARVATION_LIMIT times in a
    row is forced to play next, regardless of role, so no slot ever gets permanently
    ignored by the role-priority loop.

    Returns (card_index, card_id).
    """
    # Identify every available card in one pass (uses cached battle_iar — no ADB round-trip)
    identified: dict[int, str] = {idx: identify_hand_cards(emulator, idx) for idx in card_indices}

    # --- Starvation override ---
    # If any available slot has gone _STARVATION_LIMIT plays without being chosen, force it.
    starved = [
        idx for idx in card_indices
        if _slot_starvation.get(idx, 0) >= _STARVATION_LIMIT
    ]
    if starved:
        chosen = random.choice(starved)
        return chosen, identified[chosen]

    preferred = combat_tracker.preferred_roles()

    # Force cycle variety: if the same card has been played 3+ consecutive times,
    # skip cards that were in the last 2 plays entirely for this round so the bot
    # doesn't just hammer one card repeatedly.
    recently_played = set(list(last_played)[-2:]) if len(last_played) >= 2 else set()
    force_variety = len(last_played) >= 3 and len(set(last_played)) == 1  # all same card

    # Try each preferred role; prefer slots not played recently
    for role in preferred:
        candidates = [
            idx for idx, cid in identified.items()
            if _get_card_role(cid) == role
        ]
        # In forced-variety mode, skip recently played slots; fall back if no other options
        fresh = [idx for idx in candidates if idx not in last_played]
        if force_variety:
            very_fresh = [idx for idx in candidates if idx not in recently_played]
            if very_fresh:
                chosen = random.choice(very_fresh)
                return chosen, identified[chosen]
        if fresh:
            chosen = random.choice(fresh)
            return chosen, identified[chosen]
        # Accept a recently-played slot if nothing fresher matches
        if candidates:
            chosen = random.choice(candidates)
            return chosen, identified[chosen]

    # Fallback: any card not recently played, then any card
    fresh_any = [idx for idx in card_indices if idx not in last_played]
    chosen = random.choice(fresh_any) if fresh_any else random.choice(card_indices)
    return chosen, identified[chosen]


def play_a_card_smart(
    emulator,
    logger,
    recording_flag: bool,
    battle_strategy: "BattleStrategy",
    combat_tracker: CombatStateTracker,
) -> bool:
    """Smart card play that uses CombatStateTracker to pick the right card for the moment.

    Replaces play_a_card() in the main fight loop. Card selection is biased toward:
    - Defensive cards (buildings, swarms) when the opponent is pushing
    - Tanks and win-cons (bridge_rush / bridge_line) on a counter-push
    - Mixed push lineup when the field is quiet

    Placement still uses the existing calculate_play_coords() / PLAY_COORDS system so
    each card type goes to its tuned board position.
    """
    # Trigger champion ability opportunistically before the main play
    if is_hero_champion_ability_visible(emulator):
        logger.change_status("Hero/Champion ability ready — triggering")
        trigger_hero_champion_ability(emulator, logger)

    # Get available cards FIRST; this sets the battle_iar global that switch_side() requires.
    # check_side=True also updates the module-level play_side global so
    # calculate_play_coords() will use the correct lane automatically.
    card_indices = check_which_cards_are_available(emulator, check_side=True)
    if not card_indices:
        logger.change_status("No cards ready yet...")
        return False

    # Now safe to read battlefield activity — battle_iar is set.
    action_offset, side = switch_side()
    combat_tracker.update(action_offset, side)

    card_index, card_id = select_card_smart(emulator, card_indices, combat_tracker, last_three_cards)
    if card_index not in last_three_cards:
        last_three_cards.append(card_index)

    # Update starvation counters: increment every available slot that was NOT chosen,
    # and reset the chosen slot to 0.
    for idx in card_indices:
        if idx == card_index:
            _slot_starvation[idx] = 0
        else:
            _slot_starvation[idx] = _slot_starvation.get(idx, 0) + 1

    role = _get_card_role(card_id)
    group = get_card_group(card_id)

    # For counter-push/push modes, force the card to the active side so we push
    # the lane our defenders walked from.  For defend mode, play_side from
    # switch_side() already points at the opponent's push lane.
    elapsed = battle_strategy.get_elapsed_time()
    if combat_tracker.mode in ("counter_push", "push"):
        play_coord = calculate_play_coords(group, combat_tracker.active_side, elapsed)
    else:
        # defend — use the auto-detected side from check_which_cards_are_available
        play_coord = calculate_play_coords(group, side, elapsed)

    # Fallback: re-derive via full get_play_coords_for_card if coords came back None
    if play_coord is None:
        _, play_coord = get_play_coords_for_card(emulator, logger, card_index, elapsed)

    # In push/counter_push mode, shift back_support cards forward so they
    # support the push from mid-field rather than sitting behind our tower.
    # back_support coords default to y=440-456 (behind king tower); shift to
    # y≈370-386 so musketeer/archers/etc walk into the action sooner.
    if (
        play_coord is not None
        and combat_tracker.mode in ("push", "counter_push")
        and group in ("back_support", "king_lane")
    ):
        x, y = play_coord
        y = max(y - 70, 350)  # don't go above bridge line (y<300)
        play_coord = (x, y)

    logger.change_status(
        f"[{combat_tracker.mode.upper()}] {card_id} ({role}) → {play_coord} | "
        f"offset={int(action_offset)} side={side}"
    )

    # Click the card then the board position
    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])
    if play_coord is None:
        logger.change_status("Non-fatal: play coord is None — skipping play")
        return False
    emulator.click(play_coord[0], play_coord[1])

    combat_tracker.record_play(role)

    if recording_flag:
        log_play(card_index, play_coord[0], play_coord[1], elapsed)

    logger.add_card_played()

    if random.randint(0, 9) == 1:
        send_emote(emulator, logger)

    return True
ABILITY_TRIGGER_DELAY_S = 3


def _maybe_start_fight_recording(
    emulator, logger, recording_flag: bool, fight_mode_chosen: str, custom_path: str | None = None
) -> None:
    """Begin opt-in training-data capture for 1v1-type fights only (Trophy Road / Classic 1v1, never 2v2)."""
    if recording_flag and fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
        pack_mode = "1v1_trophy" if fight_mode_chosen == "Trophy Road" else "1v1_classic"
        start_fight_recording(emulator, pack_mode, __version__, logger=logger, custom_path=custom_path)


def do_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode,
    fight_mode_chosen,
    called_from_launching=False,
    recording_flag: bool = False,
    custom_path: str | None = None,
) -> bool:
    """Handle the entirety of a battle state (start fight, do fight, end fight)."""

    logger.change_status("Waiting for battle to start")

    # Wait for battle start
    if wait_for_battle_start(emulator, logger) is False:
        logger.change_status("Timed out waiting for battle to start")
        return False

    logger.change_status("Starting fight loop")
    logger.log(f'This is the fight mode: "{fight_mode_chosen}"')

    # Recording is started/stopped inside the fight loops themselves so each pack
    # brackets exactly the in-battle play loop (no pre-battle wait or post-fight nav).
    # Run regular fight loop if random mode not toggled
    if not random_fight_mode and _fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path) is False:
        logger.change_status("Fight loop failed")
        return False

    # Run random fight loop if random mode toggled
    if (
        random_fight_mode
        and _random_fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path) is False
    ):
        logger.change_status("Fight loop failed")
        return False

    # Only log the fight if not called from the start
    if not called_from_launching:
        if fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
            logger.add_1v1_fight()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_2v2_fights()

        if fight_mode_chosen == "Trophy Road":
            logger.increment_trophy_road_fights()
        elif fight_mode_chosen == "Classic 1v1":
            logger.increment_classic_1v1_fights()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_classic_2v2_fights()

    time.sleep(10)
    return True


def start_fight(emulator, logger, mode) -> bool:
    """Start a fight with the specified mode.

    Args:
        emulator: The emulator controller
        logger: Logger instance
        mode: Fight mode - must be one of "Classic 1v1", "Classic 2v2", or "Trophy Road"

    Returns:
        bool: True if fight started successfully, False otherwise
    """
    # Validate mode parameter
    logger.log(f'Input mode type: "{type(mode)}"')
    logger.log(f"Input mode value: {mode}")
    valid_modes = ["Classic 1v1", "Classic 2v2", "Trophy Road"]
    logger.log(f"Valid modes: {valid_modes}")
    if mode not in valid_modes:
        logger.log(f"The valid modes for start_fight() are: {valid_modes}")
        logger.log(f"But start_fight() got an invalid mode: '{mode}'")
        return False

    logger.change_status(f"Starting a {mode} fight")

    # Check if on clash main menu
    logger.log("Checking if on main menu before starting fight...")
    if not check_if_on_clash_main_menu(emulator):
        logger.change_status("Not on main menu — cannot start fight")
        return False

    # For all modes (1v1 and 2v2), use the same start button
    # Mode is already set by select_mode() in states.py, just click start button
    emulator.click(START_FIGHT_BUTTON_COORD[0], START_FIGHT_BUTTON_COORD[1])
    logger.log(f"Clicked Start button at {START_FIGHT_BUTTON_COORD}")

    # 2v2 needs a second popup after Start
    if mode == "Classic 2v2":
        logger.change_status("Classic 2v2 — clicking Quick Match popup...")
        time.sleep(3)
        emulator.click(QUICKMATCH_POPUP_BUTTON_COORD[0], QUICKMATCH_POPUP_BUTTON_COORD[1])
        logger.log(f"Clicked Quickmatch button at {QUICKMATCH_POPUP_BUTTON_COORD}")

    return True


def send_emote(emulator, logger: Logger):
    """Method to do an emote in a fight"""
    logger.change_status("Sending emote")

    # click emote button
    emulator.click(EMOTE_BUTTON_COORD[0], EMOTE_BUTTON_COORD[1])
    time.sleep(0.33)

    emote_coord = random.choice(EMOTE_ICON_COORDS)
    emulator.click(emote_coord[0], emote_coord[1])


RANDOM_PLAY_ELIXIR_MIN = 3
RANDOM_PLAY_ELIXIR_MAX = 9


def play_random_available_card(emulator, logger, recording_flag: bool, elapsed_s: float) -> bool:
    """Play one card that is actually available, at a random friendly-half coord.

    Mirrors the rl-bot RandomPlayer so recorded plays are never polluted with cards
    that didn't deploy: read the available hand slots (affordable, non-empty) and, if
    any, play a random one; if none are available, do nothing. Returns True if a card
    was played. The caller gates this behind a random elixir wait.
    """
    card_indices = check_which_cards_are_available(emulator, check_side=True)
    if not card_indices:
        return False  # nothing playable -> no play, nothing recorded (caller re-rolls)

    card_index = random.choice(card_indices)
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    play_coord = (random.randint(left, right), random.randint(top, bottom))

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])
    time.sleep(0.1)
    emulator.click(play_coord[0], play_coord[1])
    time.sleep(0.1)

    if recording_flag:
        # Card identity isn't computed for random plays; home re-derives it from pixels.
        log_play(card_index, play_coord[0], play_coord[1], elapsed_s)
    logger.add_card_played()
    return True


def wait_for_elixir(
    emulator,
    logger,
    elixir_wait_amount,
    WAIT_THRESHOLD=5000,  # noqa: N803
    PLAY_THRESHOLD=10000,  # noqa: N803
    recording_flag: bool = False,
) -> Literal["restart", "no battle"] | bool:
    """Method to wait for 4 elixir during a battle"""
    start_time = time.time()
    battle_detection_lost_count = 0
    last_logged_second = -1
    last_lost_detection_log_second = -1
    ability_available_since = None

    while not count_elixir(emulator, elixir_wait_amount):
        # debug screenshot saving removed from production
        wait_time = time.time() - start_time
        elapsed_second = int(wait_time)
        if elapsed_second != last_logged_second:
            logger.change_status(
                f"Waiting for {elixir_wait_amount} elixir for {elapsed_second}s...",
            )
            last_logged_second = elapsed_second

        card_indices = check_which_cards_are_available(emulator)
        if is_hero_champion_ability_visible(emulator):
            if ability_available_since is None:
                ability_available_since = time.time()
                logger.change_status(
                    f"Hero/Champion ability ready — triggering in {ABILITY_TRIGGER_DELAY_S}s",
                )
            elif time.time() - ability_available_since >= ABILITY_TRIGGER_DELAY_S:
                trigger_hero_champion_ability(emulator, logger)
                ability_available_since = None
        else:
            ability_available_since = None

        card_inhand = len(card_indices)
        action_offset, _ = switch_side()
        if action_offset > PLAY_THRESHOLD and card_inhand > 0:
            logger.change_status("Battle too active — playing now")
            return True

        if action_offset > WAIT_THRESHOLD and card_inhand == 4:
            logger.change_status("All cards are available!")
            return True

        if wait_time > ELIXIR_WAIT_TIMEOUT:
            logger.change_status(status="Waited too long for elixir")
            return "restart"

        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status(status="Battle ended — stopping elixir wait")
                return "no battle"

            battle_detection_lost_count += 1
            lost_detection_second = int(time.time())
            if lost_detection_second != last_lost_detection_log_second:
                logger.change_status(
                    status="Lost battle detection while waiting for elixir — assuming still in battle",
                )
                last_lost_detection_log_second = lost_detection_second
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    status="Lost battle detection repeatedly — assuming battle ended",
                )
                return "no battle"

            time.sleep(0.5)
            continue

        battle_detection_lost_count = 0

    logger.change_status(
        f"Took {str(time.time() - start_time)[:4]}s for {elixir_wait_amount} elixir.",
    )

    return True


def end_fight_state(
    emulator,
    logger: Logger,
    recording_flag,
    disable_win_tracker_toggle=True,
):
    """Method to handle the time after a fight and before the next state"""
    # count the crown score on this end-battle screen

    # get to clash main after this fight
    logger.log("Returning to main menu after fight")
    if get_to_main_after_fight(emulator, logger) is False:
        logger.log("Failed to return to main menu after fight")
        finish_fight_recording(None)
        return False

    logger.log("Returned to main menu after fight")
    time.sleep(3)

    # Determine the outcome. Always check so the adaptive learner can record it;
    # the user's win-tracker toggle only controls whether stats are incremented.
    win_check_return = check_if_previous_game_was_win(emulator, logger)

    if win_check_return == "restart":
        logger.log("Failed while checking if previous game was a win")
        _get_adaptive_session().finish_battle(False)  # count as loss on failure
        finish_fight_recording(None)
        return False

    outcome = "win" if win_check_return else "loss"

    # Feed the outcome to the adaptive learner so it can adjust weights.
    adaptive = _get_adaptive_session()
    adaptive.finish_battle(bool(win_check_return))
    stats = adaptive.get_stats()
    logger.log(
        f"[Adaptive] game {stats['games']} | wins {stats['wins']} | win-rate {stats['win_rate']}"
    )

    # Only touch the user's win/loss stats when their tracker is enabled.
    if not disable_win_tracker_toggle:
        if win_check_return:
            logger.add_win()
        else:
            logger.add_loss()

    finish_fight_recording(outcome if (is_recording() or not disable_win_tracker_toggle) else None)

    return True


def check_if_previous_game_was_win(
    emulator,
    logger: Logger,
) -> bool | Literal["restart"]:
    """Method to handle the checking if the previous game was a win or loss"""
    logger.change_status(status="Checking last game result")

    # Use wait_for_clash_main_menu to ensure we are on the main menu.
    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        logger.change_status(status="Not on main menu — cannot check last game result")
        return "restart"

    # get to clash main options menu
    if get_to_activity_log(emulator, logger, printmode=False) == "restart":
        logger.change_status(status="Failed to open battle log")

        return "restart"

    logger.change_status(status="Checking battle log for win...")
    is_a_win = check_pixels_for_win_in_battle_log(emulator)
    result = "win" if is_a_win else "loss"
    logger.change_status(status=f"Last game result: {result}")

    # close battle log
    logger.change_status(status="Returning to main menu")
    emulator.click(CLOSE_BATTLE_LOG_BUTTON[0], CLOSE_BATTLE_LOG_BUTTON[1])
    if wait_for_clash_main_menu(emulator, logger) is False:
        logger.change_status(status="Timed out returning to main menu after battle log")
        return "restart"
    time.sleep(2)

    return is_a_win


# main fight loops

# Initialize a deque with a maximum length of 3 to store the last three chosen cards
last_three_cards = collections.deque(maxlen=3)

# Starvation counter: tracks how many consecutive plays each card slot has been skipped.
# Reset every fight. When a slot reaches _STARVATION_LIMIT it is forced to play next
# regardless of role preference, so no slot can be permanently ignored.
_slot_starvation: dict[int, int] = {}
_STARVATION_LIMIT = 4


def select_card_index(card_indices, last_three_cards):
    if not card_indices:
        raise ValueError("card_indices cannot be empty")

    # First preference: Cards not in the last_three_cards queue
    preferred_cards = [index for index in card_indices if index not in last_three_cards]

    # Second preference: Cards not among the last two added to the queue
    if not preferred_cards and len(last_three_cards) == 3:
        preferred_cards = [index for index in card_indices if index not in list(last_three_cards)[-2:]]

    # Third preference: Any card except the most recently added one
    if not preferred_cards and last_three_cards:
        preferred_cards = [index for index in card_indices if index != last_three_cards[-1]]

    # Fallback: If all else fails, consider all cards
    if not preferred_cards:
        preferred_cards = card_indices

    return random.choice(preferred_cards)


def play_a_card(emulator, logger, recording_flag: bool, battle_strategy: "BattleStrategy") -> bool:
    print("\n")

    # Fire hero/champion ability opportunistically before playing a card.
    # wait_for_elixir also does this, but if elixir was already above the target
    # the wait loop is skipped entirely and the ability would never trigger.
    if is_hero_champion_ability_visible(emulator):
        logger.change_status("Hero/Champion ability ready — triggering")
        trigger_hero_champion_ability(emulator, logger)

    # check which cards are available
    logger.change_status("Looking at which cards are available")
    available_card_check_start_time = time.time()
    card_indices = check_which_cards_are_available(emulator, check_side=True)

    if not card_indices:
        logger.change_status("No cards ready yet...")
        return False

    available_card_check_time_taken = str(
        time.time() - available_card_check_start_time,
    )[:3]

    logger.change_status(
        f"These cards are available: {card_indices} ({available_card_check_time_taken}s)",
    )

    card_index = select_card_index(card_indices, last_three_cards)
    if card_index not in last_three_cards:
        last_three_cards.append(card_index)
    logger.change_status(f"Choosing this card index: {card_index}")

    # get a coord based on the selected side
    play_coord_calculation_start_time = time.time()
    card_id, play_coord = get_play_coords_for_card(emulator, logger, card_index, battle_strategy.get_elapsed_time())
    play_coord_calculation_time_taken = str(
        time.time() - play_coord_calculation_start_time,
    )[:3]

    logger.change_status(
        f"Calculated play for: {card_id} at {play_coord} ({play_coord_calculation_time_taken}s)",
    )

    # click the card index
    click_and_play_card_start_time = time.time()
    if None in [HAND_CARDS_COORDS, card_index]:
        logger.change_status("Non-fatal error: card index is None")
        return False

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])

    # click the play coord
    if play_coord is None:
        logger.change_status("Non-fatal error: play coordinates are None")
        return False

    emulator.click(play_coord[0], play_coord[1])
    click_and_play_card_time_taken = str(time.time() - click_and_play_card_start_time)[:3]
    if recording_flag:
        log_play(card_index, play_coord[0], play_coord[1], battle_strategy.get_elapsed_time())

    logger.change_status(f"Made the play {click_and_play_card_time_taken}s")
    logger.add_card_played()

    if random.randint(0, 9) == 1:
        send_emote(emulator, logger)
    return True


_PHASE_STRATEGIES = {
    # 0-10s: Never play — let first elixir cycle build up before committing a card.
    "early": [0, 0, 0, 0, 0, 0.5, 0.5],
    # 10-90s (single elixir): Wait for 6-8 elixir; occasional 5 to keep pressure.
    "single": [0, 0.05, 0.1, 0.2, 0.3, 0.3, 0.05],
    # 90-180s (double elixir): Elixir comes faster — target 6-8 more aggressively.
    "double": [0, 0, 0.05, 0.15, 0.3, 0.4, 0.1],
    # 180s+ (triple elixir / sudden death): Dump everything; target 6-7 for max plays/min.
    "triple": [0, 0, 0.05, 0.2, 0.45, 0.3, 0],
}

_PHASE_THRESHOLDS_1V1 = {
    "early": (6000, 9000),
    "single": (6000, 9000),
    "double": (7000, 10000),
    "triple": (8000, 11000),
}

_PHASE_THRESHOLDS_2V2 = {
    "early": (7000, 13000),
    "single": (7000, 14000),
    "double": (8000, 15000),
    "triple": (9000, 16000),
}


class BattleStrategy:
    """Manages battle timing and elixir selection strategy.

    Encapsulates the sophisticated elixir selection logic that changes
    based on battle phase, eliminating the need for global variables.
    """

    def __init__(self, fight_mode: str = "Classic 1v1"):
        self.start_time = None
        self.elixir_amounts = [3, 4, 5, 6, 7, 8, 9]

        is_2v2 = fight_mode == "Classic 2v2"
        self.phase_strategies = _PHASE_STRATEGIES
        self.phase_thresholds = _PHASE_THRESHOLDS_2V2 if is_2v2 else _PHASE_THRESHOLDS_1V1

    def start_battle(self):
        """Call when battle begins to start timing."""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Get seconds elapsed since battle start."""
        return time.time() - self.start_time if self.start_time else 0

    def get_battle_phase(self):
        """Determine current battle phase based on elapsed time.

        Clash Royale timing (3-minute match):
          0-10s   → early      (let first cycle fill before spending)
          10-90s  → single     (normal 2.8 elixir/s)
          90-180s → double     (double elixir, 5.6 elixir/s)
          180s+   → triple     (triple elixir / sudden death, 8.4 elixir/s)
        """
        elapsed = self.get_elapsed_time()
        if elapsed < 10:
            return "early"
        elif elapsed < 90:
            return "single"
        elif elapsed < 180:
            return "double"
        else:
            return "triple"

    def select_elixir_amount(self):
        """Select elixir amount to wait for based on current battle phase.

        Uses adaptive learned weights when available, blended with the base
        strategy so the bot gets smarter the more games it plays.
        """
        phase = self.get_battle_phase()
        base_weights = self.phase_strategies[phase]

        # Pull learned weights from the adaptive session and blend them with
        # the base strategy (60/40 blend so the bot stays sane early on).
        adaptive = _get_adaptive_session()
        learned = adaptive.get_weights(phase)
        blended = [0.6 * b + 0.4 * l for b, l in zip(base_weights, learned)]

        elixir = random.choices(self.elixir_amounts, weights=blended, k=1)[0]

        # Record this play choice so the learner can credit it after the battle.
        adaptive.record_play(phase, elixir)

        return elixir

    def get_thresholds(self):
        """Get (WAIT_THRESHOLD, PLAY_THRESHOLD) for current battle phase."""
        phase = self.get_battle_phase()
        return self.phase_thresholds[phase]


def _fight_loop(
    emulator, logger: Logger, recording_flag: bool, fight_mode: str = "Classic 1v1", custom_path: str | None = None
) -> bool:
    """Method for handling dynamically timed fight"""
    create_default_bridge_iar(emulator)
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode, custom_path)
    # Reset the card-variety tracker and starvation counters at the start of every
    # fight so we don't carry over stale history from the previous match.
    last_three_cards.clear()
    _slot_starvation.clear()
    prev_cards_played = logger.get_cards_played()
    battle_detection_lost_count = 0

    # Initialize battle strategy and start timing
    battle_strategy = BattleStrategy(fight_mode)
    battle_strategy.start_battle()

    # Initialize counter-push combat state tracker
    combat_tracker = CombatStateTracker()

    # Notify the adaptive learner that a new battle is starting so it can
    # track elixir choices for this game independently.
    _get_adaptive_session().start_battle()

    # Refresh bridge_iar every N seconds so switch_side() stays accurate as
    # troops move across the bridge during the match.
    _BRIDGE_IAR_REFRESH_INTERVAL_S = 15
    last_bridge_refresh = time.time()

    while True:
        # Periodically refresh the bridge baseline image for side-switching accuracy.
        if time.time() - last_bridge_refresh >= _BRIDGE_IAR_REFRESH_INTERVAL_S:
            create_default_bridge_iar(emulator)
            last_bridge_refresh = time.time()

        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}) — waiting it out",
            )

            # If we've lost detection several times in a row, assume the battle
            # ended even if we couldn't confirm it (prevents infinite loops if UI changes).
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly — assuming battle ended",
                )
                break

            time.sleep(1)
            continue

        battle_detection_lost_count = 0
        # debug screenshot saving removed from production

        # Get elixir amount and thresholds based on current battle phase
        elixir_amount = battle_strategy.select_elixir_amount()
        wait_threshold, play_threshold = battle_strategy.get_thresholds()

        wait_output = wait_for_elixir(
            emulator,
            logger,
            elixir_amount,
            wait_threshold,
            play_threshold,
            recording_flag,
        )

        if wait_output == "restart":
            logger.change_status("Failed while waiting for elixir")
            return False

        if wait_output == "no battle":
            logger.change_status("Not in battle anymore!")
            break

        if not check_if_in_battle(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status("Battle ended (confirmed)")
                break

            logger.change_status("Lost battle detection — continuing fight loop")
            continue

        # Hard early-game gate: never spend a card in the first 10 seconds no matter
        # what wait_for_elixir returned (the PLAY_THRESHOLD path can fire early).
        if battle_strategy.get_battle_phase() == "early":
            logger.change_status("Early game — holding cards, waiting for elixir to build")
            time.sleep(1)
            continue

        play_start_time = time.time()
        if play_a_card_smart(emulator, logger, recording_flag, battle_strategy, combat_tracker) is False:
            logger.change_status("Failed to play a card, retrying...")
        # play_time_taken = str(time.time() - play_start_time)[:4]
        logger.change_status(
            f"Made a play in {str(time.time() - play_start_time)[:4]}s",
        )

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Fight complete")
    time.sleep(2.13)
    cards_played = logger.get_cards_played()
    logger.change_status(f"Played ~{cards_played - prev_cards_played} cards this fight")

    return True


def _random_fight_loop(
    emulator,
    logger,
    recording_flag: bool = False,
    fight_mode_chosen: str = "Trophy Road",
    custom_path: str | None = None,
) -> bool:
    """Method for handling dynamically timed fight with random plays"""
    logger.change_status(status="Starting battle with random plays")
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode_chosen, custom_path)
    create_default_bridge_iar(emulator)
    fight_timeout = 5 * 60  # 5 minutes
    start_time = time.time()
    battle_detection_lost_count = 0

    # while in battle:
    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}) — waiting it out",
            )

            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly — assuming battle ended",
                )
                break

            time.sleep(1)
            continue

        battle_detection_lost_count = 0
        if time.time() - start_time > fight_timeout:
            logger.change_status("Random fight loop timed out after 5 minutes")
            return False

        # Clean random-play flow (mirrors rl-bot RandomPlayer): wait for a random
        # elixir gate, then play only if a card is actually available.
        target = random.randint(RANDOM_PLAY_ELIXIR_MIN, RANDOM_PLAY_ELIXIR_MAX)
        elixir_result = wait_for_elixir(emulator, logger, target, recording_flag=recording_flag)
        if elixir_result == "no battle":
            break
        if elixir_result == "restart":
            return False
        play_random_available_card(emulator, logger, recording_flag, time.time() - start_time)

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Random-plays fight complete")
    return True


if __name__ == "__main__":
    pass
