"""Deck builder: scans the card collection, scores meta decks, equips the best one.

Flow
----
1. Navigate to the Cards page → Collection tab
2. Scroll through the grid taking screenshots, detect card fingerprints
3. Score each meta deck by how many of its 8 cards are owned
4. Navigate to the Decks page, slot 1
5. Clear the existing deck and add the best-scoring cards one by one

The collection scanner adapts the same color-fingerprint approach used for
hand cards during battle: it samples 4 corner regions of each visible card
thumbnail and matches against card_color_data.

Coordinates are for the standard 419×633 emulator resolution.
"""

import time

import numpy

from pyclashbot.bot.card_detection import (
    HALF_HEIGHT,
    HALF_WIDTH,
    card_color_data,
    find_closest_card,
    get_corner_pixels,
)
from pyclashbot.bot.coords import (
    DECK_BUILDER_BACK_BUTTON_COORD,
    DECK_BUILDER_CLEAR_DECK_BUTTON_COORD,
    DECK_BUILDER_COLLECTION_TAB_COORD,
    DECK_BUILDER_CONFIRM_CLEAR_COORD,
    DECK_BUILDER_DECK_SLOT_1_COORD,
    DECK_BUILDER_EDIT_DECK_BUTTON_COORD,
    DECK_BUILDER_EDITOR_GRID_ROW_START_Y,
    DECK_BUILDER_EDITOR_GRID_ROW_STEP,
    DECK_BUILDER_GRID_COL_POSITIONS,
    DECK_BUILDER_GRID_ROW_START_Y,
    DECK_BUILDER_GRID_ROW_STEP,
    DECK_BUILDER_SCROLL_END,
    DECK_BUILDER_SCROLL_START,
)
from pyclashbot.bot.nav import (
    _navigate_to_deck_selection,
    get_to_card_page_from_clash_main,
    wait_for_clash_main_menu,
)
from pyclashbot.utils.logger import Logger

# ---------------------------------------------------------------------------
# Meta deck database — card IDs must match keys in card_color_data
# ---------------------------------------------------------------------------
META_DECKS: list[tuple[str, list[str]]] = [
    (
        "Hog Cycle",
        ["hog", "ice_spirit", "skeletons", "log", "fireball", "musketeer", "cannon", "knight"],
    ),
    (
        "Goblin Barrel Cycle",
        ["goblin_barrel", "princess", "log", "skeletons", "ice_spirit", "knight", "fireball", "goblin_gang"],
    ),
    (
        "Miner Poison Control",
        ["miner", "poison", "bats", "mega_minion", "inferno_tower", "goblin_gang", "log", "fireball"],
    ),
    (
        "Lava Hound Balloon",
        ["lava_hound", "balloon", "mega_minion", "tombstone", "lightning", "minions", "bats", "arrows"],
    ),
    (
        "PEKKA Bridge Spam",
        ["pekka", "battle_ram", "electro_wizard", "mega_minion", "dark_prince", "zap", "fireball", "magic_archer"],
    ),
    (
        "Giant Graveyard",
        ["giant", "graveyard", "poison", "mega_minion", "electro_wizard", "tombstone", "zap", "log"],
    ),
    (
        "Goblin Giant Sparky",
        ["goblin_giant", "sparky", "goblin_gang", "dart_goblin", "zap", "minions", "fireball", "electro_wizard"],
    ),
    (
        "Balloon Freeze",
        ["balloon", "freeze", "mega_minion", "lumberjack", "tombstone", "bats", "zap", "fireball"],
    ),
    (
        "Mortar Cycle",
        ["mortar", "ice_spirit", "skeletons", "log", "knight", "archers", "royal_delivery", "bats"],
    ),
    (
        "Elixir Golem Beatdown",
        ["elixir_golem", "battle_healer", "night_witch", "dark_prince", "mega_minion", "zap", "log", "lightning"],
    ),
    (
        "X-Bow Cycle",
        ["xbow", "tesla", "ice_spirit", "skeletons", "log", "archers", "fireball", "knight"],
    ),
    (
        "Ram Rider Bridge Spam",
        ["ram_rider", "battle_ram", "electro_wizard", "dark_prince", "zap", "fireball", "mega_minion", "log"],
    ),
    (
        "Royal Giant Furnace",
        ["royal_giant", "furnace", "electro_wizard", "minions", "zap", "log", "mega_minion", "fireball"],
    ),
    (
        "Witch Beatdown",
        ["witch", "giant_skeleton", "fireball", "arrows", "tombstone", "night_witch", "dark_prince", "log"],
    ),
    (
        "Three Musketeers",
        ["three_musketeers", "elixir_collector", "battle_ram", "skeletons", "ice_spirit", "zap", "minions", "log"],
    ),
]

# ---------------------------------------------------------------------------
# Collection screen layout constants (all coords live in coords.py)
# ---------------------------------------------------------------------------

_VISIBLE_ROWS = 4       # rows visible in the collection grid per screenshot
_EDITOR_VISIBLE_ROWS = 3
_MAX_SCROLL_PAGES = 10  # max scroll pages before giving up
_SCROLL_DURATION_MS = 350

# Card thumbnail half-dimensions for fingerprint corner sampling.
# Collection thumbnails are roughly half the size of a hand card.
_COLL_HALF_W = max(4, HALF_WIDTH // 2)
_COLL_HALF_H = max(4, HALF_HEIGHT // 2)
_COLL_TOTAL_W = _COLL_HALF_W * 2
_COLL_TOTAL_H = _COLL_HALF_H * 2


# ---------------------------------------------------------------------------
# Card fingerprinting helpers (adapted from card_detection.py)
# ---------------------------------------------------------------------------

def _sample_card_fingerprint(iar: numpy.ndarray, cx: int, cy: int) -> list[dict]:
    """Sample 4-quadrant color fingerprint of a card thumbnail centered at (cx, cy).

    Returns the same list-of-corner-pixel-dicts format expected by find_closest_card.
    """
    x0 = cx - _COLL_HALF_W
    y0 = cy - _COLL_HALF_H

    corners = [
        ((x0,              x0 + _COLL_HALF_W), (y0,              y0 + _COLL_HALF_H)),
        ((x0 + _COLL_HALF_W, x0 + _COLL_TOTAL_W), (y0,              y0 + _COLL_HALF_H)),
        ((x0,              x0 + _COLL_HALF_W), (y0 + _COLL_HALF_H, y0 + _COLL_TOTAL_H)),
        ((x0 + _COLL_HALF_W, x0 + _COLL_TOTAL_W), (y0 + _COLL_HALF_H, y0 + _COLL_TOTAL_H)),
    ]

    h, w = iar.shape[:2]
    result = []
    for (x_range, y_range) in corners:
        x1, x2 = max(0, x_range[0]), min(w, x_range[1])
        y1, y2 = max(0, y_range[0]), min(h, y_range[1])
        if x2 <= x1 or y2 <= y1:
            return []  # out of bounds — skip this cell
        pixels = get_corner_pixels((x1, x2), (y1, y2), iar)
        result.append(pixels)
    return result


def _identify_collection_card(iar: numpy.ndarray, cx: int, cy: int) -> str | None:
    """Return the card_id at collection position (cx, cy), or None if unknown."""
    fingerprint = _sample_card_fingerprint(iar, cx, cy)
    if not fingerprint:
        return None
    try:
        card_id = find_closest_card(fingerprint)
        if card_id and card_id != "UNKNOWN":
            return card_id
    except Exception:
        pass
    return None


def _scan_visible_grid(
    emulator,
    row_start_y: int,
    row_step: int,
    col_positions: list[int],
    n_rows: int,
) -> list[str]:
    """Take a screenshot and fingerprint every card in the visible grid.

    Returns a list of identified card_ids (duplicates allowed).
    """
    iar = emulator.screenshot()
    found = []
    for row in range(n_rows):
        cy = row_start_y + row * row_step
        for cx in col_positions:
            card_id = _identify_collection_card(iar, cx, cy)
            if card_id:
                found.append(card_id)
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_collection(emulator, logger: Logger) -> set[str]:
    """Navigate to the card collection and return the set of owned card IDs.

    Scrolls through the full collection page by page. Returns an empty set
    if navigation fails.
    """
    logger.change_status("Navigating to card collection...")

    if not wait_for_clash_main_menu(emulator, logger):
        logger.change_status("Could not reach main menu — aborting collection scan")
        return set()

    if not get_to_card_page_from_clash_main(emulator, logger):
        logger.change_status("Could not reach card page — aborting collection scan")
        return set()

    # Click Collection tab (not Decks)
    emulator.click(*DECK_BUILDER_COLLECTION_TAB_COORD)
    time.sleep(1)

    owned: set[str] = set()
    prev_count = -1

    logger.change_status("Scanning card collection...")
    for page in range(_MAX_SCROLL_PAGES):
        found = _scan_visible_grid(
            emulator,
            DECK_BUILDER_GRID_ROW_START_Y,
            DECK_BUILDER_GRID_ROW_STEP,
            DECK_BUILDER_GRID_COL_POSITIONS,
            _VISIBLE_ROWS,
        )
        owned.update(found)
        logger.change_status(
            f"Collection scan page {page + 1} — {len(owned)} unique cards found so far"
        )

        # Stop if no new cards appeared (we've reached the end of the collection)
        if len(owned) == prev_count:
            break
        prev_count = len(owned)

        # Swipe up to reveal the next rows
        emulator.swipe(
            DECK_BUILDER_SCROLL_START[0], DECK_BUILDER_SCROLL_START[1],
            DECK_BUILDER_SCROLL_END[0], DECK_BUILDER_SCROLL_END[1],
            _SCROLL_DURATION_MS,
        )
        time.sleep(0.6)

    logger.change_status(f"Collection scan complete — {len(owned)} cards detected")
    return owned


def score_decks(owned: set[str]) -> list[tuple[int, str, list[str]]]:
    """Score each meta deck by how many of its cards the player owns.

    Returns [(score, deck_name, card_list), ...] sorted best-first.
    """
    # Build a normalised owned set: strip evo_/hero_ prefixes so an evo card
    # counts toward a deck that lists the base card, and vice versa.
    normalised: set[str] = set(owned)
    for card in owned:
        for prefix in ("evo_", "hero_"):
            if card.startswith(prefix):
                normalised.add(card[len(prefix):])

    known = set(card_color_data.keys())
    results = []
    for name, cards in META_DECKS:
        # Only count cards the bot's fingerprinter actually knows
        valid = [c for c in cards if c in known]
        score = sum(1 for c in valid if c in normalised)
        results.append((score, name, cards))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def best_buildable_deck(owned: set[str]) -> tuple[str, list[str]] | None:
    """Return (deck_name, card_list) for the highest-scoring meta deck, or None."""
    scored = score_decks(owned)
    if not scored or scored[0][0] == 0:
        return None
    score, name, cards = scored[0]
    return name, cards


def _find_and_click_card_in_editor(emulator, target_card: str) -> bool:
    """Scan the deck editor grid (scrolling if needed) to find and click target_card.

    Returns True if the card was found and clicked.
    """
    for _page in range(5):
        iar = emulator.screenshot()
        for row in range(_EDITOR_VISIBLE_ROWS):
            cy = DECK_BUILDER_EDITOR_GRID_ROW_START_Y + row * DECK_BUILDER_EDITOR_GRID_ROW_STEP
            for cx in DECK_BUILDER_GRID_COL_POSITIONS:
                card_id = _identify_collection_card(iar, cx, cy)
                if card_id == target_card:
                    emulator.click(cx, cy)
                    time.sleep(0.25)
                    return True

        # Scroll down in the editor and try again
        emulator.swipe(
            DECK_BUILDER_SCROLL_START[0], DECK_BUILDER_SCROLL_START[1],
            DECK_BUILDER_SCROLL_END[0], DECK_BUILDER_SCROLL_END[1],
            _SCROLL_DURATION_MS,
        )
        time.sleep(0.5)

    return False


def equip_deck(emulator, logger: Logger, card_list: list[str]) -> bool:
    """Navigate to Deck slot 1 and replace it with the given cards.

    Returns True if at least one card was placed.
    """
    logger.change_status("Navigating to deck editor...")

    if not wait_for_clash_main_menu(emulator, logger):
        logger.change_status("Could not reach main menu — aborting deck equip")
        return False

    if not _navigate_to_deck_selection(emulator, logger):
        logger.change_status("Could not reach deck selection — aborting deck equip")
        return False

    time.sleep(0.5)

    # Click deck slot 1, then edit
    emulator.click(*DECK_BUILDER_DECK_SLOT_1_COORD)
    time.sleep(0.8)
    emulator.click(*DECK_BUILDER_EDIT_DECK_BUTTON_COORD)
    time.sleep(1)

    # Clear existing deck
    emulator.click(*DECK_BUILDER_CLEAR_DECK_BUTTON_COORD)
    time.sleep(0.5)
    emulator.click(*DECK_BUILDER_CONFIRM_CLEAR_COORD)
    time.sleep(1)

    placed = 0
    for target_card in card_list:
        if _find_and_click_card_in_editor(emulator, target_card):
            placed += 1
            logger.change_status(f"Added {target_card} to deck ({placed}/{len(card_list)})")
        else:
            logger.change_status(f"Could not find {target_card} in editor — skipping")

    logger.change_status(f"Deck equip done — placed {placed}/{len(card_list)} cards")

    # Navigate back to main
    emulator.click(*DECK_BUILDER_BACK_BUTTON_COORD)
    time.sleep(1)

    return placed > 0


def run_deck_builder(emulator, logger: Logger) -> bool:
    """Full deck-builder run: scan collection → pick best meta deck → equip it.

    Returns True if a deck was successfully equipped.
    """
    logger.change_status("Deck builder starting...")

    owned = scan_collection(emulator, logger)
    if not owned:
        logger.change_status("Deck builder: no cards detected — cannot build deck")
        return False

    scored = score_decks(owned)
    logger.change_status(
        f"Best deck match: {scored[0][1]} ({scored[0][0]}/8 cards owned)"
    )

    for score, name, cards in scored:
        if score >= 6:
            logger.change_status(f"Building deck: {name} ({score}/8 cards matched)")
            if equip_deck(emulator, logger, cards):
                logger.change_status(f"Deck builder complete — equipped: {name}")
                return True

    logger.change_status("Deck builder: no meta deck had 6+ owned cards matched")
    return False
