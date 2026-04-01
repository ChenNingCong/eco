"""
game_log.py — print a full game log for random play.

Usage:
    python game_log.py              # random seed
    python game_log.py --seed 42    # fixed seed
    python game_log.py --seed 42 --no-color
"""

import sys, argparse
sys.path.insert(0, '.')

import numpy as np
from hearts_env import (
    HeartsEnv, NUM_PLAYERS, NUM_ROUNDS, MAX_SCORE,
    card_suit, card_rank, card_points,
    CLUBS, DIAMONDS, HEARTS, SPADES,
    TWO_OF_CLUBS, QUEEN_OF_SPADES,
)

# ── Card rendering ────────────────────────────────────────────────────────────

RANK_NAMES = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUIT_SYMBOLS = {
    CLUBS:    '♣',
    DIAMONDS: '♦',
    HEARTS:   '♥',
    SPADES:   '♠',
}
SUIT_NAMES = {CLUBS: 'Clubs', DIAMONDS: 'Diamonds', HEARTS: 'Hearts', SPADES: 'Spades'}

USE_COLOR = True

ANSI = {
    'red':    '\033[91m',
    'cyan':   '\033[96m',
    'yellow': '\033[93m',
    'green':  '\033[92m',
    'bold':   '\033[1m',
    'dim':    '\033[2m',
    'reset':  '\033[0m',
}

def col(text, *codes):
    if not USE_COLOR:
        return text
    prefix = ''.join(ANSI[c] for c in codes)
    return f"{prefix}{text}{ANSI['reset']}"

def card_str(card_id: int, highlight: bool = False) -> str:
    suit = card_suit(card_id)
    rank = RANK_NAMES[card_rank(card_id)]
    sym  = SUIT_SYMBOLS[suit]
    pts  = card_points(card_id)
    s = f"{rank}{sym}"
    if suit in (HEARTS, DIAMONDS):
        s = col(s, 'red')
    if pts > 0:
        s = col(s, 'bold')
    if highlight:
        s = col(f"[{s}{col(']', 'yellow')}", 'yellow')
    return s

def hand_str(hand_bitmap: np.ndarray) -> str:
    cards = sorted(np.where(hand_bitmap)[0],
                   key=lambda c: (card_suit(c), card_rank(c)))
    return '  '.join(card_str(c) for c in cards) if cards else col('(empty)', 'dim')

def player_label(p: int) -> str:
    colors = ['cyan', 'yellow', 'green', 'red']
    return col(f"P{p}", colors[p], 'bold')

def sep(char='─', width=72):
    print(col(char * width, 'dim'))

def header(text):
    sep('═')
    print(col(f"  {text}", 'bold'))
    sep('═')


# ── Main log ──────────────────────────────────────────────────────────────────

def run(seed: int):
    env = HeartsEnv(seed=seed)
    rng = np.random.default_rng(seed)
    state = env.reset()

    header(f"HEARTS — Random Play  (seed={seed})")
    print()

    # Print initial hands
    print(col("  INITIAL HANDS", 'bold'))
    sep()
    for p in range(NUM_PLAYERS):
        print(f"  {player_label(p)}  {hand_str(state.hands[p])}")
    print()

    trick_num = 0
    card_plays = []   # (trick, player, card, is_winner)

    while not state.done:
        # ── Trick header ──────────────────────────────────────────────────────
        if state.current_trick_count == 0:
            trick_num += 1
            sep()
            print(f"  {col(f'TRICK {trick_num:2d}', 'bold')}  "
                  f"  Scores: " +
                  '  '.join(f"{player_label(p)} {state.scores[p]:2d}" for p in range(NUM_PLAYERS)) +
                  (f"  {col('♥ broken', 'red')}" if state.hearts_broken else ""))
            sep()

        s = state
        p = s.current_player
        mask = env.legal_actions()

        # Choose random legal action
        action = rng.choice(np.where(mask)[0])
        pts = card_points(action)

        # Print play line
        legal_cards = sorted(np.where(mask)[0], key=lambda c: (card_suit(c), card_rank(c)))
        trick_pos = s.current_trick_count

        print(f"  {player_label(p)} plays {card_str(action, highlight=True)}"
              + (col(f"  [{'+' if pts>0 else ''}{pts} pts]", 'red' if pts > 0 else 'dim') if pts else "")
              + f"  {col('hand:', 'dim')} {hand_str(s.hands[p])}"
              )

        # Step
        state, rewards, done, _ = env.step(action)

        # If trick just resolved
        if state.current_trick_count == 0 or done:
            rec = state.history[-1]
            winner = rec.winner
            trick_pts = sum(card_points(c) for c in rec.cards if c >= 0)
            print()
            print(f"  {col('→ Winner:', 'bold')} {player_label(winner)}"
                  + (f"  takes {col(str(trick_pts) + ' pts', 'red', 'bold')}" if trick_pts else
                     f"  {col('(no points)', 'dim')}")
                  )
            if not done:
                mid_rewards = rewards
                nonzero = [(p, rewards[p]) for p in range(NUM_PLAYERS) if rewards[p] != 0]
                if nonzero:
                    reward_str = '  '.join(
                        f"{player_label(p)} {col(f'{r:+.4f}', 'red')}"
                        for p, r in nonzero
                    )
                    print(f"  {col('  rewards:', 'dim')} {reward_str}")
            print()

    # ── Final results ─────────────────────────────────────────────────────────
    header("FINAL RESULTS")
    terminal = env._terminal_rewards(state)

    # Check shoot the moon
    shooter = np.where(state.scores == MAX_SCORE)[0]
    if len(shooter) == 1:
        print(col(f"  🌙  SHOOT THE MOON by {player_label(int(shooter[0]))}!", 'yellow', 'bold'))
        print()

    print(f"  {'Player':<10} {'Score':>6}  {'/ 26':>5}  {'Reward':>8}  {'Result'}")
    sep()
    sorted_players = sorted(range(NUM_PLAYERS), key=lambda p: state.scores[p])
    for rank_i, p in enumerate(sorted_players):
        score = state.scores[p]
        reward = terminal[p]
        bar = '█' * int(score / 2)
        result = col('WIN', 'green', 'bold') if rank_i == 0 else ''
        print(f"  {player_label(p):<18} {score:>4}  {score:>3}/26  "
              f"{col(f'{reward:+.4f}', 'red' if reward < 0 else 'green'):>18}  "
              f"{bar}  {result}")
    sep()
    print(f"  {'Total:':<10} {state.scores.sum():>4}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print Hearts game log for random play.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random)')
    parser.add_argument('--no-color', action='store_true', help='Disable ANSI colors')
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 10000))
    run(seed)