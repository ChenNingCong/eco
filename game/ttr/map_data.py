"""
Ticket to Ride US map data — routes, destinations, cities.
All data extracted from the official US map.
"""

# ── Colors ───────────────────────────────────────────────────────────────────
# WILD (gray) routes can be paid with any single color
WHITE, BLUE, GREEN, ORANGE, PINK, RED, YELLOW, BLACK, WILD = range(9)
NUM_COLORS = 9  # 8 real colors + WILD
COLOR_NAMES = ["WHITE", "BLUE", "GREEN", "ORANGE", "PINK", "RED", "YELLOW", "BLACK", "WILD"]

# Cards per color in the deck
CARDS_PER_COLOR = [12, 12, 12, 12, 12, 12, 12, 12, 14]  # 14 wilds, 12 each other
TOTAL_CARDS = sum(CARDS_PER_COLOR)  # 110

# ── Cities ───────────────────────────────────────────────────────────────────

CITIES = [
    "ATLANTA", "BOSTON", "CALGARY", "CHARLESTON", "CHICAGO",
    "DALLAS", "DENVER", "DULUTH", "EL_PASO", "HELENA",
    "HOUSTON", "KANSAS_CITY", "LAS_VEGAS", "LITTLE_ROCK", "LOS_ANGELES",
    "MIAMI", "MONTREAL", "NASHVILLE", "NEW_ORLEANS", "NEW_YORK",
    "OKLAHOMA_CITY", "OMAHA", "PHOENIX", "PITTSBURGH", "PORTLAND",
    "RALEIGH", "SAINT_LOUIS", "SALT_LAKE_CITY", "SAN_FRANCISCO", "SANTA_FE",
    "SAULT_ST_MARIE", "SEATTLE", "TORONTO", "VANCOUVER", "WASHINGTON",
    "WINNIPEG",
]
NUM_CITIES = len(CITIES)  # 36
CITY_IDX = {name: i for i, name in enumerate(CITIES)}

# ── Route scoring ────────────────────────────────────────────────────────────

ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}

# ── Routes ───────────────────────────────────────────────────────────────────
# (city1, city2, color, length, adjacent_route_id or -1)

_ROUTE_DATA = [
    # WILD (gray) routes
    ("ATLANTA", "NASHVILLE", WILD, 1, -1),           # 0
    ("VANCOUVER", "SEATTLE", WILD, 1, 2),             # 1
    ("VANCOUVER", "SEATTLE", WILD, 1, 1),             # 2
    ("PORTLAND", "SEATTLE", WILD, 1, 4),              # 3
    ("PORTLAND", "SEATTLE", WILD, 1, 3),              # 4
    ("DALLAS", "HOUSTON", WILD, 1, 6),               # 5
    ("DALLAS", "HOUSTON", WILD, 1, 5),               # 6
    ("OMAHA", "KANSAS_CITY", WILD, 1, 8),            # 7
    ("OMAHA", "KANSAS_CITY", WILD, 1, 7),            # 8
    ("DULUTH", "OMAHA", WILD, 2, 10),                # 9
    ("DULUTH", "OMAHA", WILD, 2, 9),                 # 10
    ("KANSAS_CITY", "OKLAHOMA_CITY", WILD, 2, 12),   # 11
    ("KANSAS_CITY", "OKLAHOMA_CITY", WILD, 2, 11),   # 12
    ("DALLAS", "OKLAHOMA_CITY", WILD, 2, 14),        # 13
    ("DALLAS", "OKLAHOMA_CITY", WILD, 2, 13),        # 14
    ("ATLANTA", "RALEIGH", WILD, 2, 16),             # 15
    ("ATLANTA", "RALEIGH", WILD, 2, 15),             # 16
    ("WASHINGTON", "RALEIGH", WILD, 2, 18),          # 17
    ("WASHINGTON", "RALEIGH", WILD, 2, 17),          # 18
    ("BOSTON", "MONTREAL", WILD, 2, 20),              # 19
    ("BOSTON", "MONTREAL", WILD, 2, 19),              # 20
    ("OKLAHOMA_CITY", "LITTLE_ROCK", WILD, 2, -1),   # 21
    ("DALLAS", "LITTLE_ROCK", WILD, 2, -1),          # 22
    ("SAINT_LOUIS", "LITTLE_ROCK", WILD, 2, -1),     # 23
    ("SAINT_LOUIS", "NASHVILLE", WILD, 2, -1),       # 24
    ("RALEIGH", "PITTSBURGH", WILD, 2, -1),          # 25
    ("PITTSBURGH", "WASHINGTON", WILD, 2, -1),       # 26
    ("PITTSBURGH", "TORONTO", WILD, 2, -1),          # 27
    ("HOUSTON", "NEW_ORLEANS", WILD, 2, -1),         # 28
    ("ATLANTA", "CHARLESTON", WILD, 2, -1),          # 29
    ("RALEIGH", "CHARLESTON", WILD, 2, -1),          # 30
    ("LOS_ANGELES", "LAS_VEGAS", WILD, 2, -1),       # 31
    ("SANTA_FE", "DENVER", WILD, 2, -1),             # 32
    ("SANTA_FE", "EL_PASO", WILD, 2, -1),            # 33
    ("SAULT_ST_MARIE", "TORONTO", WILD, 2, -1),      # 34
    ("MONTREAL", "TORONTO", WILD, 3, -1),            # 35
    ("PHOENIX", "EL_PASO", WILD, 3, -1),             # 36
    ("PHOENIX", "SANTA_FE", WILD, 3, -1),            # 37
    ("PHOENIX", "LOS_ANGELES", WILD, 3, -1),         # 38
    ("VANCOUVER", "CALGARY", WILD, 3, -1),           # 39
    ("DULUTH", "SAULT_ST_MARIE", WILD, 3, -1),       # 40
    ("SEATTLE", "CALGARY", WILD, 4, -1),             # 41
    ("CALGARY", "HELENA", WILD, 4, -1),              # 42
    # GREEN
    ("SAINT_LOUIS", "CHICAGO", GREEN, 2, 79),         # 43
    ("PITTSBURGH", "NEW_YORK", GREEN, 2, 78),         # 44
    ("LITTLE_ROCK", "NEW_ORLEANS", GREEN, 3, -1),     # 45
    ("HELENA", "DENVER", GREEN, 4, -1),               # 46
    ("SAINT_LOUIS", "PITTSBURGH", GREEN, 5, -1),      # 47
    ("PORTLAND", "SAN_FRANCISCO", GREEN, 5, 90),       # 48
    ("EL_PASO", "HOUSTON", GREEN, 6, -1),             # 49
    # BLUE
    ("KANSAS_CITY", "SAINT_LOUIS", BLUE, 2, 85),      # 50
    ("SANTA_FE", "OKLAHOMA_CITY", BLUE, 3, -1),       # 51
    ("NEW_YORK", "MONTREAL", BLUE, 3, -1),            # 52
    ("OMAHA", "CHICAGO", BLUE, 4, -1),                # 53
    ("HELENA", "WINNIPEG", BLUE, 4, -1),              # 54
    ("ATLANTA", "MIAMI", BLUE, 5, -1),                # 55
    ("PORTLAND", "SALT_LAKE_CITY", BLUE, 6, -1),       # 56
    # RED
    ("NEW_YORK", "BOSTON", RED, 2, 92),                # 57
    ("DULUTH", "CHICAGO", RED, 3, -1),                # 58
    ("SALT_LAKE_CITY", "DENVER", RED, 3, 93),         # 59
    ("DENVER", "OKLAHOMA_CITY", RED, 4, -1),          # 60
    ("EL_PASO", "DALLAS", RED, 4, -1),                # 61
    ("HELENA", "OMAHA", RED, 5, -1),                  # 62
    ("NEW_ORLEANS", "MIAMI", RED, 6, -1),             # 63
    # ORANGE
    ("WASHINGTON", "NEW_YORK", ORANGE, 2, 71),        # 64
    ("CHICAGO", "PITTSBURGH", ORANGE, 3, 65),         # 65
    ("LAS_VEGAS", "SALT_LAKE_CITY", ORANGE, 3, -1),   # 66
    ("NEW_ORLEANS", "ATLANTA", ORANGE, 4, 96),        # 67
    ("DENVER", "KANSAS_CITY", ORANGE, 4, 74),         # 68
    ("SAN_FRANCISCO", "SALT_LAKE_CITY", ORANGE, 5, 83), # 69
    ("HELENA", "DULUTH", ORANGE, 6, -1),              # 70
    # BLACK
    ("WASHINGTON", "NEW_YORK", BLACK, 2, 64),          # 71
    ("NASHVILLE", "RALEIGH", BLACK, 3, -1),           # 72
    ("CHICAGO", "PITTSBURGH", BLACK, 3, 65),          # 73
    ("DENVER", "KANSAS_CITY", BLACK, 4, 68),          # 74
    ("WINNIPEG", "DULUTH", BLACK, 4, -1),             # 75
    ("SAULT_ST_MARIE", "MONTREAL", BLACK, 5, -1),     # 76
    ("LOS_ANGELES", "EL_PASO", BLACK, 6, -1),         # 77
    # WHITE
    ("PITTSBURGH", "NEW_YORK", WHITE, 2, 44),         # 78
    ("SAINT_LOUIS", "CHICAGO", WHITE, 2, 43),         # 79
    ("LITTLE_ROCK", "NASHVILLE", WHITE, 3, -1),       # 80
    ("CHICAGO", "TORONTO", WHITE, 4, -1),             # 81
    ("DENVER", "PHOENIX", WHITE, 5, -1),              # 82
    ("SAN_FRANCISCO", "SALT_LAKE_CITY", WHITE, 5, 69), # 83
    ("CALGARY", "WINNIPEG", WHITE, 6, -1),            # 84
    # PINK
    ("KANSAS_CITY", "SAINT_LOUIS", PINK, 2, 50),      # 85
    ("LOS_ANGELES", "SAN_FRANCISCO", PINK, 3, 94),    # 86
    ("SALT_LAKE_CITY", "HELENA", PINK, 3, -1),        # 87
    ("DENVER", "OMAHA", PINK, 4, -1),                 # 88
    ("CHARLESTON", "MIAMI", PINK, 4, -1),             # 89
    ("SAN_FRANCISCO", "PORTLAND", PINK, 5, 48),        # 90
    ("DULUTH", "TORONTO", PINK, 6, -1),               # 91
    # YELLOW
    ("NEW_YORK", "BOSTON", YELLOW, 2, 57),             # 92
    ("SALT_LAKE_CITY", "DENVER", YELLOW, 3, 59),      # 93
    ("LOS_ANGELES", "SAN_FRANCISCO", YELLOW, 3, 86),  # 94
    ("NASHVILLE", "PITTSBURGH", YELLOW, 4, -1),       # 95
    ("NEW_ORLEANS", "ATLANTA", YELLOW, 4, 67),        # 96
    ("EL_PASO", "OKLAHOMA_CITY", YELLOW, 5, -1),      # 97
    ("SEATTLE", "HELENA", YELLOW, 6, -1),             # 98
]

NUM_ROUTES = len(_ROUTE_DATA)  # 99

# Structured arrays for fast lookup
ROUTE_CITY1 = [CITY_IDX[r[0]] for r in _ROUTE_DATA]
ROUTE_CITY2 = [CITY_IDX[r[1]] for r in _ROUTE_DATA]
ROUTE_COLOR = [r[2] for r in _ROUTE_DATA]
ROUTE_LENGTH = [r[3] for r in _ROUTE_DATA]
ROUTE_ADJACENT = [r[4] for r in _ROUTE_DATA]
ROUTE_POINTS_LIST = [ROUTE_POINTS[r[3]] for r in _ROUTE_DATA]

# ── Destinations ─────────────────────────────────────────────────────────────
# (city1, city2, points)

_DEST_DATA = [
    ("BOSTON", "MIAMI", 12),             # 0
    ("CALGARY", "PHOENIX", 13),          # 1
    ("CALGARY", "SALT_LAKE_CITY", 7),    # 2
    ("CHICAGO", "NEW_ORLEANS", 7),       # 3
    ("CHICAGO", "SANTA_FE", 9),          # 4
    ("DALLAS", "NEW_YORK", 11),          # 5
    ("DENVER", "EL_PASO", 4),            # 6
    ("DENVER", "PITTSBURGH", 11),        # 7
    ("DULUTH", "EL_PASO", 10),           # 8
    ("DULUTH", "HOUSTON", 8),            # 9
    ("HELENA", "LOS_ANGELES", 8),        # 10
    ("KANSAS_CITY", "HOUSTON", 5),       # 11
    ("LOS_ANGELES", "CHICAGO", 16),      # 12
    ("LOS_ANGELES", "MIAMI", 20),        # 13
    ("LOS_ANGELES", "NEW_YORK", 21),     # 14
    ("MONTREAL", "ATLANTA", 9),          # 15
    ("MONTREAL", "NEW_ORLEANS", 13),     # 16
    ("NEW_YORK", "ATLANTA", 6),          # 17
    ("PORTLAND", "NASHVILLE", 17),       # 18
    ("PORTLAND", "PHOENIX", 11),         # 19
    ("SAN_FRANCISCO", "ATLANTA", 17),    # 20
    ("SAULT_ST_MARIE", "NASHVILLE", 8),  # 21
    ("SAULT_ST_MARIE", "OKLAHOMA_CITY", 9), # 22
    ("SEATTLE", "LOS_ANGELES", 9),       # 23
    ("SEATTLE", "NEW_YORK", 22),         # 24
    ("TORONTO", "MIAMI", 10),            # 25
    ("VANCOUVER", "MONTREAL", 20),       # 26
    ("VANCOUVER", "SANTA_FE", 13),       # 27
    ("WINNIPEG", "HOUSTON", 12),         # 28
    ("WINNIPEG", "LITTLE_ROCK", 11),     # 29
]

NUM_DESTINATIONS = len(_DEST_DATA)  # 30
DEST_CITY1 = [CITY_IDX[d[0]] for d in _DEST_DATA]
DEST_CITY2 = [CITY_IDX[d[1]] for d in _DEST_DATA]
DEST_POINTS = [d[2] for d in _DEST_DATA]

# ── City coordinates (approx lat/lon for US map) ────────────────────────────
# Used for SVG map rendering. Format: {city_name: (lat, lon)}

CITY_COORDS = {
    "ATLANTA": (33.75, -84.39),
    "BOSTON": (42.36, -71.06),
    "CALGARY": (51.05, -114.07),
    "CHARLESTON": (32.78, -79.93),
    "CHICAGO": (41.88, -87.63),
    "DALLAS": (32.78, -96.80),
    "DENVER": (39.74, -104.99),
    "DULUTH": (46.79, -92.10),
    "EL_PASO": (31.76, -106.49),
    "HELENA": (46.60, -112.04),
    "HOUSTON": (29.76, -95.37),
    "KANSAS_CITY": (39.10, -94.58),
    "LAS_VEGAS": (36.17, -115.14),
    "LITTLE_ROCK": (34.75, -92.29),
    "LOS_ANGELES": (34.05, -118.24),
    "MIAMI": (25.76, -80.19),
    "MONTREAL": (45.50, -73.57),
    "NASHVILLE": (36.16, -86.78),
    "NEW_ORLEANS": (29.95, -90.07),
    "NEW_YORK": (40.71, -74.01),
    "OKLAHOMA_CITY": (35.47, -97.52),
    "OMAHA": (41.26, -95.94),
    "PHOENIX": (33.45, -112.07),
    "PITTSBURGH": (40.44, -79.99),
    "PORTLAND": (45.51, -122.68),
    "RALEIGH": (35.78, -78.64),
    "SAINT_LOUIS": (38.63, -90.20),
    "SALT_LAKE_CITY": (40.76, -111.89),
    "SAN_FRANCISCO": (37.77, -122.42),
    "SANTA_FE": (35.69, -105.94),
    "SAULT_ST_MARIE": (46.50, -84.35),
    "SEATTLE": (47.61, -122.33),
    "TORONTO": (43.65, -79.38),
    "VANCOUVER": (49.28, -123.12),
    "WASHINGTON": (38.91, -77.04),
    "WINNIPEG": (49.90, -97.14),
}

# ── Game constants ───────────────────────────────────────────────────────────

STARTING_TRAINS = 45
STARTING_HAND_SIZE = 4
VISIBLE_CARD_SLOTS = 5
DEST_DRAW_SIZE = 3
LAST_ROUND_THRESHOLD = 3  # triggers when any player has < 3 trains
MAX_POINTS = 300  # normalization upper bound
