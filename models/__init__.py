import os

# A bunch of env set variables for global control
VERBOSITY = int(os.environ.get("VERBOSITY", -1))
LOOK_AHEAD_CNT = int(os.environ.get("LOOK_AHEAD_CNT", 8))
KEEP = float(os.environ.get("KEEP", -2))
SAVE_STATS_DIR = os.environ.get("SAVE_STATS_DIR", None)