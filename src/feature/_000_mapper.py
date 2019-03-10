from feature._101_season import _101_RegularSeasonStats
from feature._102_season_elo import _102_RegularSeasonEloRating
from feature._201_seed import _201_Seed

MAPPER = {
    "_101_season": _101_RegularSeasonStats,
    "_102_season_elo": _102_RegularSeasonEloRating,
    "_201_seed": _201_Seed
}
