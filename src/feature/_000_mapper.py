from feature._101_season import _101_RegularSeasonStats
from feature._102_season_elo import _102_RegularSeasonEloRating
from feature._201_seed import _201_Seed
from feature._301_tourney_wins import _301_TourneyWins
from feature._401_massey_ordinals import _401_POMRank

MAPPER = {
    "_101_season": _101_RegularSeasonStats,
    "_102_season_elo": _102_RegularSeasonEloRating,
    "_201_seed": _201_Seed,
    "_301_tourney_wins": _301_TourneyWins,
    "_401_pomrank": _401_POMRank,
}
