from feature._101_season import _101_RegularSeasonStats
from feature._102_season_elo import _102_RegularSeasonEloRating
from feature._103_season_advanced_stats import _103_SeasonAdvancedStats
from feature._201_seed import _201_Seed
from feature._202_conference import _202_Conference
from feature._301_tourney_wins import _301_TourneyWins
from feature._302_tourney_history_stats import _302_TourneyHistoryStats
from feature._401_massey_ordinals_pom import _401_POMRank
from feature._402_massey_ordinals_sagarin import _402_SagarinRank

MAPPER = {
    "_101_season": _101_RegularSeasonStats,
    "_102_season_elo": _102_RegularSeasonEloRating,
    "_103_season_advanced_stats": _103_SeasonAdvancedStats,
    "_201_seed": _201_Seed,
    "_202_conference": _202_Conference,
    "_301_tourney_wins": _301_TourneyWins,
    "_302_tourney_history_stats": _302_TourneyHistoryStats,
    "_401_pom_rank": _401_POMRank,
    "_402_sagarin_rank": _402_SagarinRank,
}
