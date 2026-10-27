from urbanheat.change import gee


def test_hot_season_picks_the_warmest_three_months():
    pune = [30, 33, 38, 42, 43, 35, 28, 27, 29, 31, 30, 29]         # pre-monsoon peak
    assert gee.hot_season(pune) == 3                                   # March-May has the largest sum
    southern = [40, 39, 35, 30, 25, 20, 18, 20, 25, 30, 35, 41]       # wraps over the new year
    assert gee.hot_season(southern) == 12
    assert gee.hot_season([None] * 12) is None
    assert gee.hot_season([None, 30, 33, 38, 42, 43, None, 27, 29, 31, 30, 29]) == 4


def test_season_window_runs_into_the_next_year():
    assert gee.season_window(2020, 4) == ("2020-04-01", "2020-07-01")
    assert gee.season_window(2020, 11) == ("2020-11-01", "2021-02-01")
    assert gee.season_window(2020, 10) == ("2020-10-01", "2021-01-01")
