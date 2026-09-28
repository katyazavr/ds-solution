"""
features.py
Построение признаков для задачи детекции ботов по cookie_id.

Все признаки считаются строго по событиям внутри окна наблюдения
[window_start_ts, window_end_ts) — события за пределами окна (в т.ч.
показы captcha, которые в сырых данных лежат ТОЛЬКО после окна) в
признаки не попадают, иначе была бы утечка из будущего.

Единственная точка входа — build_features(events, meta). Она вызывается
ОДИНАКОВО для train и test, чтобы не было рассинхрона между выборками.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Константы, найденные в EDA
# ---------------------------------------------------------------------------

# небраузерные HTTP-клиенты (см. EDA: 4.9% ботов vs 1.5% людей)
RAW_CLIENT_MARKERS = [
    "node-fetch", "curl/", "scrapy", "python-urllib3",
    "go-http-client", "python-requests",
]

SESSION_GAP_SEC = 30 * 60   # разрыв > 30 минут = новая сессия
NIGHT_HOURS = (2, 6)        # [2:00, 6:00) - см. EDA: 18.6% событий у ботов vs 10.6% у людей

CONVERSION_EVENTS = {
    "favorite_add",
    "contact_chat_open",
    "contact_phone_show",
    "contact_message_sent",
}

TOP_N_BIGRAMS = 4

# ---------------------------------------------------------------------------
# Подготовка событий
# ---------------------------------------------------------------------------

def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Дедупликация и нормализация platform.

    platform: 11 вариантов регистра/написания в сырых данных -> 4 категории.
    desktop и web оказались ОДНИМ И ТЕМ ЖЕ трафиком под разными тегами
    (91.7% кук: идентичный user_agent под обоими тегами) - объединены
    в platform_canon.
    """
    ev = events.drop_duplicates().copy()
    ev["platform"] = ev["platform"].str.lower().replace({"iphone": "ios"})
    ev["platform_canon"] = ev["platform"].replace({"desktop": "web"})
    return ev


def events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только события внутри окна наблюдения куки.

    В сырых events 40 779 строк (только в train) лежат после window_end_ts,
    включая 100% показов captcha_shown. Это подтверждение качества разметки
    (у ботов таких событий в 10 раз больше), а не источник признаков -
    поэтому фильтр обязателен.
    """
    ev = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id", how="inner",
    )
    mask = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    return ev.loc[mask].copy()


def _is_headless(ua: pd.Series) -> pd.Series:
    return ua.str.contains("headless", case=False, na=False)


def _is_raw_client(ua: pd.Series) -> pd.Series:
    pattern = "|".join(RAW_CLIENT_MARKERS)
    return ua.str.contains(pattern, case=False, na=False, regex=True)


# ---------------------------------------------------------------------------
# Группы признаков
# ---------------------------------------------------------------------------

def _timing_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Объём, интенсивность, тайминги, сессии, время суток."""
    ev = ev.sort_values(["cookie_id", "event_ts"])
    gap = ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    ev = ev.assign(gap=gap)

    g = ev.groupby("cookie_id")
    out = pd.DataFrame({
        "n_events": g.size(),
        "n_event_types": g["event_name"].nunique(),
        "span_sec": g["event_ts"].apply(lambda s: (s.max() - s.min()).total_seconds()),
        "night_share": g["event_ts"].apply(
            lambda s: s.dt.hour.between(*NIGHT_HOURS, inclusive="left").mean()
        ),
    })

    gap_stats = ev.groupby("cookie_id")["gap"].agg(
        gap_mean="mean", gap_median="median", gap_std="std", gap_min="min",
    )
    out = out.join(gap_stats)

    new_session = ev["gap"].isna() | (ev["gap"] > SESSION_GAP_SEC)
    out["n_sessions"] = ev.assign(new_session=new_session).groupby("cookie_id")["new_session"].sum()
    out["events_per_min"] = out["n_events"] / (out["span_sec"] / 60).replace(0, np.nan)

    return out.reset_index()


def get_top_pairs(ev: pd.DataFrame, top_n: int = TOP_N_BIGRAMS) -> set:
    """Топ-N биграмм переходов - считать ТОЛЬКО на train-события (или на
    всём train при сборке test), иначе validation-куки участвуют в выборе
    топ-пар для своего же признака (transductive leakage).
    """
    ev = ev.sort_values(["cookie_id", "event_ts"]).copy()
    ev["next_event"] = ev.groupby("cookie_id")["event_name"].shift(-1)
    pairs = ev.dropna(subset=["next_event"]).copy()
    if pairs.empty:
        return set()
    pairs["pair"] = list(zip(pairs["event_name"], pairs["next_event"]))
    return set(pairs["pair"].value_counts().head(top_n).index)


def _sequence_features(ev: pd.DataFrame, top_pairs: set) -> pd.DataFrame:
    """Доля событий на top_pairs (см. get_top_pairs).

    У ботов поведение более зациклено: топ-4 биграммы (item_view <-> search_results_view)
    покрывают 63% всех переходов против 45% у людей.
    """
    ev = ev.sort_values(["cookie_id", "event_ts"]).copy()
    ev["next_event"] = ev.groupby("cookie_id")["event_name"].shift(-1)
    pairs = ev.dropna(subset=["next_event"]).copy()
    if pairs.empty:
        return pd.DataFrame(columns=["cookie_id", "top_bigram_share"])

    pairs["pair"] = list(zip(pairs["event_name"], pairs["next_event"]))
    pairs["is_top_pair"] = pairs["pair"].isin(top_pairs)

    out = pairs.groupby("cookie_id")["is_top_pair"].mean().rename("top_bigram_share")
    return out.reset_index()


def _positional_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Какое действие первое в окне - категориальный признак.

    Человек чаще стартует с поиска, бот может сразу идти в item_view -
    первое событие само по себе несёт сигнал, независимо от общего объёма.
    """
    first = (
        ev.sort_values(["cookie_id", "event_ts"])
        .groupby("cookie_id")["event_name"].first()
        .rename("first_event_name")
    )
    return first.reset_index()


def _diversity_features(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    n_events = g.size()
    out = pd.DataFrame({
        "item_nunique": g["item_id"].nunique(),
        "category_nunique": g["item_category"].nunique(),
        "location_nunique": g["item_location"].nunique(),
        "seller_pro_share": g["seller_type"].apply(lambda s: (s == "pro").mean()),
    })
    out["item_nunique_ratio"] = out["item_nunique"] / n_events.clip(lower=1)
    out["category_nunique_ratio"] = out["category_nunique"] / n_events.clip(lower=1)
    out["location_nunique_ratio"] = out["location_nunique"] / n_events.clip(lower=1)
    return out.reset_index()


def _revisit_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Повторный просмотр одних и тех же объявлений vs разных."""
    views = ev.loc[ev["event_name"] == "item_view"]
    if views.empty:
        return pd.DataFrame(columns=["cookie_id", "item_revisit_ratio", "n_item_views"])

    g = views.groupby("cookie_id")
    out = pd.DataFrame({
        "n_item_views": g.size(),
        "item_view_nunique": g["item_id"].nunique(),
    })
    out["item_revisit_ratio"] = out["n_item_views"] / out["item_view_nunique"].clip(lower=1)
    return out[["item_revisit_ratio", "n_item_views"]].reset_index()


def _conversion_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Доля конверсионных действий (contact/favorite) от всех событий куки."""
    ev = ev.copy()
    ev["is_conversion"] = ev["event_name"].isin(CONVERSION_EVENTS)
    g = ev.groupby("cookie_id")
    out = g["is_conversion"].mean().rename("conversion_share")
    return out.reset_index()


def _search_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Поведение в поиске (только search_results_view).

    query_diversity: доля уникальных запросов среди всех поисков куки
    (у людей медиана 1.0, у ботов 0.67 - повторяют запросы чаще).
    same_query_as_prev_share: повтор запроса сразу за предыдущим
    (19.2% у ботов vs 9.3% у людей; частично пагинация, но чаще у ботов).
    """
    sq = ev.loc[ev["event_name"] == "search_results_view"].sort_values(["cookie_id", "event_ts"]).copy()
    if sq.empty:
        return pd.DataFrame(columns=[
            "cookie_id", "n_searches", "query_diversity",
            "same_query_as_prev_share", "search_page_mean",
        ])

    sq["same_as_prev"] = sq.groupby("cookie_id")["search_query"].shift() == sq["search_query"]

    g = sq.groupby("cookie_id")
    out = pd.DataFrame({
        "n_searches": g.size(),
        "query_diversity": g["search_query"].nunique() / g.size(),
        "same_query_as_prev_share": g["same_as_prev"].mean(),
        "search_page_mean": g["search_page"].mean(),
    })
    return out.reset_index()


def _pointer_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Доля пропуска pointer_x, ТОЛЬКО в разрезе web (после объединения с desktop).

    На android/ios курсор отсутствует структурно в 100% случаев - там признак
    бессмысленен и задублировал бы platform, поэтому считаем только на web.
    """
    sub = ev[ev["platform_canon"] == "web"]
    if sub.empty:
        return pd.DataFrame(columns=["cookie_id", "pointer_missing_share"])
    out = sub.groupby("cookie_id")["pointer_x"].apply(lambda s: s.isna().mean())
    return out.rename("pointer_missing_share").reset_index()


def _ua_platform_features(ev: pd.DataFrame) -> pd.DataFrame:
    """is_headless (8.3% ботов vs 2.1% людей), is_raw_client (4.9% vs 1.5%),
    объединённый is_bot_ua (13.2% vs 3.6%) - покрывают меньшинство ботов,
    но с высокой точностью. platform_canon - категория для модели.
    """
    ev = ev.copy()
    ev["is_headless"] = _is_headless(ev["user_agent"])
    ev["is_raw_client"] = _is_raw_client(ev["user_agent"])

    g = ev.groupby("cookie_id")
    out = pd.DataFrame({
        "is_headless": g["is_headless"].max(),
        "is_raw_client": g["is_raw_client"].max(),
        "platform_canon": g["platform_canon"].agg(lambda s: s.mode().iat[0]),
    })
    out["is_bot_ua"] = out["is_headless"] | out["is_raw_client"]
    return out.reset_index()


def _meta_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Признаки напрямую из train/test, без events.

    cookie_age_days: медиана 19 дней у ботов против 62 у людей - один из
    самых сильных отдельных признаков (но не абсолютный).
    cookie_age_log1p: разница между 0.1 и 1 днём важнее для модели, чем
    между 100 и 101 - log1p сглаживает длинный хвост старых кук.
    """
    out = meta[["cookie_id"]].copy()
    out["cookie_age_days"] = (
        (meta["window_start_ts"] - meta["cookie_created_at"]).dt.total_seconds() / 86400
    )
    out["cookie_age_log1p"] = np.log1p(out["cookie_age_days"].clip(lower=0))
    return out


def _interaction_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Интенсивность активности относительно возраста куки.

    100 событий на кукe возрастом 0.2 дня и 100 событий на кукe возрастом
    100 дней - совсем разная история, а n_events и cookie_age_days по
    отдельности этого не выражают.
    """
    out = feats[["cookie_id"]].copy()
    out["events_per_age_day"] = feats["n_events"] / (feats["cookie_age_days"].clip(lower=0) + 1)
    return out


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def build_features(events: pd.DataFrame, meta: pd.DataFrame, top_pairs: set | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    events    : сырой events.csv.gz (ещё не очищенный, до drop_duplicates)
    meta      : train или test - cookie_id, cookie_created_at, window_start_ts,
                window_end_ts (target в meta игнорируется, если есть)
    top_pairs : топ-N биграмм для top_bigram_share (см. get_top_pairs). Если
                не передан - считается на events из этого же вызова (ок для
                train целиком и для test относительно train, т.к. они не
                пересекаются по времени/cookie_id). Для temporal CV внутри
                train считайте top_pairs на events ДО cutoff и передавайте
                явно и для train, и для valid части одного fold'а.

    Returns
    -------
    Одна строка признаков на cookie_id, в том же порядке, что и meta.
    Куки без единого события в окне получают числовые признаки = 0,
    флаги = 0, has_events = 0, platform_canon = "none".
    """
    ev_clean = clean_events(events)
    ev_win = events_in_window(ev_clean, meta)

    if top_pairs is None:
        top_pairs = get_top_pairs(ev_win)

    parts = [
        _meta_features(meta),
        _timing_features(ev_win),
        _sequence_features(ev_win, top_pairs),
        _positional_features(ev_win),
        _diversity_features(ev_win),
        _revisit_features(ev_win),
        _conversion_features(ev_win),
        _search_features(ev_win),
        _pointer_features(ev_win),
        _ua_platform_features(ev_win),
    ]

    feats = parts[0]
    for p in parts[1:]:
        feats = feats.merge(p, on="cookie_id", how="left")

    feats["has_events"] = feats["n_events"].notna().astype(int)

    numeric_cols = feats.select_dtypes(include=[np.number]).columns.difference(
        ["cookie_age_days", "cookie_age_log1p"]
    )
    feats[numeric_cols] = feats[numeric_cols].fillna(0)

    for col in ["is_headless", "is_raw_client", "is_bot_ua"]:
        feats[col] = feats[col].fillna(0).astype(int)

    feats["platform_canon"] = feats["platform_canon"].fillna("none")
    feats["first_event_name"] = feats["first_event_name"].fillna("none")

    feats = feats.merge(_interaction_features(feats), on="cookie_id", how="left")

    # порядок строк должен совпадать с meta - гарантируем явно
    feats = meta[["cookie_id"]].merge(feats, on="cookie_id", how="left")
    return feats


if __name__ == "__main__":
    train = pd.read_csv(
        "data/train.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    test = pd.read_csv(
        "data/test.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    events = pd.read_csv("data/events.csv.gz", parse_dates=["event_ts"])

    Xtr = build_features(events, train)
    Xte = build_features(events, test)

    print("train features:", Xtr.shape)
    print("test features:", Xte.shape)
    print(Xtr.head())
    print("\nпропуски после сборки (должно быть 0 везде, кроме cookie_age_days при кривых входных данных):")
    print(Xtr.isna().sum()[lambda s: s > 0])

    Xtr.to_csv("data/features_train.csv", index=False)
    Xte.to_csv("data/features_test.csv", index=False)
