from dateutil import parser as dateparser

def bucket_by_time(chunks: list, num_buckets: int = 3):
    dated = [c for c in chunks if c["date"] != "unknown-date"]
    dated.sort(key=lambda c: dateparser.parse(c["date"]))
    if not dated:
        return {"early": [], "mid": [], "late": []}
    size = max(1, len(dated) // num_buckets)
    return {
        "early": dated[:size],
        "mid": dated[size:size * 2] if len(dated) > size else [],
        "late": dated[size * 2:] if len(dated) > size * 2 else dated[size:],
    }

def drift_summary(supporting: list, opposing: list) -> str:
    support_buckets = bucket_by_time(supporting)
    oppose_buckets = bucket_by_time(opposing)

    def dominant_bucket(buckets):
        counts = {k: len(v) for k, v in buckets.items()}
        return max(counts, key=counts.get) if any(counts.values()) else "none"

    support_peak = dominant_bucket(support_buckets)
    oppose_peak = dominant_bucket(oppose_buckets)

    if support_peak == "none" or oppose_peak == "none":
        return "Not enough dated evidence to assess drift."
    if support_peak == oppose_peak:
        return f"No clear drift — both views cluster in the '{support_peak}' period."
    return (
        f"Possible belief drift: supporting evidence clusters in the "
        f"'{support_peak}' period, while opposing evidence clusters in the "
        f"'{oppose_peak}' period."
    )