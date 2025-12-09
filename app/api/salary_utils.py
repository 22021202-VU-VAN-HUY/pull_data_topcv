from typing import Optional, Union

Number = Optional[Union[int, float]]


def _to_number(value: Number) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_amount_vnd(value: float) -> str:
    """
    Format VND amount (stored in absolute VND) to a friendly 'X triệu' string.
    Example: 12_000_000 -> '12 triệu'; 12_500_000 -> '12.5 triệu'.
    """
    millions = value / 1_000_000
    if millions.is_integer():
        millions_text = f"{int(millions)}"
    else:
        millions_text = f"{millions:.1f}".rstrip("0").rstrip(".")
    return f"{millions_text} triệu"


def format_salary_text(
    salary_min: Number,
    salary_max: Number,
    currency: Optional[str],
    interval: Optional[str],
    raw_text: Optional[str],
) -> str:
    """
    Build human readable salary text using raw_text first, otherwise fall back to
    min/max with the expected Vietnamese phrasing.
    """
    if raw_text:
        return str(raw_text)

    min_v = _to_number(salary_min)
    max_v = _to_number(salary_max)

    if min_v is None and max_v is None:
        return "Thoả thuận"

    cur = (currency or "VND").upper()
    interval_vi = {
        "MONTH": "/tháng",
        "YEAR": "/năm",
        "HOUR": "/giờ",
    }.get((interval or "").upper(), "")
    suffix = f" {interval_vi}" if interval_vi else ""

    def fmt(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        if cur == "VND":
            return _format_amount_vnd(value)
        return f"{value:,.0f} {cur}"

    if min_v is not None and max_v is not None:
        return f"Từ {fmt(min_v)} đến {fmt(max_v)}{suffix}"
    if min_v is not None:
        return f"Từ {fmt(min_v)}{suffix}"
    return f"Đến {fmt(max_v)}{suffix}"
