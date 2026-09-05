"""Column schema shared across the pipeline (English snake_case names).

The synthetic generator and every analysis module work on these canonical
columns. They mirror the structure of the original research dataset without
carrying any real-world identifiers.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
# Identifier / categorical columns
# ------------------------------------------------------------------ #
UNIT_ID = "unit_id"                # anonymous sequential id (1..n)
PRODUCT_TYPE = "product_type"      # apparel / food / daily_goods
PRICE = "price"                    # CNY listing price
GENDER = "gender"                  # 0 / 1 anonymized binary attribute
TIME_SLOT = "time_slot"            # morning / afternoon / evening / late_night

# ------------------------------------------------------------------ #
# Operational (basic) columns
# ------------------------------------------------------------------ #
FOLLOWERS = "followers_10k"        # total followers, 10k unit
ACTIVE_FOLLOWERS = "active_followers_10k"
STREAM_HOURS = "stream_hours"      # session duration in hours
VIEWERS = "viewers"                # concurrent viewers
SALES_INDEX = "sales_index"        # cumulative sales index
CONVERSION = "conversion_rate"     # 0-1
FAN_ACTIVITY = "fan_activity"      # active/total followers
SALES_PER_VIEWER = "sales_per_viewer"

# ------------------------------------------------------------------ #
# Voice (acoustic) columns
# ------------------------------------------------------------------ #
MEAN_F0 = "mean_f0"                # Hz
RANGE_F0 = "range_f0"              # Hz
SD_F0 = "sd_f0"                    # Hz
HNR = "hnr"                        # dB
JITTER = "jitter"                  # %
SHIMMER = "shimmer"                # %
SPEECH_RATE = "speech_rate"        # chars / second
VOICE_INDEX = "voice_index"        # composite voice quality index

# ------------------------------------------------------------------ #
# Derived / model columns
# ------------------------------------------------------------------ #
CONVERSION_PCT = "conversion_pct"  # conversion_rate * 100
PRICE_TIER = "price_tier"          # low / mid / mid_high / high
BCC_EFF = "bcc_efficiency"
CCR_EFF = "ccr_efficiency"
SCALE_EFF = "scale_efficiency"

# ------------------------------------------------------------------ #
# Canonical orderings used by the analysis modules
# ------------------------------------------------------------------ #
VOICE_VARS: list[str] = [MEAN_F0, HNR, JITTER, SPEECH_RATE]
VOICE_VARS_FULL: list[str] = [
    MEAN_F0, RANGE_F0, SD_F0, HNR, JITTER, SHIMMER, SPEECH_RATE,
]

BASIC_MODEL_INPUTS: list[str] = [STREAM_HOURS, PRICE, VIEWERS]
VOICE_MODEL_INPUTS: list[str] = BASIC_MODEL_INPUTS + [MEAN_F0, HNR, JITTER, SPEECH_RATE]
FULL_MODEL_INPUTS: list[str] = BASIC_MODEL_INPUTS + [
    FOLLOWERS, RANGE_F0, SD_F0, SHIMMER,
] + [MEAN_F0, HNR, JITTER, SPEECH_RATE]

OUTPUTS: list[str] = [SALES_INDEX, CONVERSION_PCT]

VOICE_LABELS: dict[str, str] = {
    MEAN_F0: "Mean F0 (Hz)",
    HNR: "Harmonics-to-Noise (dB)",
    JITTER: "Jitter (%)",
    SPEECH_RATE: "Speech rate (chars/s)",
}

SHAPLEY_GROUPS: dict[str, tuple[str, ...]] = {
    "stream_hours": (STREAM_HOURS,),
    "price": (PRICE,),
    "traffic": (VIEWERS,),
    "voice_f0": (MEAN_F0,),
    "voice_quality": (HNR,),
    "voice_stability": (JITTER,),
    "speech_rate": (SPEECH_RATE,),
}
BASIC_SHAPLEY_GROUPS = ("stream_hours", "price", "traffic")
VOICE_SHAPLEY_GROUPS = ("voice_f0", "voice_quality", "voice_stability", "speech_rate")
