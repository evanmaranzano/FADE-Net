"""Shared ablation profiles for V4-small paper experiments."""

from __future__ import annotations


ABLATION_FIELDS = (
    "use_multi_scale",
    "use_spp",
    "use_texture_branch",
    "use_freq_attention",
    "use_moe",
    "use_adaptive_triplet",
    "use_asymmetric_ordinal",
)

ABLATION_PROFILES = {
    "A0": {"use_multi_scale": False, "use_spp": False},
    "A1": {"use_multi_scale": True, "use_spp": False},
    "A2": {"use_multi_scale": False, "use_spp": True},
    "A3": {"use_multi_scale": True, "use_spp": True},
    "A4": {"use_multi_scale": True, "use_spp": True, "use_texture_branch": True},
    "A5": {"use_multi_scale": True, "use_spp": True, "use_freq_attention": True},
    "A6": {"use_multi_scale": True, "use_spp": True, "use_moe": True},
    "A7": {"use_multi_scale": True, "use_spp": True, "use_adaptive_triplet": True},
    "A8": {"use_multi_scale": True, "use_spp": True, "use_asymmetric_ordinal": True},
    "A9": {
        "use_multi_scale": True,
        "use_spp": True,
        "use_texture_branch": True,
        "use_freq_attention": True,
        "use_moe": True,
        "use_adaptive_triplet": True,
        "use_asymmetric_ordinal": True,
    },
}


def parse_ablation_ids(ablation_arg):
    if not ablation_arg:
        return [None]
    ids = [item.strip().upper() for item in ablation_arg.split(",") if item.strip()]
    unknown = [item for item in ids if item not in ABLATION_PROFILES]
    if unknown:
        raise ValueError(f"Unknown ablation_id(s): {', '.join(unknown)}")
    return ids


def apply_ablation_profile(cfg, ablation_id):
    if not ablation_id:
        return cfg
    if ablation_id not in ABLATION_PROFILES:
        raise KeyError(f"Unknown ablation_id: {ablation_id!r}. Available: {sorted(ABLATION_PROFILES)}")
    for name in ABLATION_FIELDS:
        setattr(cfg, name, False)
    for name, value in ABLATION_PROFILES[ablation_id].items():
        setattr(cfg, name, value)
    return cfg


def ablation_cli_flags(ablation_id):
    if not ablation_id:
        return []

    cfg_flags = dict.fromkeys(ABLATION_FIELDS, False)
    cfg_flags.update(ABLATION_PROFILES[ablation_id])
    flag_names = {
        "use_multi_scale": ("--msff", "--no-msff"),
        "use_spp": ("--spp", "--no-spp"),
        "use_texture_branch": ("--texture", "--no-texture"),
        "use_freq_attention": ("--freq", "--no-freq"),
        "use_moe": ("--moe", "--no-moe"),
        "use_adaptive_triplet": ("--triplet", "--no-triplet"),
        "use_asymmetric_ordinal": ("--asym", "--no-asym"),
    }
    flags = []
    for name in ABLATION_FIELDS:
        enabled_flag, disabled_flag = flag_names[name]
        flags.append(enabled_flag if cfg_flags[name] else disabled_flag)
    return flags


def ablation_row_flags(cfg):
    return {name: bool(getattr(cfg, name, False)) for name in ABLATION_FIELDS}
