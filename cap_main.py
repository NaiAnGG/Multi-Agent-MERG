# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import pandas as pd
from tqdm import tqdm

from cap_agents import (
    LLMConfig,
    OllamaGenerateClient,
    agent_critic,
    agent_generator,
    agent_multimodal_analysis,
    agent_scene_integration,
    agent_strategy_planner,
    normalize_iemocap10_label,
    normalize_sentiment3_label,
    sanitize_scene_output,
    sample_video_tail_quarter_keyframes,
    to_images_b64,
)
from eval_qwen7_compat import ResponseEvaluator, summarize_eval_dicts

IEMOCAP10 = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "frustrated",
    "excited",
    "fearful",
    "disgusted",
    "surprised",
    "other",
]

EMOTION_ALIAS = {
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "hap": "happy",
    "happiness": "happy",
    "happy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "neutral": "neutral",
    "neutral state": "neutral",
    "neu": "neutral",
    "fru": "frustrated",
    "frustration": "frustrated",
    "frustrated": "frustrated",
    "exc": "excited",
    "excited": "excited",
    "fea": "fearful",
    "fear": "fearful",
    "fearful": "fearful",
    "dis": "disgusted",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "sur": "surprised",
    "surprise": "surprised",
    "surprised": "surprised",
    "oth": "other",
    "other": "other",
    "unk": "other",
    "unknown": "other",
}

EMOTION_TO_SENTIMENT = {
    "angry": "negative",
    "sad": "negative",
    "frustrated": "negative",
    "fearful": "negative",
    "disgusted": "negative",
    "happy": "positive",
    "excited": "positive",
    "neutral": "neutral",
    "surprised": "neutral",
    "other": "neutral",
}

_AGENT_ALIAS = {
    "mm": "mm",
    "agent1": "mm",
    "a1": "mm",
    "multimodal": "mm",
    "scene": "scene",
    "agent2": "scene",
    "a2": "scene",
    "emotion": "scene",
    "strategy": "strategy",
    "agent3": "strategy",
    "a3": "strategy",
    "planner": "strategy",
    "generator": "generator",
    "agent4": "generator",
    "a4": "generator",
    "response": "generator",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--video_dir", type=str, required=True)
    ap.add_argument("--oai_config", type=str, default="./OAI_CONFIG_LIST")
    ap.add_argument("--llm_name", type=str, default="")
    ap.add_argument("--sessions", type=str, default="", help='')
    ap.add_argument("--max_dialogues", type=int, default=1000)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--history_turns", type=int, default=50)
    ap.add_argument("--dialogue_start", type=int, default=None, help="1-based inclusive index after sorting selected dialogues.")
    ap.add_argument("--dialogue_end", type=int, default=None, help="1-based inclusive index after sorting selected dialogues.")
    ap.add_argument("--require_video", action="store_true", help="If set, skip dialogues whose renamed long video does not exist.")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--tail_ratio", type=float, default=0.25)
    ap.add_argument("--max_refine_iter", type=int, default=2)
    ap.add_argument("--save_path", type=str, default="")
    ap.add_argument("--summary_path", type=str, default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--print_every", type=int, default=1)
    ap.add_argument("--save_light", action="store_true")
    ap.add_argument("--eval_device", type=str, default="")
    return ap.parse_args()


def load_oai_config_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_model_config(oai_list: List[Dict[str, Any]], llm_name: str) -> Tuple[str, str, str]:
    for c in oai_list:
        if c.get("model") == llm_name:
            return c.get("base_url"), c.get("model"), c.get("api_key", "ollama")
    c0 = oai_list[0]
    return c0.get("base_url"), c0.get("model"), c0.get("api_key", "ollama")


def ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def canonical_agent_name(x: Any) -> str:
    s = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _AGENT_ALIAS.get(s, s)


def normalize_refine_agents(x: Any) -> List[str]:
    allow = {"mm", "scene", "strategy", "generator"}
    out: List[str] = []
    for item in ensure_list(x):
        item = canonical_agent_name(item)
        if item in allow and item not in out:
            out.append(item)
    return out


def _llm_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", "", "none", "null"}:
        return False
    return False


def parse_sessions_filter(raw: str) -> Optional[set]:
    if not raw or not str(raw).strip():
        return None
    out = set()
    for item in str(raw).split(","):
        s = item.strip().lower()
        if not s:
            continue
        if s.startswith("session"):
            out.add(s)
        else:
            out.add(f"session{int(s)}")
    return out or None


def dialogue_sort_key(session_id: str, dialogue_name: str) -> Tuple[int, int, int, str]:
    sm = re.search(r"(\d+)", str(session_id or ""))
    sidx = int(sm.group(1)) if sm else 10**9
    m = re.match(r"^\s*(\d+)_(\d+)\s*$", str(dialogue_name or ""))
    if m:
        return (sidx, int(m.group(1)), int(m.group(2)), str(dialogue_name))
    dm = re.findall(r"\d+", str(dialogue_name or ""))
    if len(dm) >= 2:
        return (sidx, int(dm[0]), int(dm[1]), str(dialogue_name))
    if len(dm) == 1:
        return (sidx, int(dm[0]), 0, str(dialogue_name))
    return (sidx, 10**9, 10**9, str(dialogue_name))


def normalize_emotion_token(x: Any) -> Optional[str]:
    s = str(x or "").strip().lower()
    if not s:
        return None
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return EMOTION_ALIAS.get(s, s if s in IEMOCAP10 else None)


def parse_emotion_candidates(raw: Any) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return ["other"]
    parts = re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*", text)
    out: List[str] = []
    seen = set()
    for part in parts:
        norm = normalize_emotion_token(part)
        if norm and norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out or ["other"]


def resolve_emotion_label(candidates: Sequence[str]) -> str:
    cands = [str(x).strip().lower() for x in (candidates or []) if str(x).strip()]
    if not cands:
        return "other"
    non_other = [x for x in cands if x != "other"]
    if non_other:
        cands = non_other
    if any(x != "neutral" for x in cands):
        cands = [x for x in cands if x != "neutral"] or cands
    return cands[0] if cands else "other"


def emotion_to_sentiment(emotion: Any) -> str:
    if isinstance(emotion, str) and emotion in IEMOCAP10:
        emo = emotion
    else:
        emo = resolve_emotion_label(parse_emotion_candidates(emotion))
    return EMOTION_TO_SENTIMENT.get(str(emo).strip().lower(), "neutral")


def format_context(turns: List[Dict[str, Any]], dialogue_name: str, target_speaker: str) -> str:
    lines = [f"DialogueID: {dialogue_name}"]
    for t in turns:
        speaker = str(t.get("speaker", "Unknown")).strip()
        utt = str(t.get("utterance", "")).strip().replace("\n", " ")
        lines.append(f"{speaker}: {utt}")
    lines.append(f"Target speaker: {target_speaker}")
    return "\n".join(lines)

def format_context_with_labels(
    turns: List[Dict[str, Any]],
    dialogue_name: str,
    target_speaker: str,
    sample_id: Optional[str] = None,
) -> str:
    sid = str(sample_id or dialogue_name).strip()
    lines = [
        f"SampleID: {sid}",
        f"DialogueID: {dialogue_name}",
    ]
    for t in turns:
        sent_id = int(t.get("sentence_id", -1))
        speaker = str(t.get("speaker", "Unknown")).strip()
        utt = str(t.get("utterance", "")).strip().replace("\n", " ")
        lines.append(f"{sent_id}. {speaker}: {utt}")
    lines.append(f"Target speaker: {target_speaker}")
    return "\n".join(lines)

# def format_context_with_labels(turns: List[Dict[str, Any]], dialogue_name: str, target_speaker: str) -> str:
#     lines = [f"DialogueID: {dialogue_name}"]
#     for t in turns:
#         speaker = str(t.get("speaker", "Unknown")).strip()
#         utt = str(t.get("utterance", "")).strip().replace("\n", " ")
#         emo_cands = ensure_list(t.get("emotion_candidates"))
#         emo_text = ", ".join(emo_cands) if emo_cands else "other"
#         lines.append(f"{speaker}: {utt} ({emo_text})")
#     lines.append(f"Target speaker: {target_speaker}")
#     return "\n".join(lines)


def build_video_path(video_dir: str, dialogue_name: str) -> str:
    return os.path.join(video_dir, f"{dialogue_name}.avi")


def select_samples_by_index(samples: List[Dict[str, Any]], start: Optional[int], end: Optional[int], max_samples: int) -> List[Dict[str, Any]]:
    s = 1 if start is None else max(1, int(start))
    e = len(samples) if end is None else min(len(samples), int(end))
    if e < s:
        return []
    picked = samples[s - 1:e]
    return picked[: max(0, int(max_samples))]


def build_runtime_samples_from_csv(
    csv_path: str,
    video_dir: str,
    sessions_filter: Optional[set],
    max_dialogues: int,
    max_samples: int,
    history_turns: int,
    dialogue_start: Optional[int],
    dialogue_end: Optional[int],
    require_video: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    needed_cols = {"dialogue_name", "sentence_id", "speaker", "utterance", "emotion", "session_id"}
    missing = sorted(needed_cols - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.copy()
    df["dialogue_name"] = df["dialogue_name"].astype(str).str.strip()
    df["session_id"] = df["session_id"].astype(str).str.strip().str.lower()
    df["speaker"] = df["speaker"].fillna("Unknown").astype(str).str.strip()
    df["utterance"] = df["utterance"].fillna("").astype(str)
    df["emotion"] = df["emotion"].fillna("").astype(str)
    df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce")
    df = df.dropna(subset=["sentence_id"]) 
    df["sentence_id"] = df["sentence_id"].astype(int)

    if sessions_filter:
        df = df[df["session_id"].isin(sessions_filter)].copy()

    grouped_keys = sorted(
        {(row["session_id"], row["dialogue_name"]) for _, row in df[["session_id", "dialogue_name"]].drop_duplicates().iterrows()},
        key=lambda x: dialogue_sort_key(x[0], x[1]),
    )
    grouped_keys = grouped_keys[: max(0, int(max_dialogues))]
    selected_keys = select_samples_by_index(
        [{"session_id": s, "dialogue_name": d} for s, d in grouped_keys],
        dialogue_start,
        dialogue_end,
        max_samples,
    )
    selected_pairs = [(x["session_id"], x["dialogue_name"]) for x in selected_keys]

    samples: List[Dict[str, Any]] = []
    missing_video_count = 0
    no_history_count = 0

    for session_id, dialogue_name in selected_pairs:
        g = df[(df["session_id"] == session_id) & (df["dialogue_name"] == dialogue_name)].sort_values("sentence_id")
        rows = g.to_dict("records")
        if len(rows) < 2:
            no_history_count += 1
            continue

        turns: List[Dict[str, Any]] = []
        for row in rows:
            emo_candidates = parse_emotion_candidates(row.get("emotion"))
            emo_resolved = resolve_emotion_label(emo_candidates)
            sent_candidates: List[str] = []
            for emo in emo_candidates:
                s = emotion_to_sentiment(emo)
                if s not in sent_candidates:
                    sent_candidates.append(s)
            turns.append({
                "sentence_id": int(row["sentence_id"]),
                "speaker": str(row.get("speaker", "Unknown")).strip(),
                "utterance": str(row.get("utterance", "")).strip(),
                "emotion_raw": str(row.get("emotion", "")).strip(),
                "emotion_candidates": emo_candidates,
                "emotion": emo_resolved,
                "sentiment": emotion_to_sentiment(emo_resolved),
                "sentiment_candidates": sent_candidates,
                "session_id": session_id,
                "dialogue_name": dialogue_name,
            })

        target_turn = turns[-1]
        history_all = turns[:-1]
        history_used = history_all[-history_turns:] if history_turns > 0 else history_all

        video_path = build_video_path(video_dir, dialogue_name)
        video_exists = os.path.exists(video_path)
        if require_video and not video_exists:
            missing_video_count += 1
            continue
        if not video_exists:
            missing_video_count += 1
            video_path = ""

        target_speaker = str(target_turn.get("speaker") or "Unknown").strip()
        gt_emotion_candidates = ensure_list(target_turn.get("emotion_candidates"))
        gt_sentiment_candidates = ensure_list(target_turn.get("sentiment_candidates"))

        samples.append({
            "sample_id": dialogue_name,
            "session_id": session_id,
            "dialogue_name": dialogue_name,
            "video_path": video_path,
            "video_exists": video_exists,
            "num_turns_total": len(turns),
            "history_turns_count": len(history_used),
            "context_turns": history_used,
            "context": format_context(history_used, dialogue_name, target_speaker),
            "context_with_labels": format_context_with_labels(
                history_used,
                dialogue_name=dialogue_name,
                target_speaker=target_speaker,
                sample_id=dialogue_name,
            ),
            "context_plain": format_context_with_labels(
                history_used,
                dialogue_name=dialogue_name,
                target_speaker=target_speaker,
                sample_id=dialogue_name,
            ),
            # "context_with_labels": format_context_with_labels(history_used, dialogue_name, target_speaker),
            "target_speaker": target_speaker,
            "target_sentence_id": int(target_turn.get("sentence_id") or -1),
            "gt_response": str(target_turn.get("utterance") or "").strip(),
            "gt_emotion": str(target_turn.get("emotion") or "").strip(),
            "gt_sentiment": str(target_turn.get("sentiment") or "").strip(),
            "gt_emotion_raw": str(target_turn.get("emotion_raw") or "").strip(),
            "gt_emotion_candidates": gt_emotion_candidates,
            "gt_sentiment_candidates": gt_sentiment_candidates,
            "agent_context_used": "compact_context_without_history_emotion_labels",
        })

    build_meta = {
        "csv_path": csv_path,
        "video_dir": video_dir,
        "sessions_filter": sorted(list(sessions_filter)) if sessions_filter else None,
        "num_dialogues_selected_before_skip": len(selected_pairs),
        "num_samples_built": len(samples),
        "missing_video_count": missing_video_count,
        "no_history_count": no_history_count,
        "history_turns": history_turns,
        "require_video": require_video,
        "history_emotion_labels_in_context": False,
    }
    return samples, build_meta


def _compose_scene_feedback(scene_verdict: Dict[str, Any]) -> str:
    corrected_sent = normalize_sentiment3_label(scene_verdict.get("corrected_sentiment"))
    if corrected_sent == "neutral":
        emo2 = normalize_iemocap10_label(scene_verdict.get("corrected_emotion"))
        corrected_sent = emotion_to_sentiment(emo2)
    corrected_emo = normalize_iemocap10_label(scene_verdict.get("corrected_emotion"))
    latest_evi = ensure_list(scene_verdict.get("latest_trigger_evidence"))
    history_evi = ensure_list(scene_verdict.get("history_evidence")) or ensure_list(scene_verdict.get("continuity_evidence")) or ensure_list(scene_verdict.get("prior_context_evidence"))
    latest_text = "; ".join(latest_evi[:2]) if latest_evi else "re-check the latest turn"
    history_text = "; ".join(history_evi[:2]) if history_evi else "re-check the target speaker's prior turns"
    return (
        f"Corrected sentiment={corrected_sent}; corrected emotion={corrected_emo}; "
        f"latest-trigger evidence={latest_text}; history evidence={history_text}."
    )


def _default_generator_feedback() -> str:
    return (
        "Revise Agent4. Keep target-speaker perspective, follow Agent2 emotion and Agent3 plan, "
        "address the latest trigger directly, include at least one concrete semantic anchor from the local context, "
        "and avoid generic safe replies that could fit many different dialogues."
    )

def _compose_generator_feedback_from_checks(checks: Dict[str, Any], raw_feedback: str = "") -> str:
    parts: List[str] = []
    raw_feedback = str(raw_feedback or "").strip()
    if raw_feedback:
        parts.append(raw_feedback)

    if not _llm_bool(checks.get("reply_addresses_latest_trigger", True)):
        parts.append("The reply does not address the latest trigger clearly enough.")

    if not _llm_bool(checks.get("reply_is_semantically_specific", True)):
        parts.append("The reply is too abstract; add one concrete semantic anchor from the immediate local context.")

    if _llm_bool(checks.get("reply_is_generic_safe_answer", False)):
        parts.append("The reply is too generic and could fit many different contexts; avoid safe filler replies.")

    if not _llm_bool(checks.get("generator_matches_scene", True)):
        parts.append("Keep the line aligned with Agent2's predicted emotion.")

    if not _llm_bool(checks.get("response_naturalness", True)):
        parts.append("Keep the line natural and spoken, not stiff or over-explained.")

    if not parts:
        return _default_generator_feedback()

    return " ".join(parts)

def normalize_critic_output(x: Any) -> Dict[str, Any]:
    if not isinstance(x, dict):
        x = {}

    summary = str(x.get("overall_summary") or x.get("summary") or "").strip()

    raw_checks = x.get("checks", {})
    if not isinstance(raw_checks, dict):
        raw_checks = {}

    scene_verdict = x.get("scene_verdict", {})
    if not isinstance(scene_verdict, dict):
        scene_verdict = {}

    checks = {
        "scene_label_supported_by_dialogue": _llm_bool(raw_checks.get("scene_label_supported_by_dialogue", True)),
        "scene_label_supported_by_latest_trigger": _llm_bool(raw_checks.get("scene_label_supported_by_latest_trigger", True)),
        "generator_matches_scene": _llm_bool(raw_checks.get("generator_matches_scene", True)),
        "conflict_handled": _llm_bool(raw_checks.get("conflict_handled", True)),
        "strategy_is_appropriate": _llm_bool(raw_checks.get("strategy_is_appropriate", True)),
        "strategy_is_followed": _llm_bool(raw_checks.get("strategy_is_followed", True)),
        "coherence_and_persona": _llm_bool(raw_checks.get("coherence_and_persona", True)),
        "response_naturalness": _llm_bool(raw_checks.get("response_naturalness", True)),

        # ÐÂÔöµÄ 3 ¸ö generator ÓïÒå¼ì²é
        "reply_addresses_latest_trigger": _llm_bool(raw_checks.get("reply_addresses_latest_trigger", True)),
        "reply_is_semantically_specific": _llm_bool(raw_checks.get("reply_is_semantically_specific", True)),
        "reply_is_generic_safe_answer": _llm_bool(raw_checks.get("reply_is_generic_safe_answer", False)),
    }

    need_refine = _llm_bool(x.get("need_refine", False))
    mm_feedback = str(x.get("mm_feedback", "") or "").strip()
    scene_feedback = str(x.get("scene_feedback", "") or "").strip()
    strategy_feedback = str(x.get("strategy_feedback", "") or "").strip()
    generator_feedback = str(x.get("generator_feedback", "") or "").strip()
    next_constraints = ensure_list(x.get("next_constraints"))
    refine_agents = normalize_refine_agents(x.get("refine_agents"))

    primary_failure_source = canonical_agent_name(x.get("primary_failure_source", "none"))
    if primary_failure_source not in {"none", "mm", "scene", "strategy", "generator"}:
        primary_failure_source = "none"

    if primary_failure_source == "scene":
        checks["scene_label_supported_by_dialogue"] = False
        checks["scene_label_supported_by_latest_trigger"] = False
        if isinstance(scene_verdict, dict):
            scene_verdict["supported_by_dialogue"] = False
            scene_verdict["supported_by_latest_trigger"] = False

    scene_is_wrong = (
        (not checks["scene_label_supported_by_dialogue"])
        or (not checks["scene_label_supported_by_latest_trigger"])
    )

    generator_semantic_weak = (
        (not checks["reply_addresses_latest_trigger"])
        or (not checks["reply_is_semantically_specific"])
        or checks["reply_is_generic_safe_answer"]
    )


    if scene_is_wrong:
        need_refine = True
        primary_failure_source = "scene"
        if not scene_feedback:
            scene_feedback = _compose_scene_feedback(scene_verdict)
        if "scene" not in refine_agents:
            refine_agents = ["scene"] + [a for a in refine_agents if a != "scene"]


    if generator_semantic_weak and not scene_is_wrong:
        need_refine = True
        if primary_failure_source == "none":
            primary_failure_source = "generator"
        if "generator" not in refine_agents:
            refine_agents.append("generator")
        generator_feedback = _compose_generator_feedback_from_checks(checks, generator_feedback)

        if not any("latest trigger" in str(x).lower() for x in next_constraints):
            next_constraints.append("Address the latest trigger directly.")
        if not any("semantic anchor" in str(x).lower() for x in next_constraints):
            next_constraints.append("Include one concrete semantic anchor from the immediate local context.")
        if checks["reply_is_generic_safe_answer"] and not any("generic" in str(x).lower() for x in next_constraints):
            next_constraints.append("Avoid generic safe replies that could fit many dialogues.")

    if need_refine and not refine_agents:
        refine_agents = [primary_failure_source if primary_failure_source != "none" else "generator"]

    if need_refine and "generator" in refine_agents and not generator_feedback:
        generator_feedback = _default_generator_feedback()

    if not need_refine:
        primary_failure_source = "none"
        refine_agents = []
        mm_feedback = ""
        scene_feedback = ""
        strategy_feedback = ""
        generator_feedback = ""
        next_constraints = []

    out = dict(x)
    out.update({
        "need_refine": need_refine,
        "overall_summary": summary,
        "checks": checks,
        "scene_verdict": scene_verdict,
        "primary_failure_source": primary_failure_source,
        "refine_agents": refine_agents,
        "mm_feedback": mm_feedback if "mm" in refine_agents else "",
        "scene_feedback": scene_feedback if "scene" in refine_agents else "",
        "strategy_feedback": strategy_feedback if "strategy" in refine_agents else "",
        "generator_feedback": generator_feedback if "generator" in refine_agents else "",
        "next_constraints": next_constraints if "generator" in refine_agents else [],
        "scene_veto_triggered": scene_is_wrong,
        "generator_semantic_weak": generator_semantic_weak,
    })
    return out

def _iteration_choice_tuple(bundle: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int, int, int, int, int]:
    critic = bundle.get("critic") or {}
    checks = critic.get("checks") or {}
    it = int(bundle.get("iter", 0))
    return (
        int(not _llm_bool(critic.get("need_refine", False))),
        int(not _llm_bool(critic.get("scene_veto_triggered", False))),
        int(_llm_bool(checks.get("scene_label_supported_by_dialogue", False))),
        int(_llm_bool(checks.get("scene_label_supported_by_latest_trigger", False))),
        int(_llm_bool(checks.get("strategy_is_appropriate", False))),
        int(_llm_bool(checks.get("strategy_is_followed", False))),
        int(_llm_bool(checks.get("generator_matches_scene", False))),
        int(_llm_bool(checks.get("reply_addresses_latest_trigger", False))),
        int(_llm_bool(checks.get("reply_is_semantically_specific", False))),
        int(not _llm_bool(checks.get("reply_is_generic_safe_answer", True))),
        int(_llm_bool(checks.get("response_naturalness", False))),
        -it,
    )

def select_best_iteration_bundle(bundles: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    if not bundles:
        return -1, {}
    best_idx = 0
    best_score = _iteration_choice_tuple(bundles[0])
    for idx in range(1, len(bundles)):
        score = _iteration_choice_tuple(bundles[idx])
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, bundles[best_idx]


def extract_scene_pred_label(scene_json: Dict[str, Any]) -> Optional[str]:
    if not isinstance(scene_json, dict):
        return None
    pred = scene_json.get("predicted_target_emotion")
    if isinstance(pred, dict):
        return normalize_iemocap10_label(pred.get("label"))
    legacy = scene_json.get("predicted_next_emotion")
    if legacy is not None:
        return normalize_iemocap10_label(legacy)
    return None


def extract_scene_pred_sentiment(scene_json: Dict[str, Any]) -> Optional[str]:

    return None


def extract_response(gen: Any) -> str:
    if isinstance(gen, dict):
        return str(gen.get("response", "") or "").strip()
    return str(gen or "").strip()


def maybe_print_eval(args: argparse.Namespace, sample_idx: int, it: int, eval_json: Dict[str, Any]) -> None:
    if args.quiet or args.print_every <= 0 or (sample_idx % args.print_every != 0):
        return
    print(
        f"[Eval] sample#{sample_idx} iter#{it} | "
        f"PPL={eval_json.get('PPL')} | "
        f"SceneEmotion={eval_json.get('ScenePredEmotion')} | "
        f"GTCandidates={eval_json.get('GTEmotionCandidates')} | "
        f"EmotionAcc={eval_json.get('EmotionAcc')} | "

    )
    if eval_json.get("PPL") is None:
        print(f"[Eval-Note] PPL unavailable: {eval_json.get('PPL_error')}")


def print_sample_header(args: argparse.Namespace, sample_idx: int, sample: Dict[str, Any], visual_meta: Dict[str, Any]) -> None:
    if args.quiet or args.print_every <= 0 or (sample_idx % args.print_every != 0):
        return
    print("\n" + "=" * 14 + f" SAMPLE #{sample_idx} " + "=" * 14)
    print(
        f"[Sample] dialogue={sample.get('dialogue_name')} | session={sample.get('session_id')} | "
        f"target_speaker={sample.get('target_speaker')} | target_sid={sample.get('target_sentence_id')} | "
        f"history_turns={sample.get('history_turns_count')}"
    )
    print(
        f"[Visual] video_exists={visual_meta.get('video_exists')} | frames={visual_meta.get('num_sampled_frames')} | "
        f"num_frames_request={visual_meta.get('num_frames_request')} | tail_ratio={visual_meta.get('tail_ratio')}"
    )


def main() -> None:
    args = parse_args()

    sessions_filter = parse_sessions_filter(args.sessions)
    samples, build_meta = build_runtime_samples_from_csv(
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        sessions_filter=sessions_filter,
        max_dialogues=args.max_dialogues,
        max_samples=args.max_samples,
        history_turns=args.history_turns,
        dialogue_start=args.dialogue_start,
        dialogue_end=args.dialogue_end,
        require_video=args.require_video,
    )
    if not samples:
        raise RuntimeError("No samples built. Check --sessions, --dialogue_start/--dialogue_end, CSV content, or video path settings.")

    if not args.quiet:
        print("\n========== BUILD META ==========")
        print(json.dumps(build_meta, ensure_ascii=False, indent=2))

    oai_list = load_oai_config_list(args.oai_config)
    base_url, model, _api_key = pick_model_config(oai_list, args.llm_name)
    client = OllamaGenerateClient(LLMConfig(base_url=base_url, model=model))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    summary_path = args.summary_path or args.save_path.replace(".jsonl", ".summary.json")
    t0 = time.time()
    print("[TRACE] before evaluator", flush=True)
    evaluator = ResponseEvaluator(device=(args.eval_device if args.eval_device else None))
    print(f"[TRACE] after evaluator | cost={time.time() - t0:.2f}s", flush=True)

    final_eval_list: List[Dict[str, Any]] = []

    with open(args.save_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(tqdm(samples, desc="Running revised 5-agent pipeline"), start=1):
            target_speaker = str(s.get("target_speaker") or "Unknown").strip()
            gt_text = str(s.get("gt_response") or "").strip()
            gt_emotion = str(s.get("gt_emotion") or "").strip()
            gt_sentiment = str(s.get("gt_sentiment") or "").strip()
            gt_emotion_candidates = ensure_list(s.get("gt_emotion_candidates"))
            gt_sentiment_candidates = ensure_list(s.get("gt_sentiment_candidates"))
            agent_context = str(s.get("context_plain") or s.get("context_with_labels") or s.get("context") or "").strip()

            iter_logs: List[Dict[str, Any]] = []
            eval_logs: List[Dict[str, Any]] = []
            critic_logs: List[Dict[str, Any]] = []
            iteration_bundles: List[Dict[str, Any]] = []

            video_path = str(s.get("video_path") or "").strip()
            frames_jpeg = sample_video_tail_quarter_keyframes(
                video_path=video_path,
                num_frames=args.num_frames,
                tail_ratio=args.tail_ratio,
            ) if video_path else []
            shared_images_b64 = to_images_b64(frames_jpeg) if frames_jpeg else None
            shared_visual_meta = {
                "video_path": video_path,
                "video_exists": bool(video_path),
                "num_sampled_frames": len(frames_jpeg),
                "num_frames_request": int(args.num_frames),
                "tail_ratio": float(args.tail_ratio),
            }
            print_sample_header(args, i, s, shared_visual_meta)

            mm = agent_multimodal_analysis(
                client=client,
                context_labeled=agent_context,
                target_speaker=target_speaker,
                user_video_path=None,
                num_frames=args.num_frames,
                critic_feedback="",
                previous_output=None,
                images_b64=shared_images_b64,
            )

            scene = agent_scene_integration(
                client=client,
                context_labeled=agent_context,
                mm_json=mm,
                target_speaker=target_speaker,
                user_video_path=None,
                num_frames=args.num_frames,
                critic_feedback="",
                previous_output=None,
                images_b64=shared_images_b64,
            ) if mm else {}
            scene = sanitize_scene_output(scene)

            stra = agent_strategy_planner(
                client=client,
                context_labeled=agent_context,
                mm_json=mm,
                scene_json=scene,
                target_speaker=target_speaker,
                critic_feedback="",
                previous_output=None,
            )

            previous_response = ""
            generator_feedback = ""
            extra_constraints: List[str] = []
            final_gen: Dict[str, Any] = {}
            final_eval: Dict[str, Any] = {}
            final_critic: Dict[str, Any] = {}

            for it in range(args.max_refine_iter + 1):
                gen = agent_generator(
                    client=client,
                    context_labeled=agent_context,
                    mm_json=mm,
                    scene_json=scene,
                    strategy_json=stra,
                    target_speaker=target_speaker,
                    user_video_path=None,
                    num_frames=args.num_frames,
                    extra_constraints=extra_constraints,
                    previous_response=previous_response,
                    critic_feedback=generator_feedback,
                    need_logprobs=True,
                    images_b64=shared_images_b64,
                )
                resp = extract_response(gen)
                gen_meta = gen.get("_generation_meta") if isinstance(gen, dict) else None
                gen_for_critic = {"response": resp}
                gen_for_log = {"response": resp}
                if isinstance(gen_meta, dict):
                    gen_for_log["_generation_meta"] = {
                        "model": gen_meta.get("model"),
                        "done_reason": gen_meta.get("done_reason"),
                        "prompt_eval_count": gen_meta.get("prompt_eval_count"),
                        "eval_count": gen_meta.get("eval_count"),
                    }

                scene_label = extract_scene_pred_label(scene)
                scene_sent = extract_scene_pred_sentiment(scene)
                eval_json = evaluator.evaluate_one(
                    pred=resp,
                    ref=gt_text,
                    gt_emotion=gt_emotion,
                    gt_emotion_candidates=gt_emotion_candidates,
                    scene_pred_emotion=scene_label,
                    gt_sentiment=gt_sentiment,
                    gt_sentiment_candidates=gt_sentiment_candidates,
                    scene_pred_sentiment=scene_sent,
                    generation_meta=gen_meta,
                )
                maybe_print_eval(args, i, it, eval_json)

                critic = agent_critic(
                    client=client,
                    context_labeled=agent_context,
                    target_speaker=target_speaker,
                    user_video_path=None,
                    num_frames=args.num_frames,
                    mm_json=mm,
                    scene_json=scene,
                    strategy_json=stra,
                    generator_json=gen_for_critic,
                    images_b64=shared_images_b64,
                )
                critic = normalize_critic_output(critic)
                if not args.quiet and args.print_every > 0 and (i % args.print_every == 0):
                    print(
                        f"[Critic] iter#{it} | need_refine={critic.get('need_refine')} | "
                        f"primary_failure={critic.get('primary_failure_source')} | "
                        f"refine_agents={critic.get('refine_agents')}"
                    )

                iter_logs.append({"iter": it, "generator": gen_for_log, "response": resp})
                eval_logs.append({"iter": it, "eval": eval_json})
                critic_logs.append({"iter": it, "critic": critic})
                iteration_bundles.append({
                    "iter": it,
                    "mm": json.loads(json.dumps(mm, ensure_ascii=False)),
                    "scene": json.loads(json.dumps(scene, ensure_ascii=False)),
                    "strategy": json.loads(json.dumps(stra, ensure_ascii=False)),
                    "generator": {"response": resp},
                    "eval": json.loads(json.dumps(eval_json, ensure_ascii=False)),
                    "critic": json.loads(json.dumps(critic, ensure_ascii=False)),
                })

                final_gen = {"response": resp}
                final_eval = eval_json
                final_critic = critic

                need_refine = _llm_bool(critic.get("need_refine", False))
                if (not need_refine) or (it >= args.max_refine_iter):
                    break

                refine_agents = normalize_refine_agents(critic.get("refine_agents"))
                mm_changed = False
                scene_changed = False
                strategy_changed = False

                if "mm" in refine_agents:
                    mm_prev = mm
                    mm = agent_multimodal_analysis(
                        client=client,
                        context_labeled=agent_context,
                        target_speaker=target_speaker,
                        user_video_path=None,
                        num_frames=args.num_frames,
                        critic_feedback=str(critic.get("mm_feedback", "")),
                        previous_output=mm_prev,
                        images_b64=shared_images_b64,
                    )
                    mm_changed = True

                if mm_changed or ("scene" in refine_agents):
                    scene_prev = scene
                    scene = agent_scene_integration(
                        client=client,
                        context_labeled=agent_context,
                        mm_json=mm,
                        target_speaker=target_speaker,
                        user_video_path=None,
                        num_frames=args.num_frames,
                        critic_feedback=str(critic.get("scene_feedback", "")),
                        previous_output=scene_prev,
                        images_b64=shared_images_b64,
                    )
                    scene = sanitize_scene_output(scene)
                    scene_changed = True

                if mm_changed or scene_changed or ("strategy" in refine_agents):
                    stra_prev = stra
                    stra = agent_strategy_planner(
                        client=client,
                        context_labeled=agent_context,
                        mm_json=mm,
                        scene_json=scene,
                        target_speaker=target_speaker,
                        critic_feedback=str(critic.get("strategy_feedback", "")),
                        previous_output=stra_prev,
                    )
                    strategy_changed = True

                generator_feedback = str(critic.get("generator_feedback", "")).strip()
                extra_constraints = ensure_list(critic.get("next_constraints"))
                previous_response = resp

                if (not strategy_changed) and ("generator" not in refine_agents) and (not generator_feedback) and (not extra_constraints):
                    break

            best_iter_idx, best_bundle = select_best_iteration_bundle(iteration_bundles)
            if best_bundle:
                mm = best_bundle.get("mm", mm)
                scene = best_bundle.get("scene", scene)
                stra = best_bundle.get("strategy", stra)
                final_gen = best_bundle.get("generator", final_gen)
                final_eval = best_bundle.get("eval", final_eval)
                final_critic = best_bundle.get("critic", final_critic)
            final_eval_list.append(final_eval)

            if args.save_light:
                record = {
                    "sample_id": s.get("sample_id"),
                    "session_id": s.get("session_id"),
                    "dialogue_name": s.get("dialogue_name"),
                    "target_speaker": target_speaker,
                    "context_plain": s.get("context_plain") or s.get("context_with_labels"),
                    "gt_response": gt_text,
                    "gt_emotion": gt_emotion,
                    "gt_emotion_candidates": gt_emotion_candidates,
                    "generator_final": final_gen,
                    "eval_final": final_eval,
                    "critic_final": final_critic,
                }
            else:
                record = {
                    **s,
                    "multimodal_analysis": mm,
                    "scene_integration": scene,
                    "strategy_planner": stra,
                    "best_iter": best_iter_idx,
                    "generator_final": final_gen,
                    "eval_final": final_eval,
                    "critic_final": final_critic,
                    "iter_logs": iter_logs,
                    "eval_logs": eval_logs,
                    "critic_logs": critic_logs,
                    "iteration_bundles": iteration_bundles,
                    "shared_visual_meta": shared_visual_meta,
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = summarize_eval_dicts(evaluator, final_eval_list)
        summary_record = {
            "record_type": "summary",
            "build_meta": build_meta,
            "summary": summary,
        }
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump({"build_meta": build_meta, "summary": summary}, sf, ensure_ascii=False, indent=2)

    print("\n========== FINAL SUMMARY ==========")
    print(json.dumps(build_meta, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved sample records: {args.save_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
