# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import requests

from cap_prompt import (
    MM_ANALYSIS_PROMPT,
    SCENE_INTEGRATION_PROMPT,
    STRATEGY_PLANNER_PROMPT,
    GENERATOR_PROMPT,
    CRITIC_PROMPT,
)

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
IEMOCAP10_SET = set(IEMOCAP10)

IEMOCAP_LABEL_ALIASES = {
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "hap": "happy",
    "happy": "happy",
    "happiness": "happy",
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

_SENTIMENT_ALIAS = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neutral": "neutral",
}


@dataclass
class LLMConfig:
    base_url: str
    model: str
    timeout: int = 1200


class OllamaGenerateClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.root = cfg.base_url.rstrip("/").replace("/v1", "")

    def _split_generate_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        final = str(data.get("response", "") or "").strip()
        thinking = str(data.get("thinking", "") or "").strip()
        return {
            "final": final,
            "thinking": thinking,
            "raw": data,
            "api_mode": "generate",
        }

    def _split_chat_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        msg = data.get("message") or {}
        final = str(msg.get("content", "") or "").strip()
        thinking = str(msg.get("thinking", "") or "").strip()
        return {
            "final": final,
            "thinking": thinking,
            "raw": data,
            "api_mode": "chat",
        }

    def generate(
        self,
        prompt: str,
        images_b64: Optional[List[str]] = None,
        temperature: float = 0.2,
        num_predict: int = 700,
        force_json: bool = True,
        think: bool = False,
        logprobs: bool = False,
        top_logprobs: int = 0,
        return_full: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        url = f"{self.root}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "logprobs": bool(logprobs),
            },
            "think": bool(think),
        }
        if logprobs and int(top_logprobs) > 0:
            payload["options"]["top_logprobs"] = int(top_logprobs)
        if images_b64:
            payload["images"] = images_b64
        if force_json:
            payload["format"] = "json"

        r = requests.post(url, json=payload, timeout=self.cfg.timeout)
        print("[DEBUG] HTTP", r.status_code, "url=", url)
        if r.status_code != 200:
            print("[DEBUG] body:", r.text[:1500])
        r.raise_for_status()

        data = r.json()
        packet = self._split_generate_output(data)
        if (not packet["final"]) and packet["thinking"]:
            print(f"[WARN] generate final empty; thinking_len={len(packet['thinking'])}")
        return packet if return_full else packet["final"]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        images_b64: Optional[List[str]] = None,
        temperature: float = 0.2,
        num_predict: int = 700,
        think: bool = False,
        logprobs: bool = False,
        top_logprobs: int = 0,
        return_full: bool = False,
        safe_retry: bool = True,
    ) -> Union[str, Dict[str, Any]]:
        url = f"{self.root}/api/chat"

        msg_list = [dict(m) for m in (messages or [])]
        if images_b64 and msg_list and msg_list[-1].get("role") == "user":
            msg_list[-1]["images"] = images_b64

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": msg_list,
            "stream": False,
            "think": bool(think),
            "logprobs": bool(logprobs),
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": 8192,
            },
        }
        if logprobs and int(top_logprobs) > 0:
            payload["top_logprobs"] = int(top_logprobs)

        prompt_chars = sum(len(str(m.get("content", "") or "")) for m in msg_list)
        image_count = len(images_b64 or [])
        print(
            f"[DEBUG chat payload] chars={prompt_chars} | "
            f"images={image_count} | num_predict={num_predict} | logprobs={bool(logprobs)}"
        )

        try:
            r = requests.post(url, json=payload, timeout=self.cfg.timeout)
            print("[DEBUG] HTTP", r.status_code, "url=", url)
            if r.status_code != 200:
                print("[DEBUG] body:", r.text[:1500])
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if safe_retry and status == 500 and (images_b64 or logprobs):
                print("[WARN] /api/chat failed once; retrying in safe mode (drop images, disable logprobs).")
                return self.chat(
                    messages=messages,
                    images_b64=None,
                    temperature=temperature,
                    num_predict=num_predict,
                    think=think,
                    logprobs=False,
                    top_logprobs=0,
                    return_full=return_full,
                    safe_retry=False,
                )
            raise

        data = r.json()
        packet = self._split_chat_output(data)
        if (not packet["final"]) and packet["thinking"]:
            print(f"[WARN] chat final empty; thinking_len={len(packet['thinking'])}")
        return packet if return_full else packet["final"]


# -----------------------------
# Video helpers
# -----------------------------
def _encode_frame_to_jpg_bytes(frame, max_side: int = 640, jpeg_quality: int = 80) -> Optional[bytes]:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim > max_side:
        scale = max_side / float(max_dim)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return buf.tobytes() if ok else None


def sample_video_tail_quarter_keyframes(
    video_path: str,
    num_frames: int = 16,
    tail_ratio: float = 0.25,
) -> List[bytes]:
    if not video_path or (not os.path.exists(video_path)) or num_frames <= 0:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        print(f"[WARN] Empty video: {video_path}")
        return []

    tail_ratio = max(0.0, min(1.0, float(tail_ratio)))
    start_idx = int(frame_count * (1.0 - tail_ratio))
    start_idx = min(max(0, start_idx), max(0, frame_count - 1))
    end_idx = frame_count - 1

    if start_idx >= end_idx:
        indices = [end_idx]
    elif num_frames == 1:
        indices = [int((start_idx + end_idx) / 2)]
    else:
        step = (end_idx - start_idx) / float(num_frames - 1)
        indices = [int(round(start_idx + i * step)) for i in range(num_frames)]
        indices = sorted(set(min(max(0, x), end_idx) for x in indices))

    frames: List[bytes] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        jpg = _encode_frame_to_jpg_bytes(frame)
        if jpg is not None:
            frames.append(jpg)
    cap.release()
    return frames


def to_images_b64(frames_jpeg: List[bytes]) -> List[str]:
    return [base64.b64encode(b).decode("utf-8") for b in (frames_jpeg or [])]


# -----------------------------
# Label helpers
# -----------------------------
def normalize_iemocap10_label(label: Any) -> str:
    s = str(label or "").strip().lower()
    return IEMOCAP_LABEL_ALIASES.get(s, "other")


def normalize_sentiment3_label(label: Any) -> str:
    s = str(label or "").strip().lower()
    return _SENTIMENT_ALIAS.get(s, "neutral")


def _boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x or "").strip().lower() in {"true", "1", "yes", "y"}


def sanitize_scene_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}

    inc = obj.get("inconsistency_check")
    if not isinstance(inc, dict):
        inc = {}
        obj["inconsistency_check"] = inc
    inc["exists"] = _boolish(inc.get("exists", False))
    inc_type = str(inc.get("type", "") or "").strip().lower()
    if inc_type not in {"none", "text_internal", "visual_internal", "cross_modal", "uncertain"}:
        inc_type = "uncertain" if inc["exists"] else "none"
    if not inc["exists"]:
        inc_type = "none"
    inc["type"] = inc_type
    if not isinstance(inc.get("evidence"), list):
        inc["evidence"] = []

    decision = obj.get("decision_trace")
    if not isinstance(decision, dict):
        decision = {}
        obj["decision_trace"] = decision
    override_type = str(decision.get("override_type", "") or "").strip().lower()
    if override_type not in {"history_continue", "latest_override", "humor_shift", "action_shift", "ambiguity_to_neutral"}:
        decision["override_type"] = "ambiguity_to_neutral"

    sent = obj.get("predicted_target_sentiment")
    if not isinstance(sent, dict):
        sent = {}
        obj["predicted_target_sentiment"] = sent
    sent["label"] = normalize_sentiment3_label(sent.get("label"))
    try:
        sent["confidence"] = max(0.0, min(1.0, float(sent.get("confidence", 0.0))))
    except Exception:
        sent["confidence"] = 0.0
    if not isinstance(sent.get("evidence"), list):
        sent["evidence"] = []

    emo = obj.get("predicted_target_emotion")
    if not isinstance(emo, dict):
        emo = {}
        obj["predicted_target_emotion"] = emo
    emo["label"] = normalize_iemocap10_label(emo.get("label"))
    try:
        emo["confidence"] = max(0.0, min(1.0, float(emo.get("confidence", 0.0))))
    except Exception:
        emo["confidence"] = 0.0
    if not isinstance(emo.get("evidence"), list):
        emo["evidence"] = []

    return obj


# -----------------------------
# JSON helpers
# -----------------------------
def parse_json_safely(s: str) -> Dict[str, Any]:
    if not s:
        print("[ERROR] Empty output received.")
        return {}

    raw = s.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"_raw": raw}
    except Exception:
        pass

    m_md = re.search(r"`{3}(?:json)?\s*(.*?)\s*`{3}", raw, re.DOTALL | re.IGNORECASE)
    if m_md:
        cand = m_md.group(1).strip()
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else {"_raw": cand}
        except Exception:
            raw = cand

    m = re.search(r"\{[\s\S]*", raw)
    if m:
        cand = m.group(0).strip()
        if cand.count("[") > cand.count("]"):
            cand += "]" * (cand.count("[") - cand.count("]"))
        if cand.count("{") > cand.count("}"):
            cand += "}" * (cand.count("{") - cand.count("}"))
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else {"_raw": cand}
        except Exception:
            pass

    print("[ERROR] JSON decode failed. raw head:", raw[:700])
    return {"_raw": s}


def _json_text(x: Dict[str, Any]) -> str:
    return json.dumps(x or {}, ensure_ascii=False)


def _pick_primary_text(packet: Dict[str, Any]) -> Dict[str, Any]:
    final = str(packet.get("final", "") or "").strip()
    thinking = str(packet.get("thinking", "") or "").strip()
    if final:
        return {"text": final, "source": "final", "final_len": len(final), "thinking_len": len(thinking)}
    if thinking:
        print("[WARN] final empty, fallback to thinking")
        return {"text": thinking, "source": "thinking_fallback", "final_len": 0, "thinking_len": len(thinking)}
    return {"text": "", "source": "empty", "final_len": 0, "thinking_len": 0}


def _run_json_agent_generate(
    client: OllamaGenerateClient,
    prompt: str,
    images_b64: Optional[List[str]],
    temperature: float,
    num_predict: int,
) -> Dict[str, Any]:
    packet = client.generate(
        prompt=prompt,
        images_b64=images_b64,
        temperature=temperature,
        num_predict=num_predict,
        force_json=True,
        think=False,
        return_full=True,
    )
    picked = _pick_primary_text(packet)
    text = str(picked.get("text", "") or "").strip()
    if not text:
        return {
            "_error": "empty_final_and_thinking",
            "_meta": {
                "api_mode": "generate",
                "response_source": "empty",
                "final_len": picked.get("final_len", 0),
                "thinking_len": picked.get("thinking_len", 0),
            },
        }
    obj = parse_json_safely(text)
    if isinstance(obj, dict):
        meta = obj.get("_meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta.update({
            "api_mode": "generate",
            "response_source": picked.get("source", "unknown"),
            "final_len": picked.get("final_len", 0),
            "thinking_len": picked.get("thinking_len", 0),
        })
        obj["_meta"] = meta
    return obj


def _logprob_item_bytes(item: Dict[str, Any]) -> bytes:
    b = item.get("bytes")
    if isinstance(b, list) and b:
        try:
            return bytes(int(x) & 0xFF for x in b)
        except Exception:
            pass
    return str(item.get("token", "") or "").encode("utf-8", errors="ignore")


def pick_chat_answer_text(packet: Dict[str, Any]) -> Dict[str, Any]:
    final_text = str(packet.get("final", "") or "")
    thinking = str(packet.get("thinking", "") or "")
    if final_text:
        return {"raw_text": final_text, "norm_text": final_text.strip(), "source": "content"}
    if "</think>" in thinking:
        tail = thinking.split("</think>", 1)[1]
        if tail:
            return {"raw_text": tail, "norm_text": tail.strip(), "source": "thinking_suffix"}
    m = re.search(r"(?:final answer|response|assistant)\s*:\s*(.+)$", thinking, flags=re.IGNORECASE | re.DOTALL)
    if m:
        tail = m.group(1)
        return {"raw_text": tail, "norm_text": tail.strip(), "source": "thinking_labeled_tail"}
    return {"raw_text": "", "norm_text": "", "source": "empty"}


def extract_answer_only_logprobs_chat(packet: Dict[str, Any], answer_text_raw: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    raw = packet.get("raw") or {}
    logprobs = raw.get("logprobs") or []
    answer_text_raw = str(answer_text_raw or "")
    if (not answer_text_raw) or (not logprobs):
        return [], "empty_answer_or_logprobs"

    start = 0
    for i, item in enumerate(logprobs):
        if str(item.get("token", "") or "") == "</think>":
            start = i + 1
    tail = logprobs[start:]
    if not tail:
        return [], "empty_tail"

    tail_bytes = b"".join(_logprob_item_bytes(x) for x in tail)
    candidates: List[str] = []
    for cand in [answer_text_raw, answer_text_raw.lstrip(), answer_text_raw.rstrip(), answer_text_raw.strip()]:
        if cand and cand not in candidates:
            candidates.append(cand)

    best = None
    for cand in candidates:
        cand_bytes = cand.encode("utf-8")
        pos = tail_bytes.find(cand_bytes)
        if pos >= 0:
            best = (cand_bytes, pos)
            break
    if best is None:
        return [], "alignment_not_found"

    final_bytes, pos = best
    end = pos + len(final_bytes)
    picked: List[Dict[str, Any]] = []
    cur = 0
    for item in tail:
        bts = _logprob_item_bytes(item)
        nxt = cur + len(bts)
        if nxt <= pos:
            cur = nxt
            continue
        if cur >= end:
            break
        picked.append(item)
        cur = nxt
    return picked, None


# -----------------------------
# Agents
# -----------------------------
def agent_multimodal_analysis(
    client: OllamaGenerateClient,
    context_labeled: str,
    target_speaker: str,
    user_video_path: Optional[str] = None,
    num_frames: int = 16,
    critic_feedback: str = "",
    previous_output: Optional[Dict[str, Any]] = None,
    images_b64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    print("\n========== MM_AGENT ==========")
    if images_b64 is None and user_video_path:
        images_b64 = to_images_b64(sample_video_tail_quarter_keyframes(user_video_path, num_frames=num_frames))

    user_text = (
        MM_ANALYSIS_PROMPT
        + "\n\n### Dialogue history (plain text, no history emotion labels)\n" + (context_labeled or "")
        + "\n\n### Target speaker\n" + str(target_speaker)
    )
    if previous_output:
        user_text += "\n\n### Your previous output\n" + _json_text(previous_output)
    if critic_feedback:
        user_text += "\n\n### Critic feedback for revising Agent1\n" + critic_feedback

    return _run_json_agent_generate(client, user_text, images_b64, temperature=0.2, num_predict=1000)


def agent_scene_integration(
    client: OllamaGenerateClient,
    context_labeled: str,
    mm_json: Dict[str, Any],
    target_speaker: str,
    user_video_path: Optional[str] = None,
    num_frames: int = 16,
    critic_feedback: str = "",
    previous_output: Optional[Dict[str, Any]] = None,
    images_b64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    print("\n========== SCENE_AGENT ==========")
    if images_b64 is None and user_video_path:
        images_b64 = to_images_b64(sample_video_tail_quarter_keyframes(user_video_path, num_frames=num_frames))

    user_text = (
        SCENE_INTEGRATION_PROMPT
        + "\n\n### Target speaker\n" + str(target_speaker)
        + "\n\n### Dialogue history (plain text, no history emotion labels)\n" + (context_labeled or "")
        + "\n\n### Agent1 Multimodal analysis JSON\n" + _json_text(mm_json)
    )
    if previous_output:
        user_text += "\n\n### Your previous output\n" + _json_text(previous_output)
    if critic_feedback:
        user_text += "\n\n### Critic feedback for revising Agent2\n" + critic_feedback

    out = _run_json_agent_generate(client, user_text, images_b64, temperature=0.2, num_predict=1400)
    return sanitize_scene_output(out)


def agent_strategy_planner(
    client: OllamaGenerateClient,
    context_labeled: str,
    mm_json: Dict[str, Any],
    scene_json: Dict[str, Any],
    target_speaker: str,
    critic_feedback: str = "",
    previous_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print("\n========== STRATEGY_AGENT ==========")
    user_text = (
        STRATEGY_PLANNER_PROMPT
        + "\n\n### target_speaker\n" + str(target_speaker)
        + "\n\n### Dialogue history (plain text, no history emotion labels)\n" + (context_labeled or "")
        + "\n\n### Agent1 Multimodal analysis JSON\n" + _json_text(mm_json)
        + "\n\n### Agent2 Scene integration JSON\n" + _json_text(scene_json)
    )
    if previous_output:
        user_text += "\n\n### Your previous output\n" + _json_text(previous_output)
    if critic_feedback:
        user_text += "\n\n### Critic feedback for revising Agent3\n" + critic_feedback

    return _run_json_agent_generate(client, user_text, None, temperature=0.2, num_predict=1000)


def agent_generator(
    client: OllamaGenerateClient,
    context_labeled: str,
    mm_json: Dict[str, Any],
    scene_json: Dict[str, Any],
    strategy_json: Dict[str, Any],
    target_speaker: str,
    user_video_path: Optional[str] = None,
    num_frames: int = 16,
    extra_constraints: Optional[List[str]] = None,
    previous_response: str = "",
    critic_feedback: str = "",
    need_logprobs: bool = False,
    images_b64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    print("\n========== GENERATOR_AGENT ==========")
    if images_b64 is None and user_video_path:
        images_b64 = to_images_b64(sample_video_tail_quarter_keyframes(user_video_path, num_frames=num_frames))

    extra_constraints = extra_constraints or []
    extra_text = "\n".join([f"- {c}" for c in extra_constraints]) if extra_constraints else "(none)"
    user_text = (
        GENERATOR_PROMPT
        + "\n\n### Dialogue history (plain text, no history emotion labels)\n" + (context_labeled or "")
        + "\n\n### Target speaker (must speak next)\n" + str(target_speaker)
        + "\n\n### Agent1 Multimodal analysis JSON\n" + _json_text(mm_json)
        + "\n\n### Agent2 Scene integration JSON\n" + _json_text(scene_json)
        + "\n\n### Agent3 Strategy plan JSON (must be followed)\n" + _json_text(strategy_json)
        + "\n\n### Previous response draft\n" + (previous_response or "")
        + "\n\n### Critic feedback for revising Agent4\n" + (critic_feedback or "")
        + "\n\n### Extra constraints (from Critic)\n" + extra_text
    )
    messages = [{"role": "user", "content": user_text}]

    if need_logprobs:
        packet = client.chat(
            messages=messages,
            images_b64=images_b64,
            temperature=0.55,
            num_predict=550,
            think=False,
            logprobs=True,
            return_full=True,
        )
        picked = pick_chat_answer_text(packet)
        resp_raw = str(picked.get("raw_text", "") or "")
        resp = str(picked.get("norm_text", "") or "")
        thinking = str(packet.get("thinking", "") or "")
        raw = packet.get("raw") or {}
        answer_logprobs, align_err = extract_answer_only_logprobs_chat(packet, resp_raw)
        return {
            "response": resp,
            "_generation_meta": {
                "api_mode": "chat",
                "model": raw.get("model"),
                "created_at": raw.get("created_at"),
                "done": raw.get("done"),
                "done_reason": raw.get("done_reason"),
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
                "logprobs": answer_logprobs,
                "response_source": picked.get("source", "empty"),
                "thinking_len": len(thinking),
                "ppl_alignment_error": align_err,
                "response_raw": resp_raw,
            },
        }

    final_text = client.chat(
        messages=messages,
        images_b64=images_b64,
        temperature=0.55,
        num_predict=550,
        think=False,
        logprobs=False,
        return_full=False,
    )
    return {"response": str(final_text or "").strip()}


def agent_critic(
    client: OllamaGenerateClient,
    context_labeled: str,
    target_speaker: str,
    user_video_path: Optional[str] = None,
    num_frames: int = 16,
    mm_json: Optional[Dict[str, Any]] = None,
    scene_json: Optional[Dict[str, Any]] = None,
    strategy_json: Optional[Dict[str, Any]] = None,
    generator_json: Optional[Dict[str, Any]] = None,
    images_b64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    print("\n========== CRITIC_AGENT ==========")
    mm_json = mm_json or {}
    scene_json = scene_json or {}
    strategy_json = strategy_json or {}
    generator_json = generator_json or {}

    if images_b64 is None and user_video_path:
        images_b64 = to_images_b64(sample_video_tail_quarter_keyframes(user_video_path, num_frames=num_frames))

    user_text = (
        CRITIC_PROMPT
        + "\n\n### target_speaker\n" + str(target_speaker)
        + "\n\n### Dialogue history (plain text, no history emotion labels)\n" + (context_labeled or "")
        + "\n\n### Agent1 Multimodal analysis JSON\n" + _json_text(mm_json)
        + "\n\n### Agent2 Scene integration JSON\n" + _json_text(scene_json)
        + "\n\n### Agent3 Strategy plan JSON\n" + _json_text(strategy_json)
        + "\n\n### Agent4 Candidate response JSON\n" + _json_text(generator_json)
    )
    return _run_json_agent_generate(client, user_text, images_b64, temperature=0.05, num_predict=2200)
