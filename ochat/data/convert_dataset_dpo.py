"""
Convert DPO ConversationOpenAI pairs to the target Conversation format.

Input: JSONL where each line is {"chosen": <ConversationOpenAI>, "rejected": <ConversationOpenAI>}
Output: JSONL where each line is {"chosen": <Conversation>, "rejected": <Conversation>}

Usage: python -m ochat.data.convert_dataset_dpo \
    --model-type <MODEL_TYPE> --model-path <PATH> \
    --in-files data.jsonl --out-file out.jsonl
"""

import os
import re
import uuid
import shutil
import argparse
import requests
from urllib.parse import urlparse
import concurrent.futures
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from pydantic import ValidationError

from ochat.config import MODEL_CONFIG_MAP
from ochat.config.conversation_template import ConversationOpenAI, Conversation, Tool, ImageContentPart, VideoContentPart, Message


def job_print(job_id: int, *args, **kwargs):
    print(f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] [JOB ID: {job_id}]', *args, **kwargs)

def handle_media(url: str, output_dir: str, media_type: str) -> str:
    """Downloads or copies media from URL/local path and returns the relative path."""
    if url.startswith("data:"):
        return url

    ext = os.path.splitext(urlparse(url).path)[1]
    if not ext:
        ext = ".jpg" if media_type == "image" else ".mp4"

    filename = f"{uuid.uuid5(uuid.NAMESPACE_URL, url).hex}{ext}"
    sub_dir = "images" if media_type == "image" else "videos"
    save_path = os.path.join(output_dir, sub_dir, filename)
    relative_path = os.path.join(sub_dir, filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        return relative_path

    if url.startswith("http://") or url.startswith("https://"):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return url
    else:
        local_path = url[7:] if url.startswith("file://") else url
        if os.path.exists(local_path):
            try:
                shutil.copy2(local_path, save_path)
            except Exception as e:
                print(f"Failed to copy local file {local_path}: {e}")
                return url
        else:
            print(f"Local file not found: {local_path}")
            return url

    return relative_path

def to_hf_tools(tools: Optional[List[Tool]]) -> Optional[List[Dict]]:
    """Convert Pydantic tools to HuggingFace dictionary format."""
    if not tools:
        return None
    return [t.model_dump(exclude_none=True) for t in tools]


def _build_conv_from_prefix(
    hf_messages: list,
    msg_images: list,
    msg_videos: list,
    hf_tools: Optional[List[Dict]],
    target_index: int,
    active_weights: list,
    tokenizer,
) -> Conversation:
    """Round-trip a message prefix through the tokenizer to get a Conversation.

    Applies the chat template (tokenize=True), decodes, and parses the
    ChatML output back into Message objects with correct weights.
    """
    active_prefix = hf_messages[:target_index + 1]

    # Gather media only up to this prefix
    active_imgs = [img for sublist in msg_images[:target_index + 1] for img in sublist]
    active_vids = [vid for sublist in msg_videos[:target_index + 1] for vid in sublist]

    kwargs = {"tokenize": True, "return_dict": True, "add_generation_prompt": False}
    if hf_tools:
        kwargs["tools"] = hf_tools

    tokenized_out = tokenizer.apply_chat_template(active_prefix, **kwargs)

    input_ids = tokenized_out["input_ids"]
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)

    chatml_pattern = r"<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>(?=\s*(?:<\|im_start\|>|$)))"
    matches = re.findall(chatml_pattern, decoded_str, re.DOTALL)

    out_items = []
    system_prompt = ""
    msg_index = 0

    for role, content in matches:
        role = role.strip()
        content = content.strip()

        if role == "system":
            system_prompt = content
            if msg_index < len(active_prefix) and active_prefix[msg_index]["role"] == "system":
                msg_index += 1
        else:
            w = active_weights[msg_index] if msg_index < len(active_weights) else 0.0
            out_items.append(Message(role=role, content=content, weight=w))
            msg_index += 1

    return Conversation(
        items=out_items,
        images=active_imgs if active_imgs else None,
        videos=active_vids if active_vids else None,
        system=system_prompt
    )


def _preprocess_conversation(conv_openai: ConversationOpenAI, out_dir: str):
    """Extract media, build HF messages, and record original weights.

    Returns (hf_messages, msg_images, msg_videos, original_weights, hf_tools).
    """
    hf_tools = to_hf_tools(conv_openai.tools)

    msg_images = []
    msg_videos = []
    hf_messages = []
    original_weights = []

    for msg in conv_openai.messages:
        original_weights.append(msg.weight if msg.weight is not None else 0.0)

        curr_imgs = []
        curr_vids = []
        if msg.content is not None:
            for part in msg.content:
                if isinstance(part, ImageContentPart):
                    rel_path = handle_media(part.image_url.url, out_dir, "image")
                    curr_imgs.append(rel_path)
                elif isinstance(part, VideoContentPart):
                    rel_path = handle_media(part.video, out_dir, "video")
                    curr_vids.append(rel_path)

        msg_images.append(curr_imgs)
        msg_videos.append(curr_vids)
        hf_messages.append(msg.model_dump(exclude_none=True, exclude={"weight", "name"}))

    return hf_messages, msg_images, msg_videos, original_weights, hf_tools


def _compute_strings_and_dedup(
    target_indices: list,
    hf_messages: list,
    original_weights: list,
    hf_tools: Optional[List[Dict]],
    tokenizer,
) -> Tuple[List[int], list, list]:
    """Build template strings for each target prefix and apply substring dedup.

    Returns (kept_indices, prefixes, prefix_weights) where kept_indices are
    the positions in target_indices that survive dedup, and prefix_weights
    assigns non-overlapping weights to each kept prefix.
    """
    # Build strings (tokenize=False) for substring comparison
    prefixes = []
    strings = []
    for t_idx in target_indices:
        prefix = hf_messages[:t_idx + 1]
        prefixes.append(prefix)

        kwargs = {"tokenize": False, "add_generation_prompt": False}
        if hf_tools:
            kwargs["tools"] = hf_tools

        s = tokenizer.apply_chat_template(prefix, **kwargs)
        strings.append(s)

    # Substring dedup
    num_targets = len(target_indices)
    keep_prefix = [True] * num_targets
    prefix_weights = [[] for _ in range(num_targets)]
    weights = original_weights.copy()

    for i in range(num_targets):
        if i == num_targets - 1:
            prefix_weights[i] = weights[:target_indices[i] + 1].copy()
            break
        if strings[i] in strings[i + 1]:
            keep_prefix[i] = False
            continue
        prefix_weights[i] = weights[:target_indices[i] + 1].copy()
        weights[:target_indices[i] + 1] = [0.0] * (target_indices[i] + 1)

    kept_indices = [target_indices[i] for i in range(num_targets) if keep_prefix[i]]
    kept_prefix_weights = [prefix_weights[i] for i in range(num_targets) if keep_prefix[i]]

    return kept_indices, prefixes, kept_prefix_weights


def _validate_and_align_pair(
    chosen_messages: list,
    rejected_messages: list,
) -> Tuple[int, Optional[str]]:
    """Check that non-assistant messages match between chosen and rejected.

    Assistant messages may differ (that's the whole point of DPO), but
    user/system/tool messages must be identical.  Returns:
      (num_messages_to_keep, warning_message_or_None)

    When a mismatch is found, both sides are truncated to the last
    matching turn boundary.
    """
    warnings = []

    # System prompt: compare first message if it's "system"
    c_system = chosen_messages[0] if chosen_messages else None
    r_system = rejected_messages[0] if rejected_messages else None

    c_has_sys = c_system and c_system.get("role") == "system"
    r_has_sys = r_system and r_system.get("role") == "system"

    if c_has_sys != r_has_sys:
        warnings.append("System message mismatch (one side has it, the other doesn't)")
    elif c_has_sys and r_has_sys:
        c_sys_content = _normalize_content(c_system.get("content", ""))
        r_sys_content = _normalize_content(r_system.get("content", ""))
        if c_sys_content != r_sys_content:
            warnings.append("System message content differs between chosen and rejected")

    min_len = min(len(chosen_messages), len(rejected_messages))

    # Walk through messages, stop at first mismatch in user/tool roles
    last_good = 0
    for i in range(min_len):
        c_msg = chosen_messages[i]
        r_msg = rejected_messages[i]
        c_role = c_msg.get("role", "")
        r_role = r_msg.get("role", "")

        if c_role != r_role:
            # Roles differ — truncate to last turn boundary before here
            warnings.append(f"Role mismatch at message {i}: chosen={c_role}, rejected={r_role} — truncating")
            break

        if c_role in ("user", "system", "tool"):
            c_content = _normalize_content(c_msg.get("content", ""))
            r_content = _normalize_content(r_msg.get("content", ""))
            if c_content != r_content:
                warnings.append(f"Content mismatch at message {i} (role={c_role}) — truncating")
                break

        # Track last turn boundary (after each assistant message)
        if c_role == "assistant":
            last_good = i + 1
    else:
        last_good = min_len

    # If one side is longer, the extra messages have no counterpart — truncate
    if len(chosen_messages) != len(rejected_messages):
        if last_good == min_len:
            warnings.append(f"Length mismatch: chosen={len(chosen_messages)}, rejected={len(rejected_messages)} — truncating to {min_len}")
        else:
            warnings.append(f"Length mismatch after alignment: keeping {last_good} messages")

    warning = "; ".join(warnings) if warnings else None
    return last_good, warning


def _normalize_content(content) -> str:
    """Normalize content to a string for comparison."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or "")
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _convert_dpo_pair(
    chosen_openai: ConversationOpenAI,
    rejected_openai: ConversationOpenAI,
    tokenizer,
    out_dir: str,
) -> List[Tuple[Conversation, Conversation]]:
    """Convert a DPO pair, splitting at the union of weighted-turn positions.

    When chosen and rejected have assistant turns at different message
    indices (e.g. chosen at {1,3}, rejected at {1}), the output contains a
    pair for every index in the union ({1,3}).  Substring deduplication is
    applied independently per side; a position is kept when *either* side
    survives its own dedup.

    Non-assistant messages are validated for alignment — if user/system
    messages differ or one side is longer, both are truncated to the last
    matching turn boundary.
    """
    # Validate alignment of non-assistant messages
    align_to, warning = _validate_and_align_pair(
        [msg.model_dump() for msg in chosen_openai.messages],
        [msg.model_dump() for msg in rejected_openai.messages],
    )
    if align_to == 0:
        return []
    if warning:
        print(f"DPO pair alignment: {warning}")

    # Truncate both to aligned length
    chosen_openai.messages = chosen_openai.messages[:align_to]
    rejected_openai.messages = rejected_openai.messages[:align_to]

    # Preprocess both sides
    c_msgs, c_imgs, c_vids, c_weights, c_tools = _preprocess_conversation(chosen_openai, out_dir)
    r_msgs, r_imgs, r_vids, r_weights, r_tools = _preprocess_conversation(rejected_openai, out_dir)

    # Target indices (positions where weight > 0)
    c_targets = [i for i, w in enumerate(c_weights) if w > 0]
    r_targets = [i for i, w in enumerate(r_weights) if w > 0]

    if not c_targets or not r_targets:
        return []

    # Substring dedup — each side independently against its own targets
    c_kept, _, c_w = _compute_strings_and_dedup(c_targets, c_msgs, c_weights, c_tools, tokenizer)
    r_kept, _, r_w = _compute_strings_and_dedup(r_targets, r_msgs, r_weights, r_tools, tokenizer)

    # Union of kept positions
    kept_positions = sorted(set(c_kept) | set(r_kept))

    # Maps: position → dedup weights
    c_w_map = dict(zip(c_kept, c_w))
    r_w_map = dict(zip(r_kept, r_w))

    pairs = []
    for pos in kept_positions:
        # Use dedup weights when this side kept this position.
        # When the position is beyond this side's messages, use the
        # original (full-conversation) weights unmodified — the split
        # belongs entirely to the other side.
        c_active_w = c_w_map.get(pos)
        if c_active_w is None:
            if pos >= len(c_msgs):
                c_active_w = c_weights.copy()
            else:
                c_active_w = _sidekick_weights(c_weights, c_kept, pos)

        r_active_w = r_w_map.get(pos)
        if r_active_w is None:
            if pos >= len(r_msgs):
                r_active_w = r_weights.copy()
            else:
                r_active_w = _sidekick_weights(r_weights, r_kept, pos)

        chosen_conv = _build_conv_from_prefix(c_msgs, c_imgs, c_vids, c_tools, pos, c_active_w, tokenizer)
        rejected_conv = _build_conv_from_prefix(r_msgs, r_imgs, r_vids, r_tools, pos, r_active_w, tokenizer)

        pairs.append((chosen_conv, rejected_conv))

    return pairs


def _sidekick_weights(original_weights: list, kept_positions: list, pos: int) -> list:
    """Build prefix weights for a side at a position the *other* side kept.

    Earlier positions already covered by this side's own preceding kept
    splits are zeroed out (they were consumed by those splits).
    """
    weights = original_weights[:pos + 1].copy()
    preceding = [k for k in kept_positions if k < pos]
    if preceding:
        last_consumed = max(preceding)
        for j in range(min(last_consumed + 1, len(weights))):
            weights[j] = 0.0
    return weights


def process_batch(job_id: int, batch: List[str], args, out_dir: str, pair_label: str = "DPO"):
    model_config = MODEL_CONFIG_MAP.get(args.model_type)

    if model_config:
        tokenizer = model_config.model_tokenizer_create(args.model_path)
    else:
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(args.model_path)

    results = []
    job_print(job_id, f"Processing {len(batch)} {pair_label} pairs...")
    pydantic_context = {"use_json_repair": args.use_json_repair}

    for line in batch:
        line = line.strip()
        if not line:
            continue

        try:
            import orjson
            pair_raw = orjson.loads(line)
        except Exception:
            import json
            pair_raw = json.loads(line)

        if "chosen" not in pair_raw or "rejected" not in pair_raw:
            job_print(job_id, f"Skipping line missing 'chosen' or 'rejected' keys")
            continue

        try:
            chosen_openai = ConversationOpenAI.model_validate(pair_raw["chosen"], context=pydantic_context)
        except ValidationError as e:
            job_print(job_id, f"Skipping pair with invalid 'chosen': {e}")
            continue

        try:
            rejected_openai = ConversationOpenAI.model_validate(pair_raw["rejected"], context=pydantic_context)
        except ValidationError as e:
            job_print(job_id, f"Skipping pair with invalid 'rejected': {e}")
            continue

        pairs = _convert_dpo_pair(chosen_openai, rejected_openai, tokenizer, out_dir)

        if not pairs:
            continue

        import orjson
        for chosen_conv, rejected_conv in pairs:
            pair_out = {
                "chosen": chosen_conv.model_dump(exclude_none=True),
                "rejected": rejected_conv.model_dump(exclude_none=True),
            }
            results.append(orjson.dumps(pair_out).decode("utf-8"))

    job_print(job_id, "Batch finished")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", "--model_type", type=str, required=True)
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--in-files", "--in_files", type=str, nargs="+", required=True)
    parser.add_argument("--out-file", "--out_file", type=str, required=True)
    parser.add_argument("--max-workers", "--max_workers", type=int, default=4)
    parser.add_argument("--max-jobs", "--max_jobs", type=int, default=10)
    parser.add_argument("--use-json-repair", "--use_json_repair", action="store_true",
                        help="Use json_repair for parsing arguments in tools")
    parser.add_argument("--pair-label", "--pair_label", type=str, default="DPO",
                        help="Label for log messages (e.g. DPO, ORPO)")
    args = parser.parse_args()

    if "chatml" not in args.model_type:
        raise ValueError(
            "Currently only models using ChatML format are supported. "
            "Please specify a compatible --model-type."
        )

    out_dir = os.path.dirname(os.path.abspath(args.out_file))

    lines = []
    for f_name in args.in_files:
        with open(f_name, "rt", encoding="utf-8") as f:
            lines.extend(f.readlines())

    def _split(a: list, n: int):
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    batches = list(enumerate(_split(lines, args.max_jobs)))
    all_results = []

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Starting generation using {args.max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        handles = {
            executor.submit(process_batch, job_id=job_id, batch=batch, args=args, out_dir=out_dir, pair_label=args.pair_label): job_id
            for job_id, batch in batches
        }

        for handle in concurrent.futures.as_completed(handles):
            job_id = handles.pop(handle)
            try:
                batch_results = handle.result()
                all_results.extend(batch_results)
                job_print(job_id, "Results collected successfully.")
            except Exception as e:
                job_print(job_id, f"Failed with error: {e}")

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Writing to {args.out_file}...")
    with open(args.out_file, "w", encoding="utf-8") as f:
        for res in all_results:
            f.write(res + "\n")

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Finished processing.")


if __name__ == "__main__":
    main()
