"""
Convert ConversationOpenAI dataset to the target Conversation format.

Usage: python -m convert_data --in-files data.jsonl --model-type <MODEL_TYPE> --model-path <PATH> --out-file out.jsonl
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
from typing import List, Optional, Dict

from pydantic import ValidationError

from ochat.config import MODEL_CONFIG_MAP, ConversationOpenAI, Conversation
from ochat.config.conversation_template import MessageOpenAI, Tool, ImageContentPart, VideoContentPart, Message


def job_print(job_id: int, *args, **kwargs):
    print(f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] [JOB ID: {job_id}]', *args, **kwargs)

def handle_media(url: str, output_dir: str, media_type: str) -> str:
    """Downloads or copies media from URL/local path and returns the relative path."""
    if url.startswith("data:"):
        return url  # Base64 strings are left as-is

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

    # HTTP / HTTPS
    if url.startswith("http://") or url.startswith("https://"):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return url
    # Local paths / file://
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

def to_hf_message(msg: MessageOpenAI) -> Dict:
    """Converts MessageOpenAI into a dict compatible with HF chat templates using model_dump."""
    # Exclude None values to avoid cluttering the template with empty fields,
    # and exclude 'weight' and 'name' as they aren't part of standard HF messages.
    return msg.model_dump(exclude_none=True, exclude={"weight", "name"})

def to_hf_tools(tools: Optional[List[Tool]]) -> Optional[List[Dict]]:
    """Convert Pydantic tools to HuggingFace dictionary format."""
    if not tools:
        return None
    return [t.model_dump(exclude_none=True) for t in tools]

# ==========================================
# Core Processing Logic
# ==========================================

def process_batch(job_id: int, batch: List[str], args, out_dir: str):

    model_config = MODEL_CONFIG_MAP.get(args.model_type)
    
    if model_config:
        tokenizer = model_config.model_tokenizer_create(args.model_path)
    else:
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(args.model_path)

    results = []
    job_print(job_id, f"Processing {len(batch)} conversations...")
    pydantic_context = {"use_json_repair": args.use_json_repair}

    for line in batch:
        try:
            conv_openai = ConversationOpenAI.model_validate_json(line, context=pydantic_context)
        except ValidationError as e:
            job_print(job_id, f"Skipping invalid conversation: {e}")
            continue
        except Exception as e:
            job_print(job_id, f"Unexpected error during validation: {e}")
            continue
        hf_tools = to_hf_tools(conv_openai.tools)

        msg_images = []
        msg_videos = []
        hf_messages = []
        original_weights = []
        
        # 1. Scope media and build the HF messages iteratively
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
            hf_messages.append(to_hf_message(msg))

        # 2. Find indices where weight > 0
        target_indices = [i for i, w in enumerate(original_weights) if w > 0]
        if not target_indices:
            continue

        # 3. Create prefixes and strings (tokenize=False) INCLUDING TOOLS
        prefixes = []
        strings = []
        for t_idx in target_indices:
            prefix = hf_messages[:t_idx + 1]
            prefixes.append(prefix)
            
            # Apply chat template with tools included
            kwargs = {"tokenize": False}
            if hf_tools:
                kwargs["tools"] = hf_tools
                
            s = tokenizer.apply_chat_template(prefix, **kwargs)
            strings.append(s)

        # 4 & 5. Substring logic
        num_targets = len(target_indices)
        keep_prefix = [True] * num_targets
        prefix_weights = [original_weights[:t_idx + 1].copy() for t_idx in target_indices]

        for i in range(num_targets):
            for j in range(i + 1, num_targets):
                if strings[i] not in strings[j]:
                    prefix_weights[j][target_indices[i]] = 0.0
                else:
                    keep_prefix[i] = False

        # 6. Elongate tokens, decode, and extract
        for i in range(num_targets):
            if not keep_prefix[i]:
                continue
                
            active_prefix = prefixes[i]
            active_weights = prefix_weights[i]
            target_index = target_indices[i]
            
            # Gather media dynamically only up to this prefix
            active_imgs = [img for sublist in msg_images[:target_index + 1] for img in sublist]
            active_vids = [vid for sublist in msg_videos[:target_index + 1] for vid in sublist]
            
            kwargs = {"tokenize": True, "return_dict": True}
            if hf_tools:
                kwargs["tools"] = hf_tools

            tokenized_out = tokenizer.apply_chat_template(active_prefix, **kwargs)
            
            input_ids = tokenized_out["input_ids"]
            if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
                
            decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)

            # 7. Extract assuming ChatML format (Strict Regex implementation)
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
                    # Only increment if the original message block actively provided a system prompt here
                    if msg_index < len(active_prefix) and active_prefix[msg_index]["role"] == "system":
                        msg_index += 1
                else:
                    w = active_weights[msg_index] if msg_index < len(active_weights) else 0.0
                    out_items.append(Message(role=role, content=content, weight=w))
                    msg_index += 1

            # 8. Create Final Object
            final_conv = Conversation(
                items=out_items,
                images=active_imgs if active_imgs else None,
                videos=active_vids if active_vids else None,
                system=system_prompt
            )
            
            results.append(final_conv.model_dump_json(exclude_none=True))

    job_print(job_id, "Batch finished")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--in-files", type=str, nargs="+", required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--use-json-repair", action="store_true", help="Use json_repair for parsing arguments in tools")
    args = parser.parse_args()

    if "chatml" not in args.model_type:
        raise ValueError("Currently only models using ChatML format are supported. Please specify a compatible --model-type.")

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    
    lines = []
    for f_name in args.in_files:
        with open(f_name, "rt", encoding="utf-8") as f:
            lines.extend(f.readlines())

    def _split(a: list, n: int):
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    batches = list(enumerate(_split(lines, args.max_jobs)))
    all_results = []

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Starting generation using {args.max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        handles = {
            executor.submit(process_batch, job_id=job_id, batch=batch, args=args, out_dir=out_dir): job_id
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