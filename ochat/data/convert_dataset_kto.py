"""
Convert OpenAI-format JSONL to OpenChat Conversation format for KTO.

KTO uses unpaired data — each line is a single conversation with a `label`
field (true = desirable, false = undesirable). Same converter as SFT; the
only difference is that KTO data includes `"label"` in the input JSONL.

Usage: python -m ochat.data.convert_dataset_kto \
    --model-type MODEL_TYPE_chatml \
    --model-path BASE_REPO \
    --in-files openai_data.jsonl \
    --out-file openchat_kto_data.jsonl

Input JSONL format:
    {"messages": [...], "label": true}   # desirable
    {"messages": [...], "label": false}  # undesirable

See convert_dataset.py for details.
"""

from ochat.data.convert_dataset import main

if __name__ == "__main__":
    main()
