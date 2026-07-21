"""
Convert DPO/ORPO ConversationOpenAI pairs to the target Conversation format.

ORPO uses the same paired data format as DPO:
  Input:  {"chosen": <ConversationOpenAI>, "rejected": <ConversationOpenAI>}
  Output: {"chosen": <Conversation>, "rejected": <Conversation>}

This is a thin alias for convert_dataset_dpo — identical behaviour, same
command-line interface.

Usage: python -m ochat.data.convert_dataset_orpo \
    --model-type <MODEL_TYPE>_chatml --model-path <PATH> \
    --in-files data.jsonl --out-file out.jsonl --pair-label ORPO
"""

from ochat.data.convert_dataset_dpo import main


if __name__ == "__main__":
    import sys
    if "--pair-label" not in sys.argv:
        sys.argv.append("--pair-label")
        sys.argv.append("ORPO")
    main()
