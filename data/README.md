# Data

Contents of this folder are gitignored. Regenerate with `python scripts/prepare_data.py`.

- `raw/` — untouched ConvoKit download of `conversations-gone-awry-corpus`.
- `processed/seeds.jsonl` — `ConversationSeed` records only. This is the only data agents may see.
- `processed/labels.jsonl` — `conversation_id`, `real_outcome`, `split`, `pair_id`. Read only by `evaluate.py`.
