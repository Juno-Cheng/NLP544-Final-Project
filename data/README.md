# Data

Contents of this folder are gitignored. Regenerate with `python scripts/prepare_data.py`.

- `raw/`: untouched ConvoKit download of `conversations-gone-awry-corpus`.
- `processed/seeds.jsonl`: the opening comments of each conversation, cut before the first personal attack. This is all the agents may see.
- `processed/labels.jsonl`: `conversation_id`, `real_outcome`, `split`, `pair_id`, `verified`. Only the comparison step reads `real_outcome`.
