# RoBERTa-base tokenizer

`tokenizer.json` contains the original fast tokenizer from Facebook AI's RoBERTa-base release. It is sufficient to reproduce the BPE token IDs offline with `tokenizers==0.22.2`. No encoder weights are included. [Pinned upstream snapshot](https://huggingface.co/FacebookAI/roberta-base/blob/e2da8e2f811d1448a5b465c236feacd80ffbac7b/tokenizer.json)

The copy was retrieved as Git blob `ad0bcbeb288f0d1373d88e0762e66357f55b8311` from a [public mirror](https://github.com/edng5/roBERTa-base). The blob ID matches the upstream tokenizer ETag recorded in [huggingface/transformers #12571](https://github.com/huggingface/transformers/issues/12571). The local file has 1,355,863 bytes and SHA-256 `847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e`. The model's MIT license is included in this directory.

## Project windowing rule

Encode the cleaned speech once with the tokenizer's original case and leading-space behavior. Count content tokens before adding special tokens. For up to 254 tokens, retain the full speech as one window. For 255–508, split at the midpoint, placing any extra token in the first window. For 509 or more, take the first 254 and last 254 tokens. The spans never overlap. Wrap each span in `<s>` and `</s>` so every window has at most 256 encoded positions. Padding can be added by the batch collator. Apply the split assignment by speech ID to every window and average window probability vectors to obtain the speech prediction.

The script `scripts/week1_windows.py` regenerates `data/week1/windows.csv`. The offsets are zero-based, half-open, and relative to the content token IDs for the full cleaned speech. The text windows can be shared by every metadata condition using year and chamber as separate features.
