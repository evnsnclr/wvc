# Roadmap

wvc 0.1.0 shipped on 2026-11-03. What comes next, in order, with the check that closes each item. Dates are not promised; one person works on this. Issues that change the order are welcome.

1. **A census of the Hub.** Each week, `wvc diff --rank` against the declared base for the most downloaded derivative uploads, with one receipt per pair and a growing table: what relation the card claims, how many tensors changed, the rank structure of the change, how many shards had to be fetched, and what broke. Closes when four weekly reports are out and the first upload that broke wvc has a fix and a test.
2. **GGUF reading.** Diff a quantized upload against its base, with the quantization recorded in the diff and a conformance vector per supported quantization type, so GGUF-only uploads enter the census.
3. **Change classification.** From the diff alone, say what kind of change an upload is: a merged LoRA (rank and layers), a direction removal across many layers, a dense fine-tune, a quantization, or a direct edit, with the evidence. Closes when precision is measured on at least thirty uploads whose method is stated, misses listed.
4. **An interpretability bridge.** Export per-tensor deltas as safetensors, and a notebook that takes a census pair, localises the change, and runs a published probe or sparse autoencoder on the touched layers. Closes with one analysis where the diff localises an edit the literature describes, and one where it does not and says so.
5. **Bisect over a real training run.** A fine-tuning run committed every step and judged by a fixed evaluator, so `wvc bisect --sweep` runs over the whole curve; predictions written before the run, as in the existing receipts. Closes with a receipt that finds an onset or shows again that none exists.
6. **A write-up** of what the census found, with wvc as the instrument.

Done on request rather than pursued: a `wvc-manifest.json` sidecar convention and a loader-side verify path, signing and attestation of manifest roots, and a proposal to Hugging Face for an `edit` relation on model cards.

Always: the open items from the security reviews (a per-checkout byte cap, verification markers in copied stores, typed records), pinned Actions and CodeQL, and fast answers to the first issues.
