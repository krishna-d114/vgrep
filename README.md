# grep for videos

Search a video with text, a face photo, or both, and get back **clips**, not just timestamps.

| Query | Example | How it's answered |
|---|---|---|
| Text | `a red tie` | DFN CLIP embeddings of every frame |
| Image | a photo of a person | face recognition (InsightFace) |
| Image + text | photo + `wearing a black t shirt` | face match first, then an attribute check on that person's body region |

Matching frames are merged into clips, cut with ffmpeg, and can be stitched into one reel.

## How it works

- **Index once (at upload):** ffmpeg decodes the video, DFN2B CLIP ViT-L/14 embeds every frame, InsightFace detects and embeds every face. Vectors are cached.
- **Query many times:** searching only touches the cached vectors, so the video is never reprocessed.
- **Routing:** a reference image with a prominent face goes to the face lane. Anything else goes to image-to-image CLIP search.

## Why these models

Compared four on the same frames (OpenAI CLIP L/14, DFN2B CLIP L/14, SigLIP 2 base, SigLIP 2 so400m). DFN2B gave the best top-10 for "lion" by eye, at about 1.07x the encode time of OpenAI's L/14 (T4, fp16). CLIP-style models match concepts, not identity, which is why faces get a dedicated model.

## Results

Hand-labeled on 3 test videos. Numbers below come from the final notebook run.

| Query type | Precision | Recall | Query time |
|---|---|---|---|
| Text only | _fill in_ | _fill in_ | _fill in_ s |
| Image only (face) | _fill in_ | _fill in_ | _fill in_ s |
| Image + text | _fill in_ | _fill in_ | _fill in_ s |

One-time indexing cost per video: _fill in_ s (T4). Model load: _fill in_ s.

## Run it

Kaggle notebook `video_search_mvp.ipynb`: GPU on, Internet on. Set the paths in the config cell, run all cells. The last cell runs the test queries and prints clips, one-time cost, and query time in seconds.

## Limits

- Small test: 3 videos and a handful of queries.
- Half-visible faces are missed. The attribute check is color-based and is weak on layered clothing and close-ups.
- Match thresholds were tuned by hand.
- InsightFace's pretrained models are **non-commercial research use only**.
- Face search is biometric identification. Use it only on footage you have the right to process.
