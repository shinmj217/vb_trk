"""SAM3 텍스트 프롬프트로 코트/네트 마스크가 잘 나오는지 확인하는 실험.
python sam3_court_test.py  → data/sam3_test/<이미지>__<프롬프트>.jpg
"""
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

PROMPTS = ["volleyball court", "orange court floor", "volleyball net", "net post"]
OUT = Path("data/sam3_test")
OUT.mkdir(parents=True, exist_ok=True)

model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16).cuda().eval()
proc = Sam3Processor.from_pretrained("facebook/sam3")

for img_path in sorted(Path("data").glob("clip_*_preview.jpg")):
    img = Image.open(img_path).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for prompt in PROMPTS:
        inputs = proc(images=img, text=prompt, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].half()
        t = time.time()
        with torch.no_grad():
            out = model(**inputs)
        torch.cuda.synchronize()
        dt = time.time() - t
        r = proc.post_process_instance_segmentation(
            out, threshold=0.4, mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist())[0]
        vis = bgr.copy()
        for i, (m, s) in enumerate(zip(r["masks"], r["scores"])):
            m = m.cpu().numpy().astype(bool)
            color = np.array([(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)][i % 4])
            vis[m] = (0.5 * vis[m] + 0.5 * color).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color.tolist(), 2)
        scores = [round(float(s), 2) for s in r["scores"]]
        cv2.putText(vis, f"{prompt}  n={len(scores)} {scores}  {dt*1000:.0f}ms", (20, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        name = f"{img_path.stem.replace('_preview', '')}__{prompt.replace(' ', '_')}.jpg"
        cv2.imwrite(str(OUT / name), vis)
        print(f"{img_path.stem:24s} {prompt:20s} n={len(scores)} scores={scores} {dt*1000:.0f}ms")
