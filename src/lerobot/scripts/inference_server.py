import torch
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pathlib import Path
import traceback
import sys

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

app = FastAPI()

policy = None
preprocessor = None
postprocessor = None
device = torch.device("cuda")

class InferenceRequest(BaseModel):
    base_image: List[int]
    wrist_image: List[int]
    proprio: List[float]
    shape_base: List[int]
    shape_wrist: List[int]
    reset: Optional[bool] = False

@app.on_event("startup")
def load_model():
    global policy, preprocessor, postprocessor
    ckpt_path = Path("/home/daniel-kovari/v160/checkpoints/060000/pretrained_model")
    
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt_path
    )
    policy.reset()

@app.post("/infer")
async def infer(req: InferenceRequest):
    try:
        if req.reset:
            policy.reset()

        base_pixels = np.array(req.base_image, dtype=np.uint8)
        base_img = base_pixels.reshape(req.shape_base)
        base_t = torch.from_numpy(base_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        wrist_pixels = np.array(req.wrist_image, dtype=np.uint8)
        wrist_img = wrist_pixels.reshape(req.shape_wrist)
        wrist_t = torch.from_numpy(wrist_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        state_t = torch.tensor(req.proprio, dtype=torch.float32).unsqueeze(0).to(device)

        raw_obs = {
            "observation.state": state_t,
            "observation.images.base": base_t,
            "observation.images.gripper": wrist_t
        }

        obs_processed = preprocessor(raw_obs)

        with torch.inference_mode():
            action_norm = policy.select_action(obs_processed)
            action_phys = postprocessor(action_norm)
            actions_np = action_phys.cpu().numpy()

        return {
            "actions": actions_np.flatten().tolist(),
            "shape": list(actions_np.shape)
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)