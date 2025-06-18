import json
import time

import numpy as np
import numpy.typing as npt
import osam
from loguru import logger


def get_bboxes_from_texts(
    model: str, image: np.ndarray, texts: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    request: osam.types.GenerateRequest = osam.types.GenerateRequest(
        model=model,
        image=image,
        prompt=osam.types.Prompt(
            texts=texts,
            iou_threshold=1.0,
            score_threshold=0.01,
            max_annotations=1000,
        ),
    )
    logger.debug(
        f"Requesting with model={model!r}, image={(image.shape, image.dtype)}, "
        f"prompt={request.prompt!r}"
    )
    t_start: float = time.time()
    response: osam.types.GenerateResponse = osam.apis.generate(request=request)

    num_annotations: int = len(response.annotations)
    logger.debug(
        f"Response: num_annotations={num_annotations}, "
        f"elapsed_time={time.time() - t_start:.3f} [s]"
    )

    boxes: npt.NDArray[np.float32] = np.empty((num_annotations, 4), dtype=np.float32)
    scores: npt.NDArray[np.float32] = np.empty((num_annotations,), dtype=np.float32)
    labels: npt.NDArray[np.float32] = np.empty((num_annotations,), dtype=np.int32)
    for i, annotation in enumerate(response.annotations):
        boxes[i] = [
            annotation.bounding_box.xmin,
            annotation.bounding_box.ymin,
            annotation.bounding_box.xmax,
            annotation.bounding_box.ymax,
        ]
        scores[i] = annotation.score
        labels[i] = texts.index(annotation.text)

    return boxes, scores, labels


def nms_bboxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_classes: int = max(labels) + 1
    scores_of_all_classes: npt.NDArray[np.float32] = np.zeros(
        (len(boxes), num_classes), dtype=np.float32
    )
    for i, (score, label) in enumerate(zip(scores, labels)):
        scores_of_all_classes[i, label] = score
    logger.debug(f"Input: num_boxes={len(boxes)}")
    boxes, scores, labels = osam.apis.non_maximum_suppression(
        boxes=boxes,
        scores=scores_of_all_classes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_num_detections=max_num_detections,
    )
    logger.debug(f"Output: num_boxes={len(boxes)}")
    return boxes, scores, labels


def get_shapes_from_bboxes(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, texts: list[str]
) -> list[dict]:
    shapes: list[dict] = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        text: str = texts[label]
        xmin, ymin, xmax, ymax = box
        shape: dict = {
            "label": text,
            "points": [[xmin, ymin], [xmax, ymax]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": json.dumps(dict(score=score, text=text)),
        }
        shapes.append(shape)
    return shapes


# labelme/_automation/bbox_from_vlm_qwen.py

import time
import ast
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from loguru import logger

'''
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
logger.debug(f"Loading Qwen2.5-VL model from {model_path}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
processor = AutoProcessor.from_pretrained(model_path)
model.eval()
device = torch.device("cpu")
model.to(device)
'''
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
logger.debug(f"Loading Qwen2.5-VL model from {model_path}")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model with appropriate dtype
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained(model_path)

model.to(device)
model.eval()


import json
import logging
import re
from typing import List, Dict, Tuple

def parse_qwen_output(text: str) -> Tuple[List[Dict], str]:
    """
    Attempts to parse Qwen’s JSON output in one of several schemas:
      1) [{ "bbox": [x1,y1,x2,y2], "label": <str>, … }, …]
      2) [{ "bbox_2d": [x1,y1,x2,y2], "label": <str>, … }, …]
      3) [{ "position": { "x": <int>, "y": <int> }, "type": <str>, … }, …]

    If wrapped in markdown fences (```json …```), extracts the JSON before parsing.
    Returns (list_of_shape_dicts, "") on success. Raises ValueError on unrecognized schema.
    """
    # 1) Strip whitespace and extract JSON from code fences if present
    cleaned_text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.+?)```", cleaned_text, re.S)
    if fence_match:
        cleaned_text = fence_match.group(1).strip()

    try:
        data = json.loads(cleaned_text)

        # --- Case A: [{ "bbox": […], "label": … }, …]
        if (
            isinstance(data, list)
            and all(isinstance(item, dict) and "bbox" in item and "label" in item for item in data)
        ):
            return data, ""

        # --- Case B: [{ "bbox_2d": […], "label": … }, …]
        if (
            isinstance(data, list)
            and all(isinstance(item, dict) and "bbox_2d" in item and "label" in item for item in data)
        ):
            converted: List[Dict] = []
            for item in data:
                converted.append({
                    "bbox": item["bbox_2d"],
                    "label": item["label"],
                    "description": item.get("description", "")
                })
            return converted, ""

        # --- Case C: [{ "position": { "x":…, "y":… }, "type": … }, …]
        if (
            isinstance(data, list)
            and all(isinstance(item, dict) and "position" in item and "type" in item for item in data)
        ):
            converted: List[Dict] = []
            for item in data:
                pos = item["position"]
                x = pos.get("x")
                y = pos.get("y")
                bbox = [x, y, x, y]  # zero-area box for a point
                converted.append({
                    "bbox": bbox,
                    "label": item["type"],
                    "description": item.get("description", "")
                })
            return converted, ""
    except Exception as e:
        logging.debug(f"RAW Qwen output:\n{text}")
        raise ValueError(f"Failed to parse Qwen output: {e}\n{text}")
'''
def parse_qwen_output(text: str):
    """
    Parses the output string from Qwen2.5-VL to extract bounding boxes and captions.
    Handles both bounding boxes (shapes) and plain text descriptions.
    Expected format:
    1. For shapes: [{"bbox_2d": [x1, y1, x2, y2], "label": "label name"}]
    2. For descriptions: plain text string.
    """
    lines = text.splitlines()
    
    # Check if the output is in JSON format (bounding boxes)
    if "```json" in text:
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_str = "\n".join(lines[i+1:])
                json_str = json_str.split("```")[0]
                break
        else:
            json_str = text

        try:
            parsed = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse Qwen output: {e}\n{text}")

        # Prepare the results with 'bbox_2d' and 'label'
        results = [
            {
                "label": box.get("label", "object"),
                "bbox": box["bbox_2d"]  # direct access to bbox_2d
            }
            for box in parsed if "bbox_2d" in box
        ]
        return results, ""  # Return the shapes and an empty string for description

    else:
        # If it's not JSON, treat it as a plain text description (custom prompt)
        return [], text  # Return empty list for shapes, and the text as description
'''
def get_vlm_shapes(
    image: np.ndarray,
    prompt: str = "Outline the position of each and output all the coordinates in JSON format.",
) -> tuple[list[dict], str]:  # Ensure we return both shapes and description

    image_pil = Image.fromarray(image)

    # Use built-in AI label prompt for shapes
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image_pil}]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_pil], return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Calculate image dimensions for bounding box scaling
    input_height = inputs['image_grid_thw'][0][1] * 14
    input_width = inputs['image_grid_thw'][0][2] * 14

    # Parsing the bounding box output
    results, _ = parse_qwen_output(output_text)

    shapes = []
    description_text = ""  # No description for this task, only shapes

    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        abs_x1 = int(x1 / input_width * image.shape[1])
        abs_y1 = int(y1 / input_height * image.shape[0])
        abs_x2 = int(x2 / input_width * image.shape[1])
        abs_y2 = int(y2 / input_height * image.shape[0])

        # Ensure coordinates are in the correct order
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        caption = result.get("caption", "")  # Default to empty string if no caption

        # Create the shape dictionary for LabelMe
        shapes.append({
            "label": result["label"],
            "points": [[abs_x1, abs_y1], [abs_x2, abs_y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": caption
        })

    return shapes, description_text


def inference(
    image_path: str,
    prompt: str,
    sys_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 4096,
    return_input: bool = False
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image}]}
    ]
    """
    # Prepare text input for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to('cpu')
    
    # Generate output text from the model
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    """
    # Prepare text input for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Move inputs to the correct device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    # Generate output text from the model
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Post-process generated output
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    

    if return_input:
        return output_text[0], inputs  # Optionally return input for debugging or tracing
    else:
        return output_text[0]  # Just the generated description text
