# Fine-tuning LLM Model on Medical Dataset
## Project Overview
This report documents the fine-tuning process of a Large Language Model (LLM) for medical image analysis, specifically focusing on radiographic interpretation. The project demonstrates the enhancement of model performance through specialized medical dataset training.

## Initial Model Architecture
- Base Model: unsloth/Llama-3.2-11B-Vision-Instruct
- Implementation: Vision-language model capable of processing and describing medical images
- Initial Configuration: 11B parameters with vision-instruction capabilities

## Libraries and Dependencies
```python
import os
from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
```

## Model Implementation
### Base Model Loading
```python
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
```

### Model Fine-tuning Configuration
```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=False,
    loftq_config=None,
)
```

## Dataset Implementation
```python
dataset = load_dataset("unsloth/Radiology_mini", split="train")
instruction = "You are an expert radiographer. Describe accurately what you see in this image."
```

## Performance Analysis

### Pre-Fine-tuning Output Example
The initial model produced detailed but general anatomical descriptions:

> This image depicts a panoramic radiograph of the human skull. The primary focus of a panoramic radiograph is to display the entire dental arch of each jaw. When examining the radiograph, one should observe the position and shape of all 28 teeth in both the upper and lower arches. In addition to the teeth, one must note the entire skull including the orbits (the eye sockets), zygoma (cheekbone), and mandible (jaws), to ensure that the images of all of the above structures appear unobstructed.
>
> Upon first glance at the image, one will notice the shadows of the maxillary (upper) and mandibular (lower) sinuses. These should be of uniform radiographic density throughout. Next, one must inspect the teeth by viewing the enamel, dentin, and pulp chambers in the teeth. It is of utmost importance that the enamel of each tooth be uniform, and all areas of the tooth free of dark or radiolucent shadows indicating caries. Lastly, one must carefully inspect the periapical (root end) areas of the teeth to determine if any radiographic density indicative of bone or dental problems.

### Post-Fine-tuning Output Example
After fine-tuning on the Radiology_mini dataset, the model produced more precise, clinically relevant observations:

> Panoramic radiography shows multiple small osteolytic lesions in the right mandibular ramus, condyle and ascending ramus, the left mandibular condyle, left first molar and body, right first and second premolar, and first and second molar.

## Key Improvements
1. **Specificity**: Transition from general anatomical descriptions to specific pathological findings
2. **Clinical Focus**: Enhanced ability to identify and describe lesions and abnormalities
3. **Conciseness**: More efficient reporting without sacrificing crucial clinical information
4. **Professional Terminology**: Improved use of specific medical terminology

## Conclusions
The fine-tuning process successfully enhanced the model's capability to provide clinically relevant radiographic interpretations. The transition from general anatomical descriptions to specific pathological findings demonstrates the effectiveness of specialized medical dataset training in improving model performance for clinical applications.
