import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import cv2
from transformers import Owlv2Processor

# Load the processor and ONNX session
model_path = "./owlv2-onnx"
processor = Owlv2Processor.from_pretrained(model_path)
onnx_session = ort.InferenceSession(f"{model_path}/model.onnx")

# Load the test image
image_path = "test.jpg"
image = Image.open(image_path)
# Also load with OpenCV for visualization
cv_image = cv2.imread(image_path)
cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

# Define text labels
text_labels = [["a photo of a cat", "a photo of a dog", "a photo of a person"]]

# Process inputs
inputs = processor(text=text_labels, images=image, return_tensors="pt")

# Prepare inputs for ONNX
onnx_inputs = {}
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        onnx_inputs[key] = value.numpy()
    else:
        onnx_inputs[key] = value

# Get input names from ONNX model
input_names = [inp.name for inp in onnx_session.get_inputs()]
print("ONNX input names:", input_names)

# Run ONNX inference
onnx_outputs = onnx_session.run(None, {name: onnx_inputs[name] for name in input_names if name in onnx_inputs})
print("ONNX output shapes:", [out.shape for out in onnx_outputs])

# Convert outputs back to torch format for post-processing
output_names = [out.name for out in onnx_session.get_outputs()]
print("ONNX output names:", output_names)

# Create outputs object similar to model output
from types import SimpleNamespace
outputs = SimpleNamespace()
for i, name in enumerate(output_names):
    setattr(outputs, name, torch.from_numpy(onnx_outputs[i]))

# Process results
target_sizes = torch.tensor([[image.height, image.width]])
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1
)

# Print results
result = results[0]
print(f"Image: {image_path}")
print(f"Image size: {image.width}x{image.height}")
print(f"Result keys: {result.keys()}")

if "boxes" in result:
    boxes, scores = result["boxes"], result["scores"]
    labels = result.get("labels", [i for i in range(len(boxes))])
    
    # Create a copy for visualization
    vis_image = cv_image.copy()
    
    print("\nDetections:")
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue for different classes
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        box = [round(i, 2) for i in box.tolist()]
        label_text = text_labels[0][labels[i]] if labels[i] < len(text_labels[0]) else f"class_{labels[i]}"
        print(f"Detected '{label_text}' with confidence {round(score.item(), 3)} at location {box}")
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        color = colors[labels[i] % len(colors)]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{label_text.replace('a photo of ', '')}: {score.item():.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the visualization
    output_path = "test_detected.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"\nVisualization saved as: {output_path}")
    
    # Optionally display the image (requires GUI environment)
    try:
        cv2.imshow('OWLv2 Detection Results', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Image displayed successfully. Press any key to close.")
    except:
        print("Cannot display image (no GUI environment). Check the saved file instead.")
        
else:
    print("No detections found or unexpected result format")
    print("Result structure:", result)