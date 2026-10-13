---
tier: on-demand
read_when: profiling shows the symptom below
---
# IPluginV3 version of a built-in plugin

- Category: compute
- Applies to: TensorRT 10.11 or later engines that call a built-in plugin, such as multi-scale deformable attention

## Symptom
The engine inspector shows `Reformatting CopyNode` layers and Myelin transposes around each plugin call; the time around the plugin exceeds its kernel.

## Technique
Look for a version 2 creator (`trt.get_plugin_registry().all_creators`, IPluginCreatorV3One) and select it with the ONNX node attribute `plugin_version="2"`: the same kernel behind the IPluginV3 interface, which Myelin keeps inside its graph. Delete ONNX files cached by name, which keep the old version.

## Effect
Engine 8.76 -> 7.37 ms (-14.8%): 956 -> 634 layers, 48 reformats -> 0; outputs within the FP16 rebuild spread.

## Side effects
The plan lost its auxiliary streams (2 -> 0), giving back 0.73 ms of overlap; a plan without them runs about 0.3 ms slower after one with them has run in the same process.

## Does not apply
- Plugins with no V3 creator in the installed TensorRT.

## Evidence
- ✓ 2026-10: DETR-style pose model, 12 deformable attention calls, FP16, RTX 5090, TensorRT 10.16: bench 9.80 -> 8.34 ms.
