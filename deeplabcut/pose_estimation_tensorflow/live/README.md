# Live DeepLabCut

Use DeepLabCut on live video feed to provide feedback.

This package contains a DLCLive class which enables pose estimation online to provide feedback. This object loads and prepares a DLC network for inference, and will return the predicted pose for single images.

To perform processing on poses (such as predicting the future pose of an animal given it's current pose, or to trigger external hardware like send TTL pulses to a laser for optogenetic stimulation), this object takes in a `Processor` object. Processor objects must contain two methods: process and save.
- The `process` method takes in a pose, performs some processing, and returns processed pose.
- The `save` method saves any valuable data created by or used by the processor
For examples, please see the [processor directory](processor)

###### Note :: this object does not record video or capture images from a camera. This must be done separately.

DLCLive parameters:
  - config = string; full path to the DLC project's 'config.yaml' file
  - cropping = list of int, optional; cropping parameters in pixel number: [x1, x2, y1, y2]
  - iteration = int, optional; which iteration of network to use (overrides default in config.yaml)
  - shuffle = int, optional; which shuffle of network to use (overrides default in config.yaml)
  - gputouse = int, optional; specify gpu
  - useFrozen = bool, optional; use frozen tensorflow model
  - useTFGPUInference = bool, optional; Perform inference on GPU with Tensorflow code
  - processor = dlc pose processor object, optional

#### Instructions for use:

1. Initialize processor (if desired)
2. Initialize the DLCLive object
3. Perform pose estimation

```
from deeplabcut import DLCLive, Processor
dlc_proc = Processor()
dlc_live = DLCLive(<path to config file>, processor=dlc_proc)
dlc_live.get_pose(<your image>)
```
