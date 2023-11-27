config = {}

#config for teeth dector
#min of V (HSV) to detect teeth in image
config["Brightness_threshold"] = 135

#min normalize brightness in mouth to define Nice teeth
config["niceteeth_threshold"] = 0.075

#min distance normalize to check mouth openning
config["normalize_mouth_threshold"]  = 0.2
#
config["FACEMESH_lips"] = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,78, 191, 80,81, 82, 13, 312, 311, 310, 415, 308]
config["FACEMESH_EYEs"] = [243, 463]

config["Ratio Nice Teeth frame"] = 0.1

config["Thresold_Distance"] = 0.3

#config for crop video
#Image shape
config["image_shape"] = (256, 256)
#The minimal allowed iou with inital bbox
config["iou_with_initial"] = 0.25
#Increase bbox by this amount
config["increase"] = 0.2
#Minimum number of frames
config["min_frames"] = 90
