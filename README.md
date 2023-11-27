
pipeline: 

Video -> frame -> detectface (mobilenet retina)
If not detected:
    frame -> detectface (resnet retina)
If detected:
  crop image and expand size -> mediapipe to extract keypoint
#Run 
$ conda create -n env python==3.7
$ conda activate env 
$ pip install -r requirements.txt
$ python pipeline_mobile_resnet.py --path_video ["Your path video"] --output_path ["Path to your output"]# DeepFake_b
# DeepFake_b
