# OpenCV_Vehicle_Detection_Tracking_Matching
Vehicle Detection, Tracking and Matching using OpenCV.

@Python Version: 2.7.12;  
@OpenCV Library Versiton: 2.4.12;  

The Cascade Classifier Training process can be learned from: http://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html  

The outcome of the Cascade Classifier Training is an XML file which contains several number of stages, for each stage it is constructed by several decision stumbs and boosted by Adaboost algorighm.

The detailed process is shown in the following links:  
http://docs.opencv.org/2.4/modules/ml/doc/boosting.html  
http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html  

The detection job is based on the Viola-Jones object detection framework and done through the interface provided by OpenCV.  
The tracking job is based on the Mean-shift tracking algorithm and done through the interface provided by OpenCV.  
The vehicle matching job is based on the SIFT(Scale-Invariant Feature Transform) algorithm for feature extraction & description and using the kNN algorithm to do the matching between the candidates.  
The detailed implementation can be found in each folder, please follow the instructions to run the programs.  
