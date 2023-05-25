# light-VGG11
light-VGG11 is the lightweight version of VGG11, which is used for identifying chicken distress calls and more suitable for practical deployment.

# Requirements
This is my experiment environment:
- python3.7
- pytorch+cuda11.2

# Original Datasets
The dataset named 'Used_AudioAndLabels.pkl'(https://figshare.com/articles/dataset/Automated_identification_of_chicken_distress_vocalisations_using_deep_learning_models/20049722) contains 3,363 distress calls and 1,973 natural barn sounds with one second. Each second contains 22,050 time series points.

# Preprocessed Data
According to the original datasets, we should preprocess them and split them into three parts (training, validation, and test sets) based on fivefold cross-validation technique. The detailed procedures can be found in Section 2.3. Herein, the split five folds can find at the link "https://drive.google.com/drive/folders/1W8yypgILtBFo5307zk8shT3Mnps_m7Tq?usp=sharing". Each fold contains a training set, a validation set, and a test set. We can directly place them into our own local folder and change the path command "parser.add_argument('--data_path',type=str, default='..\\1_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt', help='saved path of input data')" in the train.py . Then, the model can be run normally.

# Paper
The related paper titled "Automated identification of chicken distress vocalisations using deep learning models" is available at "https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0921".
