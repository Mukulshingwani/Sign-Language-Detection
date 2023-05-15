# Sign_Language_Detector-PyTorch
Recognition of hand gestures in 3D space using a single low resolution camera for converting American Sign Language into any spoken language.

## Set Up Instructions

Expected Environments - Python 39

Steps to run on Local (New User):
1. Make a Virtual Environment named env
```
python -m virtualenv -p C:\Python\Python39\python.exe env
```
2. git clone:
```
https://github.com/Gunnika/Sign_Language_
```
3. Activate env: 
```
.\env\Scripts\activate (windows)
```
```
source env/bin/activate (Linux)
```
4. Move into the cloned repo
```
cd .\Sign_Language_Detector-PyTorch
```
5. Import the requirements using Pip install
```
pip install -r requirements.txt
```
6. Run app.py to do training on author’s dataset and run app on localhost 8080 port
```
python app.py -i 0.0.0.0 -o 8080
```
7. Wait for few seconds (First attempt will take time as training done first time, pickle file of model will be stored)
8. After few seconds link for localhost (http://172.31.37.145:8080/) pops on terminal

## Inspiration
This project focuses on developing a machine learning model to accurately detect American Sign Language (ASL) gestures. Preprocessing techniques such as contour image conversion, segmentation, and resizing were employed to enhance the model’s accuracy. Data augmentation techniques like random Gaussian noise and random horizontal flip were also used to improve model performance. The model was implemented in Pytorch, Torchscript, and ONNX environments, demonstrating its versatility and effectiveness. The results of this project suggest that machine learning can be a valuable tool in solving real-world problems related to sign language detection and interpretation, improving the quality of life for individuals in the deaf community.


![percentage](https://user-images.githubusercontent.com/34855465/76789152-42404700-67e2-11ea-8e96-718ba4ae0a36.png)

### American Sign Language (ASL)
American Sign Language (ASL) is a visual language. With signing, the brain processes linguistic information through the eyes. The shape, placement, and movement of the hands, as well as facial expressions and body movements, all play important parts in conveying information. 
![ASL](https://user-images.githubusercontent.com/34855465/76790591-28ecca00-67e5-11ea-990d-b6540acb9a1b.png)


## Dataset used
American Sign Language Train (https://drive.google.com/drive/folders/1-XTAjPPRPFeRqu3848z8dMXaolILWizn?usp=sharing)

American Sign Language Test (https://drive.google.com/drive/folders/18e1F1n1SWPF8lUF8pCKdUzSzKAbmSbVN?usp=share_link)
Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).

## Working
![steps](https://user-images.githubusercontent.com/34855465/76790048-1625c580-67e4-11ea-9fcb-77339e2c4658.png)

Autocompletion and Word Suggestion simplify and accelerate the process of information transmission. The user can select one out of the top 4 suggestions or keep making more gestures until the desired word is obtained. 

## Use Cases
1. Deaf people can have a common classroom by asking their questions/doubts without any hesitation
2. Inclusion of this community in normal schools.
3. Tourist Guides can communicate better using sign language

## Contributors
 - Mukul Shingwani
 - Saurabh Modi
 - Jaimin Sanjay Gajjar

