# Lab Report

Title: Face Recognition using Different Models

## Introduction:

The objective of this lab is to evaluate the performance of different deep learning models, namely Facenet, Deepface, and VGG-Face, in performing facial recognition tasks. The models were evaluated on the Labeled Faces in the Wild (LFW) dataset, a validation dataset, and a test dataset. Performance was assessed based on accuracy, precision, recall, and the F1 score.

## Methods:

The experiment started with preparing the datasets. For this, three different datasets were used, namely the LFW dataset, a validation dataset, and a testing dataset. The LFW dataset was prepared and saved to Google Drive. Similarly, the validation and testing datasets were also prepared and randomly visualized.

Following the preparation of datasets, different models were tested on a random image pair. These models include DeepFace, Facenet, and VGG-Face. The verification was done using cosine distance.

After testing the models, a function was defined to evaluate the model on a given dataset. This function takes as input the model name and the dataset name. It calculates various metrics including accuracy, precision, recall, and F1 score. These metrics were calculated for each model on the validation dataset.

Lastly, the predictions were generated for each model on the testing dataset, and a majority vote was taken to finalize the prediction. These predictions were then saved to a CSV file.

## Results:

The Facenet, DeepFace, and VGG-Face models were compared based on the calculated metrics. These metrics included accuracy, precision, recall, and the F1 score. The results were plotted for a clearer comparison between the models.



## Discussion:

The lab provided an insightful comparison between different deep learning models used for face recognition. It also demonstrated the process of preparing datasets, testing models, and evaluating their performance. The metrics calculated served as a valuable measure to understand the strengths and weaknesses of each model.

Accuracy is a crucial measure to identify the overall effectiveness of a model. However, in scenarios where the datasets are not symmetric and the false positives and negatives are of significant concern, other measures such as precision, recall, and F1 score provide a more comprehensive evaluation. 

Sure, here is how the section could be added to the report:

---

**Challenges with Auto-cropping in Image Preprocessing**

In our endeavor to enhance the image detection performance of our application, we experimented with the integration of an auto-cropping mechanism in the image preprocessing pipeline. Auto-cropping, an image processing technique, is utilized to trim unnecessary outer portions from a picture. The hypothesis driving this experiment was that by focusing only on the significant parts of an image and negating the background noise, we could facilitate our object detection model to recognize the target objects more effectively.

Regrettably, the application of auto-cropping did not lead to the desired enhancement in our object detection performance. We propose the following potential explanations for this outcome:

1. **Unreliable Cropping Mechanism:** The auto-cropping algorithm we implemented may have been imprecise in determining the boundaries of the object of interest. If these boundaries are inaccurately identified, the resultant cropped image might omit vital portions of the object, creating a more challenging scenario for our object detection model.

2. **Varied Object Locations:** Our image dataset consists of objects that are not necessarily located centrally or uniformly. If the auto-cropping algorithm cannot accurately detect and focus on these objects, it might result in cropped images where the object is partially or even entirely excluded.

3. **Loss of Contextual Information:** The auto-cropping process could be inadvertently removing vital contextual information. Object detection models can frequently utilize background elements and environmental hints to more effectively identify objects. By eradicating these elements, our model might be handicapped.

4. **Size and Aspect Ratio Distortion:** Auto-cropping can lead to distortions in the size and aspect ratio of objects in an image. This is particularly true for objects of varying shapes and sizes, which can confuse the object detection model, especially if it has been trained on images maintaining a standard aspect ratio.

Our future endeavors will be focused on improving the precision of our auto-cropping algorithm and exploring alternative image preprocessing methods that maintain a balance between focusing on the object and preserving contextual information. Additionally, we will be working on enhancing the robustness of our object detection model to account for variations in object location, size, and aspect ratio.

---


## Conclusion:

In conclusion, this lab offered a practical approach to face recognition using deep learning models. The process involved data preparation, model testing, performance evaluation, and output generation. The comparative analysis of different models provides a valuable insight into the model selection based on their performance. 

Further work could involve optimizing these models and experimenting with additional models to enhance the face recognition task's performance. The use of different datasets could also provide further validation of the models' effectiveness. 

In addition, the majority voting system used for finalizing the predictions could be analyzed further. Other decision-making approaches could be tested to observe their impact on the overall performance of the face recognition task.

## References:

- Labeled Faces in the Wild (LFW) dataset
- Facenet, Deepface, and VGG-Face models
- sklearn.metrics for calculating performance metrics
- DeepFace library for face recognition
- TensorFlow Datasets
- Python libraries: Numpy, Pandas, matplotlib, os, and PIL

## Appendices:

The complete Python code used in this lab is included in the appendix. The code includes the necessary comments to understand its functionality.

