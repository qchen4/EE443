# Face Verification Using Deep Learning: A Comparative Study

## Abstract

This report documents a comprehensive study comparing the performance of three prevalent deep learning models, Facenet, DeepFace, and VGG Face, in the task of face verification. Using the Labeled Faces in the Wild (LFW) dataset, the performance of each model was evaluated, and a majority voting ensemble method was used to consolidate the models' predictions. The ensemble approach did not demonstrate a significant improvement over individual model performance, contrary to initial expectations. This report discusses the process, findings, challenges, and future work.

## Introduction

Face verification is a critical aspect of biometrics and security systems, verifying a person's identity by comparing a presented face with the one on file. Deep learning models, capable of extracting intricate features for accurate identification, significantly enhanced this task. Nevertheless, achieving high accuracy remains a challenge due to factors such as lighting, facial expressions, and poses. This study's objective was to evaluate and compare the accuracy of three popular deep learning models – Facenet, DeepFace, and VGG Face – for the face verification task.

## Background

Face verification involves a one-to-one match that verifies the authenticity of a specific user's claimed identity. This task is distinct from face recognition, which is a one-to-many problem identifying a person from a group of known individuals.

Deep learning-based models have emerged as state-of-the-art in face verification tasks in recent years. This study focuses on three such models:

1. **Facenet**: Proposed by Google researchers, Facenet uses a deep convolutional network trained to directly optimize the embedding itself, rather than intermediate representations.
   
2. **DeepFace**: Developed by Facebook AI, DeepFace leverages a nine-layer deep neural network with over 120 million parameters to learn representations directly from raw pixels.
   
3. **VGG Face**: Developed by the Visual Geometry Group (VGG) at Oxford, VGG Face applies the principles of VGG's work on deep convolutional networks for image recognition to the problem of face verification.

These models were selected due to their widespread use in the field and their performance in benchmark tests.

## Implementation

The study used the LFW dataset, which consists of over 13,000 face photographs designed for studying the problem of unconstrained face recognition. These image pairs were inputted into our models, each of which returned a binary output indicating whether the images were of the same individual (true) or not (false).

To improve the comparability of results, the same image pre-processing steps were applied across all models. However, the attempt to enhance detection accuracy using an autocropping function did not yield significant improvements, as each model had embedded autocropping functions.

For the implementation, a series of Python scripts were created. The first script was used to predict labels for each model and save these predictions in individual pandas DataFrames. Each DataFrame consisted of two columns: the image pair ID and the corresponding prediction.

```python
def generate_predictions(model_name, dataset_name):
    dataset_path = f"./{dataset_name}"
    predictions = []

    for id in range(400):  
        img1_path = f"{dataset_path}/{dataset_name}_{id:03}_0.jpg"
        img2_path = f"{dataset_path}/{dataset_name}_{id:03}_1.jpg"

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            result = DeepFace.verify(img1_path, img2_path, model_name=model_name, enforce_detection=False)
            prediction = result['verified']
            predictions.append(int(prediction))
        else:
            print(f"Images {img1_path} or {img2_path} does not exist.")

    return predictions
```

The second script combined these predictions into a single DataFrame and calculated the majority

 vote for each image pair. This DataFrame was then saved as a CSV file for further analysis.

```python
# Combine the model labels into a single DataFrame
df = pd.concat([Facenet_labels_df, DeepFace_labels_df, VGG_labels_df], axis=1)

# Calculate the majority vote for the final label
df['majority_vote'] = df.mode(axis=1)[0]

# Save the DataFrame to a CSV file in the current directory
current_directory = os.getcwd()
output_file_path = os.path.join(current_directory, 'model_predictions.csv')
df.to_csv(output_file_path, index=False)
```

## Results
The FaceNet, DeepFace, and VGG-Face models were compared based on the calculated metrics. 


### Model Performance

| Model           | LFW Dataset Accuracy | Validation Dataset Accuracy |
|-----------------|----------------------|-----------------------------|
| FaceNet         | 0.801                | 0.89                        |
| DeepFace        | 0.602                | 0.69                        |
| VGG-Face        | 0.84                 | 0.84                        |
| Combined Models | 0.81                 | 0.90                        |

In this table, we present the accuracy scores achieved by different face recognition models on two datasets: LFW dataset and a validation dataset. The models evaluated are FaceNet, DeepFace, VGG-Face, and a combination of multiple models. The accuracy scores are provided for both datasets separately.


When evaluating the performance of our model, we utilize several key evaluation metrics. Accuracy serves as an intuitive measure, representing the ratio of correctly predicted observations to the total number of observations. While high accuracy is desirable, it is most effective when dealing with symmetric datasets where false positives and false negatives have similar values.

Precision, on the other hand, focuses on the precision of our predictions. It quantifies the model's ability to correctly identify positive observations, even if it captures only a few. High precision indicates that the model minimizes false positives, showcasing its capability to avoid labeling negative samples as positive.

Recall, also known as sensitivity, represents the ratio of correctly predicted positive observations to all observations within the actual positive class. A high recall indicates that the model effectively captures a large number of positive instances, but it may also generate more false positives.

To strike a balance between precision and recall, we employ the F1 Score. This metric calculates the weighted average or harmonic mean of precision and recall. By considering both false positives and false negatives, the F1 Score provides a comprehensive assessment of the model's performance. However, it may not be the optimal metric when dealing with imbalanced class distributions, as it assigns equal weight to false positives and false negatives. In such cases, the F1 Score is more applicable and informative in multi-class scenarios.

Comparing the results, we find that the DeepFace model is under-performing and the reason can be attributed to our failure to tune the threshold. Without this crucial step, our model's ability to accurately recognize and classify faces is compromised. By neglecting to optimize the threshold parameter, we hinder the model's discriminative capacity, leading to reduced accuracy and reliability in face verification tasks. The threshold serves as a decision boundary, determining the similarity threshold required to accept or reject face matches. Without proper tuning, we fail to strike a balance between false acceptance and false rejection rates, resulting in compromised performance. 

In future work, to address this issue, we need to undertake a comprehensive threshold-tuning process. This involves evaluating various threshold values and assessing performance metrics such as precision, recall, F1 score, and accuracy. Through iterative adjustments, we can identify an optimal threshold value that maximizes the model's discriminative power and enhances its ability to accurately recognize and classify faces. By incorporating this crucial threshold tuning step, we can improve the DeepFace model's performance and strengthen the overall efficacy of our face verification system.

 

## Challenges

One of the main challenges was the variability in the LFW dataset. As the images were not taken under controlled conditions, factors such as different lighting conditions, facial expressions, and poses made it difficult to achieve a high verification accuracy across all models.

Another challenge was the computational cost associated with running these models. While necessary for an accurate comparison, the process was time-consuming and required substantial computational resources.

## Conclusion

This study offers a comprehensive comparison of three widely-used deep learning models in face verification tasks. Although each model performed well, none could achieve perfect accuracy, demonstrating the inherent challenges in face verification tasks. The majority voting ensemble method, though theoretically sound, did not offer improvements, indicating the necessity for further exploration of model combinations and ensemble methods.

The results provide insights into the strengths and weaknesses of each model and serve as a stepping-stone for future investigations in the face verification field.

## References

1. Serengil, S. I., & Ozpinar, A. (2020). LightFace: A Hybrid Deep Face Recognition Framework. In 2020 Innovations in Intelligent Systems and Applications Conference (ASYU) (pp. 23-27). IEEE.

2. Serengil, S. I., & Ozpinar, A. (2021). HyperExtended LightFace: A Facial Attribute Analysis Framework. In 2021 International Conference on Engineering and Emerging Technologies (ICEET) (pp. 1-4). IEEE.

3. Serengil, S. I., & Ozpinar, A. (2023). An Evaluation of SQL and NoSQL Databases for Facial Recognition Pipelines. Cambridge Open Engage.

4. Labeled Faces in the Wild (LFW): [link](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/)
