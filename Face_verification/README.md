# Face Verification Using Deep Learning: A Comparative Study

## Abstract

This report details a comprehensive study conducted to compare the performance of three prominent deep learning models, Facenet, DeepFace, and VGG Face, in the task of face verification. Using the Labeled Faces in the Wild (LFW) dataset, each model's performance was evaluated, and a majority voting ensemble method was employed to consolidate the models' predictions. Contrary to expectations, the ensemble approach did not show a significant improvement over the individual models' performance. This report presents the process, findings, challenges faced, and offers avenues for future work in this area.

## Introduction

Face verification, a vital aspect of biometrics and security systems, aims to verify a person's identity by comparing a presented face with the one on file. This task has been significantly enhanced by deep learning models that can extract intricate features for accurate identification, even under varying conditions. Despite this, achieving high accuracy remains a persistent challenge due to factors such as lighting, facial expressions, and poses. This study aims to evaluate and compare the accuracy of three popular deep learning models – Facenet, DeepFace, and VGG Face – for the face verification task.

## Background

Face verification involves a one-to-one match that verifies the authenticity of a specific user's claimed identity. This task is different from face recognition, which is a one-to-many problem that involves identifying a person from a group of known people.

In recent years, deep learning-based models have emerged as the state-of-the-art in face verification tasks. This study focuses on three such models:

1. **Facenet**: This model, proposed by researchers at Google, uses a deep convolutional network trained to directly optimize the embedding itself, rather than intermediate representations.

2. **DeepFace**: This model, developed by Facebook AI, leverages a nine-layer deep neural network with over 120 million parameters to learn representations directly from raw pixels.

3. **VGG Face**: Developed by the Visual Geometry Group (VGG) at Oxford, this model applies the principles of VGG's work on deep convolutional networks for image recognition to the problem of face verification.

These models were selected based on their widespread use in the field and their performance in benchmark tests.

## Implementation

The study used the LFW dataset, which comprises over 13,000 face photographs designed for studying the problem of unconstrained face recognition. We utilized these image pairs to feed into our models, each returning a binary output indicating whether the images were of the same individual (true) or not (false).

To improve the comparability of the results, the same image pre-processing steps were applied across all models. However, the attempt to improve detection accuracy using an autocropping function yielded no substantial improvements, as each model had embedded autocropping functions.

Following this, each model’s output was consolidated into a Pandas DataFrame, and a majority voting system was applied to compute the final verdict. The consolidated results were then exported as a CSV file for further analysis. 

## Results

The results revealed that all models performed well on the face verification task, but none of them achieved perfect accuracy. This demonstrates the persistent challenges in face verification tasks, particularly when dealing with unconstrained images, as is the case with the LFW dataset.

An interesting finding was that the ensemble method, despite its theoretical potential to improve accuracy, did not provide an accuracy boost in this case. This suggests that the models may have been making similar errors, negating the advantage of combining their predictions.

## Challenges

One major challenge encountered was the inconsistent performance of the autocropping function. While theoretically beneficial, the autocropping function failed to improve detection accuracy. This suggests the complexity of the problem and the difficulty of identifying a one-size-fits-all preprocessing step that works effectively across different

 models.

Another challenge was the computational cost associated with running these models. While necessary for an accurate comparison, the process was time-consuming and required substantial computational resources.

## Conclusion

This study offers a comprehensive comparison of three widely-used deep learning models in face verification tasks. Although each model performed well, none could achieve perfect accuracy, demonstrating the inherent challenges in face verification tasks. The majority voting ensemble method, though theoretically sound, did not offer improvements, indicating the necessity for further exploration of model combinations and ensemble methods.

The results provide insights into the strengths and weaknesses of each model and serve as a stepping-stone for future investigations in the face verification field.

## References

1. Serengil, S. I., & Ozpinar, A. (2020). LightFace: A Hybrid Deep Face Recognition Framework. In 2020 Innovations in Intelligent Systems and Applications Conference (ASYU) (pp. 23-27). IEEE.

2. Serengil, S. I., & Ozpinar, A. (2021). HyperExtended LightFace: A Facial Attribute Analysis Framework. In 2021 International Conference on Engineering and Emerging Technologies (ICEET) (pp. 1-4). IEEE.

3. Serengil, S. I., & Ozpinar, A. (2023). An Evaluation of SQL and NoSQL Databases for Facial Recognition Pipelines. Cambridge Open Engage.

4. Labeled Faces in the Wild (LFW): [link](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/)
