# SMS Spam Collection Classifier
## Introduction

This code implements a binary classification model to classify text messages as either "spam" or "not spam". The data used for training and testing the model is the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) dataset from the UCI Machine Learning Repository. The model uses the DistilBERT transformer from the Huggingface [transformers](https://github.com/huggingface/transformers) library for tokenization and classification.

## Importance

The importance of SMS Spam Classification lies in the fact that it helps individuals and organizations to filter out unwanted messages, reducing the amount of time and effort spent sifting through spam messages. By automatically identifying and blocking spam messages, users can save time and protect themselves from potential security threats such as phishing or malware.

SMS Spam Classification is used in a variety of settings, including personal and business email, telecommunication companies, and social media platforms. The process is also useful for individuals and organizations that rely on SMS messaging for marketing, as it helps to ensure that their messages reach their intended audience, rather than being filtered out as spam.

## Dependencies

The following packages are required to run the code:

- pandas
- numpy
- matplotlib
- scikit-learn
- transformers
- torch

## Data Preprocessing

* The dataset is loaded into a pandas dataframe and the columns are named "label" and "text".
* The text data is tokenized using the DistilBERT tokenizer and the labels are transformed into numerical values using the LabelEncoder class from scikit-learn.
* The tokenized text data is padded to ensure that all texts have the same length.
* The data is split into training and testing sets using the train_test_split function from scikit-learn.
* The training and testing data is converted into tensors using the torch library.

## Model

* The DistilBERTForSequenceClassification class from the transformers library is used to initialize the model.
* The model is trained using the Adam optimizer and the CrossEntropyLoss function.
* The model is trained for 10 epochs with a learning rate of 1e-5.
* The training loss is plotted over iterations to visualize the convergence of the model.

## Evaluation

The model is evaluated on the test set by computing the F1 score, which is a measure of the balance between precision and recall. The F1 score is calculated using the f1_score function from the scikit-learn library. A confusion matrix is also created to visualize the performance of the model. The model achieves a F1 score of 97.5.

## Usage

The code can be run on either a CPU or a GPU, depending on the availability. The device is determined at runtime and the model is transferred to the specified device. The code assumes that the SMS Spam Collection dataset is present in the same directory and is named "SMSSpamCollection". To run the code, simply run the script.

## Conclusion

This code demonstrates how to perform text classification using the DistilBERT model in PyTorch. By using the DistilBERT model, the code can achieve high accuracy (99%) on the SMS Spam Collection dataset with relatively little training data.