# Automatic Categorization of Documents

### CONTENT

This is a simple program written in matlab language using conv1D to classify text in terms of 7 categories:
- Space
- Medicine
- Motorcycles
- Christian religion
- Firearms
- Baseball

The program achieves an accuracy of about 87% on the validation set.

![image](https://user-images.githubusercontent.com/83314524/224814682-bb81528b-93e2-4ab2-867b-5972c8836edb.png)

It allows you to perform single predictions in the console of the MATLAB environment, using the command :

*new_doc = tokenizedDocument(new_doc);new_doc = lower(new_doc);new_doc = erasePunctuation(new_doc);X_new = doc2sequence(enc,new_doc,'Length',sequence_length);new_labels = classify(CNN,X_new)*

### Neural network architecture:

 - sequenceInputLayer()
 - wordEmbeddingLayer(100, 8874)
 - convolution1dLayer(2, 200, Padding="causal")
 - batchNormalizationLayer
 - reluLayer
 - dropoutLayer(0.2)
 - globalAveragePooling1dLayer
 - fullyConnectedLayer(6)
 - softmaxLayer
 - classificationLayer
