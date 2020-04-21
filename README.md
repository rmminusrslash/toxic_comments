# Kaggle Challenge Toxic Comments 
In the kaggle challenge "toxic comments" a simple baseline based on naive bayes word features achieves a performance 
within 1.1% of the winning models that required recurrent neural networks (RNNs), ensembles and data augmentation. 

|Model|Kaggle Private Leaderboard AUC|
|---|---|
|Challenge Winner based on RNNs, stacking|98.9
|Transfer learning with Bert|98.4
|Logistic Regression on Naive Bayes Features from TF-IDF (baseline)|97.8



## Baseline Ablation Experiments
I studied the baseline to understand it better.


|Experiment|AUC ValidationSet|Kaggle Private Leaderboard AUC|
|---|---|---|
|Logistic Regression on Naive Bayes Features from TF-IDF (class probabilities if-idf, word-doc-matrix tf-idf)|98.3|97.8
|Logistic Regression on Naive Bayes Features (class probabilities binary, word-document-matrix TF-IDF)|98.1|97.8
|Logistic Regression on Naive Bayes Features (class probabilities binary, word-document-matrix binary)|96.9|97.2

Switching from binary word encoding to tf-idf encoding improved +0.6% (line 3 vs line 1). 

I tried to understand if the uplift comes from the better estimation of the 
conditional class probabilities or from an improved representation of the document. Looking at Experiment two we see see that the improvement comes from the
better document representation. That makes sense since tf-idf weights up words that are more frequent in a document and are more specific to that document. This serves as a simple way for a word to have differrent encodings (numeric values) depending on the context where it is observed.
