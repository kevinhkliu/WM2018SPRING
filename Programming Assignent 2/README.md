# Unsupervised text classification in Persian using Probabilistic Latent Semantic Analysis

## Task description
Given 20 News Groups and some seed words of each class, and your task is to classify documents according to the given seed words.
## Dependency
`Python3`  `numpy` `pandas` `string` `re`
## Usage
Download pre-trained PLSA model:
```
bash ./compile.sh
```
For training:
```
bash ./compile.sh [-e] [-b] -d [doc.csv file path] -g [group.csv file path] -o [result file path]
```
```
-e 
	If specified, use the dictionary you made to classify the document
-b 
	if specified, run your best version of your program
-d doc.csv
	The doc.csv
-g group.csv
	The group.csv
-o output.csv
	The output path of your classification

```

## Result
Evaluation: Accuracy  
For sklearn version:  private score = 92.24%  
For numpy version:  private score = 57.82%  

## Reference
1. [Kaggle link](https://www.kaggle.com/c/ntucsie-wm2018-topic-modeling)
2. [Probabilistic Latent Semantic Analysis](https://github.com/laserwave/PLSA)
3. [Probabilistic Latent Semantic Analysis](https://github.com/hitalex/PLSA)
