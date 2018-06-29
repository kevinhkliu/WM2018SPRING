# Information retrieval system

## Task description
Implement a small information retrieval systems using Vector Space Model and Rocchio Relevance Feedback.

## Data
CIBR010 dataset.

## Dependency
`Python3` `numpy` `pandas` `jieba` `sklearn` `collections`
## Usage
Download jieba dict and stopword list:
```
bash ./compile.sh
```
For training:
```
bash ./compile.sh [-R] [-b] -i [query-file] -o [ranked-list] -m [model-dir] -d NTCIR-dir
-r 
	If specified, turn on the relevance feedback on you program.
-b 
	if specified, run your best version of your program.
-i query-file
	The input query file.
-o ranked-list
	The output ranked list file.
-m model-dir
	The input model dictionary, which includes three files:
		model-dir/vocab.all
		model-dir/file-list
		model-for/inverted-file
-d NTCIR-dir
	The dictionary of NTCIR documents, which is the path name of CIB010 dictionary.
	Ex. If the dictionary's pathname is /tmp2/CIB010, it will be "-d /tmp2/CIB010".
```

## Result
Mean Average Precision  
private score = 0.74578  

## Reference
1. [Kaggle link](https://www.kaggle.com/c/ntucsie-wm2018-vsm)
