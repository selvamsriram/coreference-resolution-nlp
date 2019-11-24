# Results of various trials

## Closest-Link
*Comparing only with the last mention*
```
Micro Average Precision: 0.32172 (120/373)
Micro Average Recall:    0.17070 (120/703)
F-score:                 0.22305
```

## Best-Link
*Comparing only with all the  mention previously in the cluster*
```
Micro Average Precision: 0.26236 (138/526)
Micro Average Recall:    0.19630 (138/703)
F-score:                 0.22457
```

## All Negative Samples Included
*Logistic Regression*
```
Micro Average Precision: 0.71223 (99/139)
Micro Average Recall:    0.14083 (99/703)
F-score:                 0.23515
```

*SVM*
```
Micro Average Precision: 0.71642 (96/134)
Micro Average Recall:    0.13656 (96/703)
F-score:                 0.22939
```

## 1/3 Negative Samples Included
*Logistic Regression*
```
Micro Average Precision: 0.68790 (108/157)
Micro Average Recall:    0.15363 (108/703)
F-score:                 0.25116
```

*SVM*
```
Micro Average Precision: 0.71223 (99/139)
Micro Average Recall:    0.14083 (99/703)
F-score:                 0.23515
```

*SVM + Manual*
```
Micro Average Precision: 0.32079 (179/558)
Micro Average Recall:    0.25462 (179/703)
F-score:                 0.28390
```

*Logistic Regression - Manual match with only the gold antecedent*
```
Micro Average Precision: 0.36667 (176/480)
Micro Average Recall:    0.25036 (176/703)
F-score:                 0.29755
```

*Logistic Regression*
*Manual Match - Only Gold Ante Comparison*
*Sent Dist divides result*
```
Micro Average Precision: 0.37605 (179/476)
Micro Average Recall:    0.25462 (179/703)
F-score:                 0.30365
```

*Logistic Regression*
*All above + selecting only the substring in anaphor that matches*
```
Micro Average Precision: 0.53678 (197/367)
Micro Average Recall:    0.28023 (197/703)
F-score:                 0.36822
```

