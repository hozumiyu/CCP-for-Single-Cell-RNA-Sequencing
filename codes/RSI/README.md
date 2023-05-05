# Residue-Similarity (R-S) scores
Compute the RS scores for the data

The rs_score.py computes the R-score, S-score and RS-score for each data. This requires your feature and a set of labels for your features.
These labels can be from the true-label or from your predicted labels. For true-labels, R-,S- and RS-score reveals the geometric property of the data.

You can utilize RS score in 2 ways. If you have a train-test split, you can compute RS scores for the test split, using the training set as the embedding space.
If there is not train-test split, you can compute RS scores by using the entire data.

python rs_score.py has 2 main codes.
rs(
