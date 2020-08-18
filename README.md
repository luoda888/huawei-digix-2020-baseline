# huawei-digix-2020-baseline
huawei digix 2020 competition baselines for ctr &amp; recommand

In the CTR part, we coded the ID with count/nunique/target, crossed the category and numerical features, and constructed the embedding feature of word2vec. And it uses the xDeepFM model in the elegant and easy-to-use deepctr library to provide a simple neural network baseline. Affected by the selection of days in the training set and the instability of the neural network, the score will fluctuate between `0.76-0.77`. Perhaps we can use the migration learning method that spans the number of days in this question (refer to the plan of plantsgo in IJCAI 2018)

In the search correlation prediction part, we consider the monotonicity between tags, replace the Rank model with a regression model, filter the original features for variance, and directly put them into the xgboost and catboost models for learning and get the answer. Using regression modeling can improve your baseline to a score of `0.43+`. Perhaps you can consider cross-combination of different features to further improve your score.


Thanks: https://github.com/shenweichen/DeepCTR-Torch & https://github.com/shenweichen/DeepCTR
