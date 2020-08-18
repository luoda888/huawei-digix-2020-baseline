# huawei-digix-2020-baseline
huawei digix 2020 competition baselines for ctr &amp; recommand

In the CTR part, we coded the ID with count/nunique/target, crossed the category and numerical features, and constructed the embedding feature of word2vec. And it uses the xDeepFM model in the elegant and easy-to-use deepctr library to provide a simple neural network baseline.

In the search correlation prediction part, we consider the monotonicity between tags, replace the Rank model with a regression model, filter the original features for variance, and directly put them into the xgboost and catboost models for learning and get the answer.
