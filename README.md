Predicting Oscar Best Picture Winners: Utilising Multivariate Metrics and Classification Models for Inference 

Introduction
The allure of the Academy Awards ceremony lies not only in celebrating cinematic achievements but also in the suspense surrounding the coveted Best Picture award. Can a data-driven approach shed light on this unpredictable outcome? This project delves into the world of machine learning to explore the possibility of predicting Best Picture winners. By analysing a multitude of factors that influence a film's critical and commercial success, we aim to develop a robust model capable of making informed inferences about future Academy Award winners. Through this exploration, we hope to not only unveil some of the secrets behind Oscar glory but also gain valuable insights into the complex interplay between artistic merit, audience appeal, and industry trends that shape cinematic history.

Data Sources:
https://developer.imdb.com/non-commercial-datasets/
https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews
https://www.kaggle.com/datasets/unanimad/the-oscar-award
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Data Collection: 
For this project, data was collected from multiple sources to construct a comprehensive dataset for analysis. The primary sources utilised include IMDb's non-commercial datasets, the Clapper Rotten Tomatoes dataset available on Kaggle, the Oscar Award dataset also on Kaggle, and the Movies dataset on Kaggle. The datasets were merged using the film title as the key identifier. To handle cases where exact matches were not found, cosine similarity was employed to identify the best match, with further manual verification to ensure accuracy. Initially, four distinct datasets were obtained, comprising box office data with 5025 records, Rotten Tomatoes data with 143259 records, Oscar data with 10890 records, and IMDb data with a vast collection of 10680368 records. After merging the tables, movies nominated for the Best Picture category were filtered out, resulting in a dataset of 191 records, before focusing primarily on columns such as film title, category of nomination, award wins, IMDb ratings, Rotten Tomato ratings, and box office figures for subsequent analysis and modelling.

Data Selection and Preprocessing
 Data Size
The shape function revealed the dataset's dimensions to be:
IMDB dataset:  (10680368, 9)
Rotten tomatoes dataset: (143258, 16)
Oscars dataset: (10889, 8)
Box Office dataset: (5043, 28)

Duplicate Check
The duplicated() function was used to ensure the absence of duplicate entries in the dataset.
IMDB dataset: 0
Rotten tomatoes dataset: 1204 
Oscars dataset: 7
Box Office dataset: 45

Handling Missing Values
The info() function was used to identify data types and null values.

Considering columns of interest, Identified 19 NaN values in the 'Budget' column from the box office dataset which was a minor chunk of the data therefore decided to remove the records.

Column and Row Removal
Removed many columns not relevant for modelling eg: Language, Country, Color, criticName,reviewState etc

Summary Statistics
Utilised the describe() function to generate a statistical summary of the data.

Data Field Creation and Encoding
Certain qualitative columns of focus such as the Category of nomination/win , Movie Genre and Plot keywords needed to be cleaned and encoded. Movie Genre and Plot keywords were multivalued which needed to be split and encoded using one hot encoding.

New fields were created to count the number of award wins (apart from BEST PICTURE) and total number of nominations.

Exploratory Data Analysis (EDA) and Feature Selection

Data Imbalance Check
The number of films nominated for the Best Picture category exceeds the number of actual winners. This creates a data imbalance issue where the class representing winners is smaller compared to the class representing nominees and results in an uneven distribution of data points across different classes.

To address data imbalance, oversampling was performed using SMOTE(Synthetic Minority Oversampling Technique), which involves artificially increasing the number of instances in the minority class (in this case, the winners) to balance it with the majority class (the nominees).
 
Outlier Detection
There were outliers observed in certain fields of focus such as Gross revenue of movies, number of users who voted and the imdb score.

Feature Selection

To identify the most informative features for our model, random forest feature importance was used. This technique leverages the inherent structure of random forests to assess each feature's contribution to the splitting process within the decision trees. A key advantage of this approach is its computational efficiency, as feature importance is calculated during the random forest training itself. This allows us to prioritise features based on their actual impact on the model's performance, leading to a more interpretable and potentially more accurate final model.

Feature importance was performed at two stages, first one with just the information from the academy awards dataset and second with the merged final dataset. After scoring the importance , top 10 observed important features were selected , some contained qualitative variables such as Actor Name and Director Name which were encoded using label encoding, these were removed since label encoding did not seem like a good encoding technique as these fields had many unique values.

According to the graph, the most important feature is  DIRECTING_winner, which refers to whether or not a movie won an award for directing. Other important features include total_wins ( how many awards a movie won in total), actor_1_name (the name of the lead actor), total_nominations (how many nominations the movie received), and imdb_score (its rating on the IMDB website)

Strong Positive Correlations:
DIRECTING_winner is strongly positively correlated with total_wins and total_nominations. This suggests that movies which win the Director award tend to also win other awards and receive more nominations overall. This aligns with the idea that these awards are indicators of a well-made film that resonates with critics and voters.
total_nominations also has a strong positive correlation with total_wins, which is intuitive as more nominations often lead to more wins.

Moderate Positive Correlations:
imdb_score shows a moderate positive correlation with most other features. This indicates that movies with higher ratings tend to also be nominated for more awards and win more awards, including the Best Picture award.
There is a moderate positive correlation between DIRECTING_winner and num_critic_for_reviews, suggesting that movies that win the Director award tend to also be reviewed by a larger number of critics. This might reflect increased interest and critical attention surrounding these films.

Weak Positive/Negative Correlations:
The remaining correlations appear weak and mixed. There isn't a clear positive or negative association between actor_1_name (lead actor) and other features. This suggests that the star power of the lead actor might not be a determining factor for a Best Picture win according to this model.
is_best_picture is weakly correlated with most other features, but this is likely because it represents the target variable itself (whether the movie won Best Picture).

Model Implementation and Baseline Evaluation
This project explored three machine learning methods to predict movie Best Picture wins: Logistic Regression, KNN, and Random Forests. Logistic Regression was chosen for its effectiveness in binary classification tasks . KNN offered an advantage since the movie data included diverse feature types (numbers and categories). Finally, Random Forests were included due to their robustness to outliers and ability to capture complex relationships between features, potentially leading to more nuanced predictions

Comparison of Results :
Accuracy: Random Forest appears to have the highest accuracy (0.85) followed by KNN (0.833) and Logistic Regression (0.733). This suggests that Random Forest might be the most effective model for predicting Best Picture wins in this dataset.
Precision and Recall: 
Random Forest Precision (0.759) seems lower than Recall (0.88), indicating the model might predict some false positives (classifying non-winners as winners)
KNN has the highest F1 score (0.8302), followed closely by Random Forest (0.8148) and Logistic Regression (0.6522). This indicates that KNN might be the most effective model for predicting Best Picture wins in this dataset, considering both precision and recall. While KNN has the highest F1 score, the differences between KNN and Random Forest are relatively small. The Logistic Regression model has a noticeably lower F1 score compared to the other two models. This suggests that it might not be performing as well in capturing the nuances of the data required for accurate Best Picture win prediction.

Observations from the ROC curve:
Random Forest: The green curve represents Random Forest. It has the largest AUC among the three models, indicating the best overall performance in distinguishing between Best Picture winners and non-winners.

KNN (Orange Curve): Its AUC is lower than Random Forest but potentially higher than Logistic Regression. This suggests KNN might perform better than Logistic Regression in differentiating the classes.

Logistic Regression:  Logistic Regression has the smallest AUC, suggesting the weakest performance in differentiating between the classes. This aligns with the previous F1 score analysis where Logistic Regression had the lowest 

In conclusion Random Forest seems to be the most effective model based on the metrics calculated. Both the F1 score analysis and the ROC curve analysis point towards Random Forest's superior ability to balance between correctly identifying winners and avoiding false positives.

Hyperparameter Tuning
To ensure the best possible performance from each model, GridSearchCV was employed for hyperparameter tuning. GridSearchCV systematically evaluates a predefined grid of hyperparameter values for each model (Logistic Regression, KNN, and Random Forest). It then selects the combination that yields the best performance on a validation set, effectively mitigating bias and overfitting. This process helps identify the optimal configuration for each model, allowing it to learn from the data more effectively in the context of predicting movie Best Picture wins

Logistic Regression: A higher 'C' value might have led to overfitting, while a lower value could result in underfitting. The chosen value of 1.0 likely achieved a balance between these extremes.

KNN: The optimal number of neighbours (11) is likely dataset-specific. Too few neighbours might not capture enough information, while too many could lead to noise sensitivity.

Random Forest: The high number of estimators (100 trees) suggests a complex model was beneficial for this task. No limit on the maximum depth might have allowed the trees to learn intricate relationships between features.

While Random Forest still has the highest accuracy, all three models show improvements in accuracy compared to the values before tuning


Similar to the accuracy analysis, all three models show improvements in F1 scores compared to before tuning.


This analysis of the ROC curve after tuning strengthens the conclusion that Random Forest is the most effective model. It consistently has the highest AUC, indicating a superior ability to balance between correctly identifying winners (high TPR) and avoiding false positives (low FPR).

Conclusion and Recommendations

This project explored three machine learning models (Logistic Regression, KNN, and Random Forest) for predicting movie Best Picture wins. Here's a summary of the findings and recommendations:

Findings:
Random Forest emerged as the most effective model for predicting Best Picture wins based on various performance metrics (accuracy, F1 score, ROC AUC). This suggests it can effectively learn complex relationships between movie features and Best Picture outcomes.
Hyperparameter tuning using GridSearchCV played a crucial role in optimising the performance of all three models. It helped identify the best configurations for each model, leading to improvements in their ability to predict Best Picture wins.
KNN performed better than Logistic Regression after hyperparameter tuning. However, it still fell short of Random Forest's performance.
Logistic Regression, while being a good baseline model, showed the weakest performance among the three.

Recommendations:
For the task of predicting movie Best Picture wins based on the current dataset and problem type, Random Forest is the most recommended model. Its ability to handle complex relationships and achieve a good balance between precision and recall makes it a strong choice.
GridSearchCV is a valuable technique for hyperparameter tuning and should be considered when deploying any of these models for this task. It helps ensure optimal model performance by finding the best hyperparameter configurations.
Further exploration could involve incorporating additional features or using more advanced ensemble methods to potentially improve prediction accuracy.



