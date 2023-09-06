# LIME Parametrization
Given a classifier, an attribution method assigns real values to input features of a sample which indicate their importance for the model's decision.
Our goal is to explore how explanations change for different choices of hyperparameters for attribution methods, taking a model-centric as well as a data-centric perspective in the parameter selection. In this project, we implement LIME with different parametrizations based on Model-centric and Data-centric views to observe how this would affect the attribution results. 
Local Interpretable Model-agnostic Explanations (LIME) is an interpretability technique used for explaining the predictions of complex machine learning models in a human-understandable manner. 
LIME aims to explain the prediction of any classifier or regressor in a faithful way by approximating it locally with an interpretable model. Explaining a prediction provides a qualitative understanding of the relationship between the instance components and the model prediction.
The main idea behind LIME is to perturb the input data slightly and observe how the predictions change. Therefore, LIME utilizes an interpretable representation, regardless of the actual features used by the model. 
We evaluate the effectiveness of our attribution methods by assessing the impact on the black-box model's certainty of covering different percentages of the document's tokens based on their attribution scores.
The experiments conducted confirm our assumption that it is effective to consider different perspectives, LIME when explaining an NLP black-box model.

