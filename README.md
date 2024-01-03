The neural network model for assessing the likelihood of clinical pregnancy occurrence in a specific in vitro fertilization (IVF) protocol:

The developed deep neural network serves as a unique tool for IVF clinics, aiming to enhance the efficiency of internal quality control and the determination of its target indicators. This model, based on recurrent neural networks, exhibits high accuracy in predicting the occurrence or absence of clinical pregnancy in a specific IVF protocol (AUC 0.86; SD 0.064, Test accuracy 0.78, F1 Score 0.71).

Key features of the model and a comparison of its metrics with counterparts:

The model was trained on more than 8000 protocols with known transfer outcomes, showing no statistically significant differences across various age groups of patients regardless of the fertilization method. Further training and validation were conducted using 6000 protocols from a clinic in Russia and 4000 protocols from a foreign clinic. To verify the completeness of predictions, 1600 protocols with preimplantation genetic testing for aneuploidy (PGT-A) were analyzed, demonstrating the algorithm's effectiveness (AUC: 0.67–0.75). The model's accuracy was compared with logistic regression models and other machine learning approaches (AUC: 0.62–0.64), currently available as commercial solutions for pregnancy probability assessment in protocols, as well as other neural network-based solutions (63%–74%). High prediction accuracy of embryo transfer outcomes with the developed model was demonstrated during cross-validation (78%–87%).

The neural network model outperforms traditional machine learning prediction models in correctly classifying clinical pregnancy occurrences (OR = 6.66). Additionally, the developed neural network exhibits an AUC-ROC metric (0.68–0.73), comparable to time-lapse system models such as KIDScore™, IDAScore™ V.2 (0.67), Embryoscope, Life Whisperer (0.65), and Irvine Scientific based on morphokinetic characteristics, along with similar Precision-Recall metrics describing the chance of pregnancy occurrence. When compared with the metrics of the Eeva™ algorithm (AUC: 0.53–0.61) for individual parameters (Aivf) and AUC: 0.64 for combined parameters, as well as with GERI AI™ (overall accuracy of 67.8%, AUC: 0.61), and STEM (AUC = 0.77), the developed model demonstrates comparative accuracy and completeness in class descriptions. Similar results were obtained when comparing the developed model with other neural network solutions used in IVF, including the Alife Health artificial intelligence model (ROC-AUC: 0.62–0.64) and Fairtility artificial intelligence model (ROC-AUC: 0.68–0.70).
