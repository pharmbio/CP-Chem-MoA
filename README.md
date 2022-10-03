<p align="center">
Combining molecular and cell painting image data for mechanism of action prediction 
</p>

We present a method to predict the clusters that new chemicals belong to based on network topology. Our work contains three stages. We run the codes on Google Colab.  

<Stage 1: Predicting MoA using compound structure based model based on molecular data>, folder name: Compound_structure_based_models     
· The installation of RDkit: RDkit.ipynb
· Predicting MoA based on molecular data using multi-layer perceptron (MLP): MLP.ipynb   
· Predicting MoA based on molecular data using graph convolutional network (GCN): GCN.ipynb   
· Predicting MoA based on molecular data using convolutional neural network (CNN): CNN.ipynb   
· Predicting MoA based on molecular data using long short-term memory (LSTM) without data augmentation: LSTM.ipynb   
· Predicting MoA based on molecular data using long short-term memory (LSTM) with data augmentation: LSTM_aug.ipynb   
· Predicting MoA based on molecular data using traditional machine learning algorithms: traditional_machine_learning_algorithms.ipynb   

<Stage 2: Predicting MoA using cell morphology based model based on image data>, folder name: Cell_morphology_based_model_and_global_model 
· Predicting MoA based on image data using state-of-the-art CNN: CNN_MLP_Global.ipynb  

<Stage 3: Predicting MoA using global model based on the integration of molecular data and image data>, folder name: Cell_morphology_based_model_and_global_model
· Predicting MoA based on the integration of molecular data and image data using global model: CNN_MLP_Global.ipynb    

Citation
Please cite:
Combining molecular and cell painting image data for mechanism of action prediction
Guangyan Tian, Philip J Harrison, Akshai P Sreenivasan, Jordi Carreras Puigvert, Ola Spjuth.
