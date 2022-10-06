# Combining molecular and cell painting image data for mechanism of action prediction 

In this work we developed a model capable of predicting mechanism of action (MoA) using both structural information from chemicals and morphological information from cell paining images.


---

### Setting up the environment
To create and activate the environment. <br>
```bash
conda env create -f environment.yml
conda activate chem-moa
pip install -q git+https://github.com/huggingface/transformers.git
```
To export the conda environment to jupyter notebook. <br>
```bash
python -m ipykernel install --user --name=chem-moa
```

<br>

---

# Our work contains three stages.


### Stage 1: Predicting MoA using compound structure based model based on molecular data
Folder name: [Compound_structure_based_models](Compound_structure_based_models)  
The models explored are given below.
  * [Multi-Layer Perceptron (MLP)](Compound_structure_based_models/MLP.ipynb)
  * [Graph Convolutional Network (GCN)](Compound_structure_based_models/GCN.ipynb)
  * [Convolutional Neural Network (CNN)](Compound_structure_based_models/CNN.ipynb)
  * [Long Short-Term Memory (LSTM) without data augmentation](Compound_structure_based_models/LSTM.ipynb)
  * [LSTM with data augmentation](Compound_structure_based_models/LSTM_aug.ipynb)
  * [Traditional machine learning algorithms](Compound_structure_based_models/traditional_machine_learning_algorithms.ipynb)
    * We explored random forests, light gradient boosting machines, cat boost, k-nearest neighbors classifiers, logistic regression, bagging, stacking, voting, and adaboost.        

### Stage 2: Predicting MoA using cell morphology based model based on image data 
Folder name: [Image_based_model](Image_based_model)
* [Efficient net](Image_based_model/Efficient_net.ipynb)

### Stage 3: Predicting MoA using global model based on the integration of molecular data and image data  
Folder name: [Cell_morphology_based_model_and_global_model](Cell_morphology_based_model_and_global_model)
* [Global model (CNN + MLP model)](Cell_morphology_based_model_and_global_model/CNN_MLP_Global.ipynb)

---

### Citation
Please cite:
> Combining molecular and cell painting image data for mechanism of action prediction          
> Guangyan Tian, Philip J Harrison, Akshai P Sreenivasan, Jordi Carreras Puigvert, Ola Spjuth.
> Status: Submitted
