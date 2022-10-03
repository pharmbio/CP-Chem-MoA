Combining molecular and cell painting image data for mechanism of action prediction   
We present a method to predict the clusters that new chemicals belong to based on network topology. Our work contains three stages.   

<Stage 1: Predicting MoA using compound structure based model based on molecular data>  
· The installation of RDkit  
!get   




<Stage 2: Predicting MoA using cell morphology based model based on image data>


<Stage 3: Predicting MoA using global model based on the integration of molecular data and image data>




Setting up the environment
To create and activate the environment.

 

Data preprocessing
The data collection and preprocessing of the data are done using .ipynb files in preprocessing_scripts.

1_create_stitch_string_sql_db.ipynb
Download and convert the necessary data from STITCH and STRING to an sql database.

2_data_generation.ipynb
Quantmap is run using the interaction data from the databases. The data are then assigned to clusters based on their similarity using K-Mean clustering based on a range of distance parameters.

3_data_preprocessing.ipynb
From all the clusters obtained from the above step those clusters with low support are rejected.

4_get_protein_function_of_clusters.ipynb
For the clusters selected above chemical-protein information from STITCH is used to determine the main functions of proteins in each cluster.

5_data_splits.ipynb
Split the dataset for cross validation and final training of the model.

Evaluation
Initially different architectures were evaluated using cross validation based on a subset of data. The architectures explored are present in the directory cross_validation. The parameters for the architectures can be passed using their respective json file (the parameters given here are the default values).


Training
For the final training of the MolPMoFiT architecture, the entire dataset is used. The parameters can be passed using parameters.json file. In order to run the final training of the MolPMoFiT model, pretraining has to be first carried out using pretraining_molpmofit.ipynb. After the training of the final model it can be used to make predictions for new chemicals predict_new_chem.ipynb. The input for the prediction can be given in the text file "test_cids.txt" with CIDs as input.


Citation
Please cite:

