# MMNN_STS

Repository supporting the training and evaluation of a multimodal neural network for prognostic modeling of soft tissue sarcoma patient outcomes


<h2> Installation </h2>

    git clone https://github.com/DigITs-AIML/MMNN_STS.git
    cd MMNN_STS && pip install -r requirements.txt

<h2> Usage </h2>

1. Modify config.yaml to point to your datasets

    - One tabular dataset with clinical variables (csv file)
    - One T1 Image Dataset
    - One T2 Image Dataset
        - If only one of T1 or T2 image dataset is used, change the 'modality' field to 't1' or 't2' accordingly
    - A patient key used to map UIDs across all datasets
        - If separate uids (MRNs & accession numbers, for instance) are used for the image datasets and tabular dataset, this lets you map both uids to the same patient
        - If the uids are the same across datasets, then copy and paste the UID column from your tabular dataset (with the header) and save as a csv file
    - Change the train_uid_location, val_uid_location, and test_uid_location fields to point to txt files containing one patient UID per line for each patient in the respective set
    - Change the model_weights field to point to an appropriate location to store trained model weights

2. Train model using...

        python main.py --images --preop --survival --blend
    
    - --images and --preop are modality flags, at least 1 modality is required
    - --survival is the task flag, This can be --survival for time to event modeling, or --classification for classification
    - --blend indicates is used to indicate gradient blending, and can be excluded if you do not wish to use gradient blending. This flag is ignored if only 1 modality flag is present

3. Evaluate model using...

        python main.py --inference --images --preop --survival --bootsrap --no_gradcam

    Where --inference is added to indicate model evaluation and --bootstrap is used to bootstrap models predictions and evaluate confidence intervals.

    You can use the --no_gradcam flag to forego creating attention maps for each patient prediction (strongly recommended whenever using --bootstrap)


<h2> Performance Metrics </h2>

|Model|C-Index (SD) for predicting Overall Survival|C-Index (SD) for predicting Distant Metastases|
|:----|:----|:----|
|Sarculator variables (CoxPH)|0.614 (0.117)|0.631 (0.097)|
|Larger clinical model (RSF)|0.655 (0.111)|0.639 (0.108)|
|Radiomics model (RSF)|0.532 (0.116)|0.541 (0.099)|
|Radiomics + Clinical (CoxPH with ElasticNet penalty)|0.707 (0.095)| 0.658 (0.085)|
|Unimodal DenseNet|0.553 (0.081)|0.562 (0.068)|
|**Multimodal Neural Network**|**0.769 (0.126)**|**0.699 (0.092)**|

<h2> Sagemaker </h2>

This repository supports model training and deployment on Amazon Sagemaker, contact the author at hollina1@mskcc.org for example notebooks.



    
    




