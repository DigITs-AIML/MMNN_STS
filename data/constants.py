HEADERS_TO_CONVERT = [
    'Sex',
    'Location',
    'Diagnosis from Surgical Path',
    'Use of Neo-Adjuvant Chemo (Y/N)',
]

PRE_OP_PREDICTORS = [
    'Sex', # need to one hot encode
    'Age',
    'Location', # need to one hot encode
    'Diagnosis from Surgical Path', # OHE
    'Use of Neo-Adjuvant Chemo (Y/N)', #OHE
    'Tumor Size (MRI)',
    'Tumor Volume (cm^3) (MRI)',
    'Depth (from MRI) (0=superficial, 1=deep)',
    '(use this for model) Tumor Grade from biopsy (low=1, high=2)',
    'Mets at Presentation (0=no, 1=yes)',
    'RT (0=none, 1=pre, 2=post)',
]

POST_OP_PREDICTORS = [
    # 'Margin (negative==0, microscopically positive==1, grossly positive==2)',
    # 'Necrosis % (information not known prior to operation)',
    # 'LengthOR',
    # 'Wound VAC used',
    # 'Flap at time of initial surgery',
    # 'Flap type ("local" = "pedicled", "free" are labelled as such)', #OHE
]

STRATIFY_BY = [
    'RT (0=none, 1=pre, 2=post)',
    'Location',
    '(use this for model) Tumor Grade from biopsy (low=1, high=2)',
    'VolumeCutoff'
]

TARGETS_BINARY = [
    'Vital Status (1=dead, 0 = alive) ',
    'Distant Mets  (0 = No, 1 = Yes)'
]

TARGETS_TIME = [
    'FUtime',
    'Date of Distant Met',
    'Date of Surgery'
]

RADIOMICS_EXCLUDE_COLUMNS = [
    'diagnostics_Versions_PyRadiomics',
    'diagnostics_Versions_Numpy',
    'diagnostics_Versions_SimpleITK',
    'diagnostics_Versions_PyWavelet',
    'diagnostics_Versions_Python',
    'diagnostics_Configuration_Settings',
    'diagnostics_Configuration_EnabledImageTypes',
    'diagnostics_Image-original_Hash',
    'diagnostics_Image-original_Dimensionality',
    'diagnostics_Image-original_Spacing',
    'diagnostics_Image-original_Size',
    'diagnostics_Image-original_Mean',
    'diagnostics_Image-original_Minimum',
    'diagnostics_Image-original_Maximum',
    'diagnostics_Mask-original_Hash',
    'diagnostics_Mask-original_Spacing',
    'diagnostics_Mask-original_Size',
    'diagnostics_Mask-original_BoundingBox',
    'diagnostics_Mask-original_VoxelNum',
    'diagnostics_Mask-original_VolumeNum',
    'diagnostics_Mask-original_CenterOfMassIndex',
    'diagnostics_Mask-original_CenterOfMass',
    'MRI Accession #',
    'MRI Accession # (PostRT, pre-surgery MRI)',
    'Anon Name'	,
    'Case # (pulled from XNAT sheet)',
    'DateOfDeath',
    'Surgery_Date',
    'Length of survival',

]

RADIOMICS_LABEL_COLUMNS = [
    'VitalStatus',
    'Distant metastasis',
]

RADIOMICS_SURV_LABEL_COLUMNS = [
    'FUtime',
    'Time_MET',
]
RADIOMICS_UID = 'MRN'

HEADER_PAIRS = [
        ('Vital Status (1=dead, 0 = alive) ', 'FUtime'), 
        ('Distant Mets  (0 = No, 1 = Yes)', 'Time_MET'), 
    ]

NUMROWS=150

BUCKET = 's3_bucket_name'

IMAGE_DATA_MEAN = 286.90859071507913
IMAGE_DATA_STDDEV = 581.7816096485366

NUM_DURATIONS = 30
NUM_CLASSES = 2

CLASS_FREQUENCIES = [0.386363636, 0.352272727]