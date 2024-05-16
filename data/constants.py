HEADERS_TO_CONVERT = [
    'Sex',
    'Location3',
    'Diagnosis',
    'Chemo (Neoadjuvant)',
]

PRE_OP_PREDICTORS = [
    'Sex', # need to one hot encode
    'Age',
    'Location3', # need to one hot encode
    'Diagnosis', # OHE
    'Chemo (Neoadjuvant)', #OHE
    'TumorSize',
    'TumorVolume (cm^3)',
    'TumorDepth (1 = deep to fascia, 0 = superficial) ',
    'TumorGrade',
    'Metsatpresentation',
    'RT Type (0 = preop, 1 = postop', # Added per Dr. Bozzo
]

POST_OP_PREDICTORS = [
    'Margin (negative==0, microscopically positive==1, grossly positive==2)',
    'Necrosis % (information not known prior to operation)',
    'LengthOR',
]

STRATIFY_BY = [
    'RT Type (0 = preop, 1 = postop',
    'Location3',
    'TumorGrade',
    'VolumeCutoff'
]

TARGETS_BINARY = [
    'VitalStatus',
    'Distant metastasis',
]

TARGETS_TIME = [
    'FUtime',
    'Date of Distant Mets',
    'Surgery_Date',
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
        ('VitalStatus', 'FUtime'), 
        ('Distant metastasis', 'Time_MET'), 
    ]

NUMROWS=132

BUCKET = 'bucket_name'

IMAGE_DATA_MEAN = 286.90859071507913
IMAGE_DATA_STDDEV = 581.7816096485366

NUM_DURATIONS = 30
NUM_CLASSES = 2