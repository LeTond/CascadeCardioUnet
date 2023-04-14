import torch######################################################################################################################## COMMENTS#######################################################################################################################class MetaParameters:	CROPPING = True	# CROPPING = False	## 	Network configuration	KERNEL = 128	CROPP_KERNEL = 64	CHANNELS = 1	NUM_LAYERS = 4	LR = 1e-3	BT_SZ = 32	EPOCHS = 1000	DROPOUT = 0.2	FEATURES = 32	WDC = 1e-4	EARLY_STOPPING = 20	TMAX = 50	SHUFFLE = True	# CLIP_RATE = [0.45, 0.95]	CLIP_RATE = None	## CrossEntropy Weights for labeles [Backgroung, LeftVentricle, Myocardium, Fibrosis]	DICT_LAYERS = {					0: "Background",					1: "Left_Ventricle",					2: "Myocardium",					3: "Fibrosis",					# 4: "Tromb",					# 5: "No_reflow",					}	# CE_WEIGHTS = torch.FloatTensor([0.5, 0.6, 0.7, 1.0])	# CE_WEIGHTS = torch.FloatTensor([0.1, 0.2, 0.5, 1.0])	# CE_WEIGHTS = torch.FloatTensor([0.2, 0.4, 0.7, 1.0])	CE_WEIGHTS = torch.FloatTensor([0.5, 0.7, 0.5, 0.9])	# CE_WEIGHTS = torch.FloatTensor([0.4, 0.6, 0.8, 1.0])	# CE_WEIGHTS = torch.FloatTensor([0.5, 0.5, 0.5, 1.0])	# CE_WEIGHTS = torch.FloatTensor([0.5, 1.0, 0.5, 0.5])	# CE_WEIGHTS = torch.FloatTensor([0.5, 0.5, 0.8, 1.0])	# CE_WEIGHTS = torch.FloatTensor([0.5, 0.9, 0.5, 1.0])	# CE_WEIGHTS = torch.FloatTensor([0.5, 0.8, 0.8, 0.8])	# CE_WEIGHTS = torch.FloatTensor([0.1, 0.2, 0.33, 1.0])	# 	Project configuration	FOLD_NAME = '02'	# [01:05] or full	DATASET_NAME = 'ALMAZ'	# DATASET_NAME = 'HEAD'	# DATASET_NAME = 'HCM'	# DATASET_NAME = 'EMIDEC'	MODEL_NAME = 'model_best'	DATASET_DIR = './Dataset/'	FOLD = f'Fold_{FOLD_NAME}'	CROPP_FOLD = f'Cropp_Fold_{FOLD_NAME}'	EVAL_DIR = f'./Results/{FOLD}/'	CROPP_EVAL_DIR = f'./Results/{CROPP_FOLD}/'	PROJ_NAME = f'{EVAL_DIR}Lr({LR})_Drp({DROPOUT})_batch({BT_SZ})_L2({WDC})'	CROPP_PROJ_NAME = f'{CROPP_EVAL_DIR}Lr({LR})_Drp({DROPOUT})_batch({BT_SZ})_L2({WDC})'	ORIGS_DIR = f'{DATASET_DIR}{DATASET_NAME}_origin'	MASKS_DIR = f'{DATASET_DIR}{DATASET_NAME}_mask'meta = MetaParameters()