from Preprocessing.preprocessing import *
from parameters import meta
# from configuration import *
# from parameters import MetaParameters
# from Evaluation.evaluation import *
import json
from pprint import pprint


#########################################################################################################################
# Create subject list and after shuffling it, split to train, valid and test sets
#########################################################################################################################
def create_folds_list():
    dataset_list = ReadImages(f'{meta.ORIGS_DIR}').get_dataset_list()
    random.shuffle(dataset_list)
    random.shuffle(dataset_list)

    dataset_size = len(dataset_list)
    test_list = dataset_list[round(0.8 * dataset_size):]
    train_list  = list(set(dataset_list) - set(test_list))

    train_dataset_size = len(train_list)

    valid_list_01 = train_list[round(0.8*train_dataset_size):]
    train_list_01 = list(set(train_list)-set(valid_list_01))

    valid_list_02 = train_list[round(0.6*train_dataset_size):round(0.8*train_dataset_size)]
    train_list_02 = list(set(train_list)-set(valid_list_02))

    valid_list_03 = train_list[round(0.4*train_dataset_size):round(0.6*train_dataset_size)]
    train_list_03 = list(set(train_list)-set(valid_list_03))

    valid_list_04 = train_list[round(0.2*train_dataset_size):round(0.4*train_dataset_size)]
    train_list_04 = list(set(train_list)-set(valid_list_04))

    valid_list_05 = train_list[:round(0.2*train_dataset_size)]
    train_list_05 = list(set(train_list)-set(valid_list_05))

    folds_list = {
                'train_list_01' : train_list_01,
                'valid_list_01' : valid_list_01,
                'train_list_02' : train_list_02,
                'valid_list_02' : valid_list_02,
                'train_list_03' : train_list_03,
                'valid_list_03': valid_list_03,
                'train_list_04' : train_list_04,
                'valid_list_04': valid_list_04,
                'train_list_05' : train_list_05,
                'valid_list_05' : valid_list_05,
                'train_list_full' : train_list,
                'valid_list_full' : test_list,
                'test_list' : test_list,
                }

    with open(f'{meta.DATASET_DIR}{meta.DATASET_NAME}_folds_list.json', "w") as fdct:
        json.dump(folds_list, fdct) # записываем структуру в файл


if os.path.exists(f'{meta.DATASET_DIR}{meta.DATASET_NAME}_folds_list.json'):
    with open(f'{meta.DATASET_DIR}{meta.DATASET_NAME}_folds_list.json', "r") as fdct:
        folds_dict = json.load(fdct)

        train_list = folds_dict[f'train_list_{meta.FOLD_NAME}']
        valid_list = folds_dict[f'valid_list_{meta.FOLD_NAME}']
        test_list = folds_dict[f'test_list']

        pprint(f'test_list = {test_list}')
        pprint(f'valid_list = {valid_list}')
        pprint(f'train_list = {train_list}')

else:
    create_folds_list()


























# if meta.DATASET_NAME != "ALMAZ":

#     dataset_list = ReadImages(f'{meta.ORIGS_DIR}').get_dataset_list()
#     random.shuffle(dataset_list)
#     random.shuffle(dataset_list)

#     dataset_size = len(dataset_list)
#     test_list = dataset_list[round(0.8 * dataset_size):]
#     train_list  = list(set(dataset_list) - set(test_list))

#     train_dataset_size = len(train_list)
#     # print(f'Full Dataset: size - {len(dataset_list)}, List - {dataset_list}')
#     # print(f'Train Dataset: size - {train_dataset_size}, List - {train_list}')
#     # print(f'Test dataset: size - {len(test_list)}, List - {test_list}')

#     valid_list_01 = train_list[round(0.8*train_dataset_size):]
#     train_list_01 = list(set(train_list)-set(valid_list_01))

#     valid_list_02 = train_list[round(0.6*train_dataset_size):round(0.8*train_dataset_size)]
#     train_list_02 = list(set(train_list)-set(valid_list_02))

#     valid_list_03 = train_list[round(0.4*train_dataset_size):round(0.6*train_dataset_size)]
#     train_list_03 = list(set(train_list)-set(valid_list_03))

#     valid_list_04 = train_list[round(0.2*train_dataset_size):round(0.4*train_dataset_size)]
#     train_list_04 = list(set(train_list)-set(valid_list_04))

#     valid_list_05 = train_list[:round(0.2*train_dataset_size)]
#     train_list_05 = list(set(train_list)-set(valid_list_05))

# else:

#     valid_list_01 = [
#         'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub48.nii', 'Sub92.nii', 'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii',
#         'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 'Sub100.nii']

#     train_list_01 = [
#         'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
#         'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii',
#         'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii',
#         'Sub73.nii', 'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub113.nii', 'Sub74.nii',
#         'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub40.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii',
#         'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub46.nii', 'Sub44.nii',
#         'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub19.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii',
#         'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub77.nii']

#     valid_list_02 = [
#         'Sub46.nii', 'Sub44.nii', 'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub19.nii',
#         'Sub26.nii', 'Sub75.nii', 'Sub80.nii', 'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii']

#     train_list_02 = [
#         'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
#         'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii',
#         'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii', 'Sub73.nii',
#         'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub113.nii', 'Sub74.nii', 'Sub01.nii', 'Sub81.nii',
#         'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub40.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii', 'Sub28.nii', 'Sub02.nii', 
#         'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub48.nii', 'Sub92.nii',
#         'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii',
#         'Sub100.nii', 'Sub77.nii']

#     valid_list_03 = [
#         'Sub113.nii', 'Sub74.nii', 'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub06.nii', 'Sub40.nii', 'Sub38.nii',
#         'Sub25.nii', 'Sub12.nii', 'Sub11.nii', 'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii']

#     train_list_03 = [
#         'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
#         'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii',
#         'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii', 'Sub73.nii',
#         'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub46.nii', 'Sub44.nii', 'Sub70.nii', 'Sub95.nii',
#         'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub19.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii', 'Sub59.nii', 'Sub106.nii',
#         'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub48.nii', 'Sub92.nii', 'Sub54.nii',
#         'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 'Sub100.nii',
#         'Sub77.nii']

#     valid_list_04 = [
#         'Sub84.nii', 'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii',
#         'Sub62.nii', 'Sub73.nii', 'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub77.nii']

#     train_list_04 = [
#         'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii', 
#         'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub113.nii', 'Sub74.nii', 
#         'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub40.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii', 
#         'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub46.nii', 'Sub44.nii', 'Sub70.nii', 
#         'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub19.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii', 'Sub59.nii', 
#         'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub48.nii', 'Sub92.nii', 
#         'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 
#         'Sub100.nii']

#     valid_list_05 = [
#         'Sub22.nii', 'Sub94.nii', 'Sub37.nii', 'Sub18.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
#         'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii']

#     train_list_05 = [
#         'Sub84.nii', 'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii',
#         'Sub62.nii', 'Sub73.nii', 'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub113.nii', 'Sub74.nii',
#         'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub40.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii', 
#         'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub46.nii', 'Sub44.nii', 'Sub70.nii',
#         'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub19.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii', 'Sub59.nii',
#         'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub48.nii', 'Sub92.nii',
#         'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii',
#         'Sub100.nii', 'Sub77.nii']

#     train_list_full = [
#         'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub48.nii', 'Sub92.nii', 'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii',
#         'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 'Sub100.nii', 'Sub22.nii', 'Sub94.nii',
#         'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii', 'Sub72.nii', 'Sub107.nii',
#         'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii', 'Sub31.nii', 'Sub67.nii', 'Sub60.nii',
#         'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii', 'Sub73.nii', 'Sub91.nii', 'Sub05.nii', 'Sub45.nii',
#         'Sub36.nii', 'Sub113.nii', 'Sub74.nii', 'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub40.nii',
#         'Sub25.nii', 'Sub11.nii', 'Sub12.nii', 'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii',
#         'Sub46.nii', 'Sub44.nii', 'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub19.nii', 'Sub26.nii', 
#         'Sub75.nii', 'Sub80.nii', 'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub77.nii', 'Sub27.nii']

#     valid_list_full = [
#         'Sub35.nii', 'Sub07.nii', 'Sub112.nii', 'Sub99.nii', 'Sub76.nii', 'Sub56.nii', 'Sub111.nii', 'Sub85.nii',
#         'Sub66.nii', 'Sub32.nii', 'Sub53.nii', 'Sub83.nii', 'Sub61.nii', 'Sub49.nii', 'Sub42.nii', 'Sub14.nii', 'Sub69.nii',
#         'Sub105.nii', 'Sub03.nii', 'Sub23.nii']

#     test_list = [
#         'Sub35.nii', 'Sub07.nii', 'Sub112.nii', 'Sub99.nii', 'Sub76.nii', 'Sub56.nii', 'Sub111.nii', 'Sub85.nii',
#         'Sub66.nii', 'Sub32.nii', 'Sub53.nii', 'Sub83.nii', 'Sub61.nii', 'Sub49.nii', 'Sub42.nii', 'Sub14.nii', 'Sub69.nii',
#         'Sub105.nii', 'Sub03.nii', 'Sub23.nii']

#     train_list = [
#         'Sub82.nii', 'Sub22.nii', 'Sub71.nii', 'Sub108.nii', 'Sub18.nii', 'Sub106.nii', 'Sub62.nii', 'Sub74.nii',
#         'Sub36.nii', 'Sub34.nii', 'Sub84.nii', 'Sub08.nii', 'Sub91.nii', 'Sub26.nii', 'Sub01.nii', 'Sub92.nii', 'Sub47.nii', 
#         'Sub81.nii', 'Sub110.nii', 'Sub103.nii', 'Sub25.nii', '.DS_Store', 'Sub90.nii', 'Sub54.nii', 'Sub75.nii', 'Sub60.nii',
#         'Sub89.nii', 'Sub107.nii', 'Sub12.nii', 'Sub24.nii', 'Sub70.nii', 'Sub67.nii', 'Sub04.nii', 'Sub27.nii', 'Sub50.nii',
#         'Sub02.nii', 'Sub72.nii', 'Sub79.nii', 'Sub57.nii', 'Sub58.nii', 'Sub19.nii', 'Sub37.nii', 'Sub16.nii', 'Sub77.nii', 
#         'Sub45.nii', 'Sub98.nii', 'Sub93.nii', 'Sub78.nii', 'Sub29.nii', 'Sub63.nii', 'Sub44.nii', 'Sub46.nii', 'Sub17.nii', 
#         'Sub30.nii', 'Sub109.nii', 'Sub87.nii', 'Sub113.nii', 'Sub88.nii', 'Sub21.nii', 'Sub68.nii', 'Sub06.nii', 'Sub51.nii',
#         'Sub20.nii', 'Sub80.nii', 'Sub05.nii', 'Sub64.nii', 'Sub15.nii', 'Sub95.nii', 'Sub100.nii', 'Sub38.nii', 'Sub10.nii',
#         'Sub28.nii', 'Sub55.nii', 'Sub59.nii', 'Sub40.nii', 'Sub31.nii', 'Sub48.nii', 'Sub73.nii', 'Sub11.nii', 'Sub94.nii', 
#         'Sub33.nii']



