from Preprocessing.preprocessing import *
from parameters import *
# from configuration import *
from parameters import MetaParameters
from Preprocessing.dirs_logs import *
from Evaluation.evaluation import *



########################################################################################################################
# COMMENTS
########################################################################################################################
class SaveDataset(MetaParameters):
    
    def __init__(self, files, pickle_name): 
        
        super(MetaParameters, self).__init__()
        self.files = files
        self.pickle_name = pickle_name

    def save_to_pickle(self):
        list_images, list_masks, list_names = [], [], []
        count = 0
        
        for file_name in self.files:
            if file_name.endswith('.nii'):
                
                sub_name = file_name.replace('.nii', '')

                images = view_matrix(read_nii(f"{self.ORIGS_DIR}/{file_name}"))
                masks = view_matrix(read_nii(f"{self.MASKS_DIR}/{file_name}"))

                slice_count = images.shape[2]

                if self.PRESEGMENTATION is True:
                    preseg = EvalPreprocessData(images, masks).presegmentation_tissues()
                    images = preseg[0]
                    masks = preseg[1] 

                for slc in range(slice_count):
                    count += 1
                    image = images[:, :, slc]
                    mask = masks[:, :, slc]
                    
                    # if (mask==4).any() or 0 <= (mask[mask == 3].sum() + 1) / (mask[mask == 2].sum() + 1) < 0.03:
                    if (mask==4).any():
                        pass    
                    else:
                        normalized = PreprocessData(image, mask).normalization()
                            
                        list_images.append(normalized[0])
                        list_masks.append(normalized[1])
                        list_names.append(f'{sub_name} Slice {slice_count - slc}')
                                
        print(f'Count of slice in {self.pickle_name} dataset: {len(list_names)}')
        shuff = shuff_dataset(list_images, list_masks, list_names)
        # shuff = [list_images, list_masks, list_names]
        save_pickle(f'{self.PREPROCESSED_DATASET}{self.DATASET_NAME}_{self.pickle_name}_origin.pickle', shuff[0])
        save_pickle(f'{self.PREPROCESSED_DATASET}{self.DATASET_NAME}_{self.pickle_name}_mask.pickle', shuff[1])
        save_pickle(f'{self.PREPROCESSED_DATASET}{self.DATASET_NAME}_{self.pickle_name}_sub_names.pickle', shuff[2])   


#########################################################################################################################
# Create subject list and after shuffling it, split to train, valid and test sets
#########################################################################################################################

# dataset_list = get_dataset_list(f'{meta.ORIGS_DIR}')
# random.shuffle(dataset_list)
# random.shuffle(dataset_list)

# dataset_size = len(dataset_list)
# test_list = dataset_list[round(0.8 * dataset_size):]
# train_list  = list(set(dataset_list) - set(test_list))

# train_dataset_size = len(train_list)
# print(f'Dataset: size - {len(dataset_list)}, List - {dataset_list}')
# print(f'Test dataset: size - {len(test_list)}, List - {test_list}')


# valid_list_01 = train_list[round(0.8*train_dataset_size):]
# train_list_01 = list(set(train_list)-set(valid_list_01))

# valid_list_02 = train_list[round(0.6*length_list):round(0.8*length_list)]
# train_list_02 = list(set(train_list)-set(valid_list_02))

# valid_list_03 = train_list[round(0.4*length_list):round(0.6*length_list)]
# train_list_03 = list(set(train_list)-set(valid_list_03))

# valid_list_04 = train_list[round(0.2*length_list):round(0.4*length_list)]
# train_list_04 = list(set(train_list)-set(valid_list_04))

# valid_list_05 = train_list[:round(0.2*length_list)]
# train_list_05 = list(set(train_list)-set(valid_list_05))


########################################################################################################################
## Fixed Shuffled Subject List for cross-validation [five folds]
########################################################################################################################
valid_list_01 = [
    'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub97.nii', 'Sub92.nii', 'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii',
    'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 'Sub100.nii']

train_list_01 = [
    'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
    'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii',
    'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii',
    'Sub73.nii', 'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub113.nii', 'Sub74.nii',
    'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub52.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii',
    'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub114.nii', 'Sub44.nii',
    'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub13.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii',
    'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub77.nii']

valid_list_02 = [
    'Sub114.nii', 'Sub44.nii', 'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub13.nii',
    'Sub26.nii',
    'Sub75.nii', 'Sub80.nii', 'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii']

train_list_02 = [
    'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
    'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii',
    'Sub31.nii',
    'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii', 'Sub73.nii',
    'Sub91.nii',
    'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub113.nii', 'Sub74.nii', 'Sub01.nii', 'Sub81.nii',
    'Sub88.nii',
    'Sub38.nii', 'Sub06.nii', 'Sub52.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii', 'Sub28.nii', 'Sub02.nii', 'Sub08.nii',
    'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub97.nii', 'Sub92.nii',
    'Sub54.nii',
    'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii',
    'Sub100.nii',
    'Sub77.nii']

valid_list_03 = [
    'Sub113.nii', 'Sub74.nii', 'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub06.nii', 'Sub52.nii', 'Sub38.nii',
    'Sub25.nii',
    'Sub12.nii', 'Sub11.nii', 'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii']

train_list_03 = [
    'Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
    'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii',
    'Sub31.nii',
    'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii', 'Sub73.nii',
    'Sub91.nii',
    'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub114.nii', 'Sub44.nii', 'Sub70.nii', 'Sub95.nii',
    'Sub79.nii',
    'Sub34.nii', 'Sub98.nii', 'Sub13.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii', 'Sub59.nii', 'Sub106.nii',
    'Sub24.nii',
    'Sub15.nii', 'Sub109.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub97.nii', 'Sub92.nii', 'Sub54.nii',
    'Sub82.nii',
    'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 'Sub100.nii',
    'Sub77.nii']

valid_list_04 = [
    'Sub84.nii', 'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii',
    'Sub62.nii',
    'Sub73.nii', 'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub77.nii']

train_list_04 = ['Sub22.nii', 'Sub94.nii', 'Sub18.nii', 'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii',
                 'Sub93.nii', 'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii',
                 'Sub30.nii', 'Sub113.nii',
                 'Sub74.nii', 'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub52.nii', 'Sub25.nii',
                 'Sub11.nii',
                 'Sub12.nii', 'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii',
                 'Sub114.nii', 'Sub44.nii',
                 'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub13.nii', 'Sub26.nii', 'Sub75.nii',
                 'Sub80.nii',
                 'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub57.nii', 'Sub16.nii',
                 'Sub04.nii', 'Sub97.nii',
                 'Sub92.nii', 'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii',
                 'Sub64.nii',
                 'Sub51.nii', 'Sub100.nii']

valid_list_05 = [
    'Sub22.nii', 'Sub94.nii', 'Sub37.nii', 'Sub18.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii',
    'Sub72.nii', 'Sub107.nii', 'Sub87.nii', 'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii']

train_list_05 = [
    'Sub84.nii', 'Sub31.nii', 'Sub67.nii', 'Sub60.nii', 'Sub17.nii', 'Sub63.nii', 'Sub110.nii', 'Sub89.nii',
    'Sub62.nii',
    'Sub73.nii', 'Sub91.nii', 'Sub27.nii', 'Sub05.nii', 'Sub45.nii', 'Sub36.nii', 'Sub113.nii', 'Sub74.nii',
    'Sub01.nii',
    'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub52.nii', 'Sub25.nii', 'Sub11.nii', 'Sub12.nii', 'Sub28.nii',
    'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii', 'Sub114.nii', 'Sub44.nii', 'Sub70.nii',
    'Sub95.nii',
    'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub13.nii', 'Sub26.nii', 'Sub75.nii', 'Sub80.nii', 'Sub59.nii',
    'Sub106.nii',
    'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub97.nii', 'Sub92.nii',
    'Sub54.nii',
    'Sub82.nii', 'Sub58.nii', 'Sub71.nii', 'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii',
    'Sub100.nii',
    'Sub77.nii']

train_list_full = [
    'Sub57.nii', 'Sub16.nii', 'Sub04.nii', 'Sub97.nii', 'Sub92.nii', 'Sub54.nii', 'Sub82.nii', 'Sub58.nii', 'Sub71.nii',
    'Sub90.nii', 'Sub55.nii', 'Sub50.nii', 'Sub64.nii', 'Sub51.nii', 'Sub100.nii', 'Sub22.nii', 'Sub94.nii',
    'Sub18.nii',
    'Sub37.nii', 'Sub10.nii', 'Sub33.nii', 'Sub21.nii', 'Sub29.nii', 'Sub93.nii', 'Sub72.nii', 'Sub107.nii',
    'Sub87.nii',
    'Sub108.nii', 'Sub20.nii', 'Sub68.nii', 'Sub30.nii', 'Sub84.nii', 'Sub31.nii', 'Sub67.nii', 'Sub60.nii',
    'Sub17.nii',
    'Sub63.nii', 'Sub110.nii', 'Sub89.nii', 'Sub62.nii', 'Sub73.nii', 'Sub91.nii', 'Sub05.nii', 'Sub45.nii',
    'Sub36.nii',
    'Sub113.nii', 'Sub74.nii', 'Sub01.nii', 'Sub81.nii', 'Sub88.nii', 'Sub38.nii', 'Sub06.nii', 'Sub52.nii',
    'Sub25.nii',
    'Sub11.nii', 'Sub12.nii', 'Sub28.nii', 'Sub02.nii', 'Sub08.nii', 'Sub103.nii', 'Sub78.nii', 'Sub47.nii',
    'Sub114.nii',
    'Sub44.nii', 'Sub70.nii', 'Sub95.nii', 'Sub79.nii', 'Sub34.nii', 'Sub98.nii', 'Sub13.nii', 'Sub26.nii', 'Sub75.nii',
    'Sub80.nii', 'Sub59.nii', 'Sub106.nii', 'Sub24.nii', 'Sub15.nii', 'Sub109.nii', 'Sub77.nii', 'Sub27.nii']

test_list = [
    'Sub35.nii', 'Sub07.nii', 'Sub112.nii', 'Sub99.nii', 'Sub76.nii', 'Sub56.nii', 'Sub111.nii', 'Sub85.nii',
    'Sub66.nii',
    'Sub32.nii', 'Sub53.nii', 'Sub83.nii', 'Sub61.nii', 'Sub49.nii', 'Sub42.nii', 'Sub14.nii', 'Sub69.nii',
    'Sub105.nii',
    'Sub03.nii', 'Sub23.nii']

train_list = [
    'Sub82.nii', 'Sub22.nii', 'Sub71.nii', 'Sub108.nii', 'Sub18.nii', 'Sub106.nii', 'Sub62.nii', 'Sub74.nii',
    'Sub36.nii',
    'Sub34.nii', 'Sub84.nii', 'Sub08.nii', 'Sub91.nii', 'Sub26.nii', 'Sub01.nii', 'Sub92.nii', 'Sub47.nii', 'Sub81.nii',
    'Sub110.nii', 'Sub103.nii', 'Sub25.nii', '.DS_Store', 'Sub90.nii', 'Sub54.nii', 'Sub75.nii', 'Sub60.nii',
    'Sub89.nii',
    'Sub107.nii', 'Sub12.nii', 'Sub24.nii', 'Sub70.nii', 'Sub67.nii', 'Sub04.nii', 'Sub27.nii', 'Sub50.nii',
    'Sub02.nii',
    'Sub72.nii', 'Sub79.nii', 'Sub57.nii', 'Sub58.nii', 'Sub13.nii', 'Sub37.nii', 'Sub16.nii', 'Sub77.nii', 'Sub45.nii',
    'Sub98.nii', 'Sub93.nii', 'Sub78.nii', 'Sub29.nii', 'Sub63.nii', 'Sub44.nii', 'Sub114.nii', 'Sub17.nii',
    'Sub30.nii',
    'Sub109.nii', 'Sub87.nii', 'Sub113.nii', 'Sub88.nii', 'Sub21.nii', 'Sub68.nii', 'Sub06.nii', 'Sub51.nii',
    'Sub20.nii',
    'Sub80.nii', 'Sub05.nii', 'Sub64.nii', 'Sub15.nii', 'Sub95.nii', 'Sub100.nii', 'Sub38.nii', 'Sub10.nii',
    'Sub28.nii',
    'Sub55.nii', 'Sub59.nii', 'Sub52.nii', 'Sub31.nii', 'Sub97.nii', 'Sub73.nii', 'Sub11.nii', 'Sub94.nii', 'Sub33.nii']


if __name__ == "__main__":
    ########################################################################################################################
    ## Create datasets and save it into pickle files
    ########################################################################################################################
    # SaveDataset(train_list_01, 'train_01').save_to_pickle()
    # SaveDataset(valid_list_01, 'valid_01').save_to_pickle()

    SaveDataset(train_list_02, 'train_02').save_to_pickle()
    SaveDataset(valid_list_02, 'valid_02').save_to_pickle()

    SaveDataset(train_list_03, 'train_03').save_to_pickle()
    SaveDataset(valid_list_03, 'valid_03').save_to_pickle()

    SaveDataset(train_list_04, 'train_04').save_to_pickle()
    SaveDataset(valid_list_04, 'valid_04').save_to_pickle()

    SaveDataset(train_list_05, 'train_05').save_to_pickle()
    SaveDataset(valid_list_05, 'valid_05').save_to_pickle()

    # # SaveDataset(train_list_full, 'train_full').save_to_pickle()

    SaveDataset(test_list, 'test').save_to_pickle()


