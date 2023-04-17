from Evaluation.evaluation import *
from Preprocessing.preprocessing import ReadImages
from parameters import MetaParameters
from configuration import *


#########################################################################################################################
##TODO: COMMENTS
#########################################################################################################################
class Evaluation(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()
        self.eval_dir = f'{self.EVAL_DIR}{self.DATASET_NAME}_mask_NEW/'
        self.cropp_eval_dir = f'{self.CROPP_EVAL_DIR}{self.DATASET_NAME}_mask_NEW'
        self.dataset_path = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'
        self.project_name = f'./Results/{self.FOLD}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'

    def base_evaluation(self, file_name):
        ##  Presegmenation with matrix size KERNEL by KERNEL    
        create_dir(self.eval_dir)
        neural_model = torch.load(f'{self.project_name}/{self.MODEL_NAME}.pth').to(device=device)
        images, image_shp, def_cord = GetListImages(file_name, self.eval_dir, self.dataset_path, preseg = False).array_list(self.KERNEL)
      
        masks_list = PredictionMask(neural_model, self.KERNEL, images, image_shp, def_cord).get_predicted_mask()
        NiftiSaver(masks_list, file_name, self.eval_dir).save_nifti()
        PdfSaver(file_name, self.dataset_path, self.eval_dir).save_pdf()


    def preseg_evaluation(self, file_name):
        ##  Segmenation with matrix size CROPP_KERNEL by CROPP_KERNEL
        create_dir(self.cropp_eval_dir)
        project_name = f'{self.CROPP_EVAL_DIR}/Lr({self.LR})_Drp({self.DROPOUT})_batch({self.BT_SZ})_L2({self.WDC})'
        neural_model = torch.load(f'{project_name}/{self.MODEL_NAME}.pth').to(device=device)        
        images, image_shp, def_cord = GetListImages(file_name, self.eval_dir, self.dataset_path, preseg = True).array_list(self.CROPP_KERNEL)

        masks_list = PredictionMask(neural_model, self.CROPP_KERNEL, images, image_shp, def_cord).get_predicted_mask()
        NiftiSaver(masks_list, file_name, self.cropp_eval_dir).save_nifti()
        PdfSaver(file_name, self.dataset_path, self.cropp_eval_dir).save_pdf()
 

    def run_process(self):

        dataset_list = ReadImages(f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/').get_dataset_list()
        for file_name in dataset_list:
            if file_name.endswith('.nii'):
                self.base_evaluation(file_name)
                print(f'New subject {file_name} was saved with base evaluation Model')

                if self.CROPPING is True:
                    self.preseg_evaluation(file_name)
                    print(f'New subject {file_name} was saved with presegment_evaluation Model')



if __name__ == "__main__":
    Evaluation().run_process()
