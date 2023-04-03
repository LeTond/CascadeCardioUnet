from configuration import *
from parameters import MetaParameters
from Validation.metrics import DiceLoss


ds = DiceLoss()


########################################################################################################################
##TODO: COMMENTS
########################################################################################################################

class TrainNetwork(MetaParameters):

    def __init__(self, model, optimizer, loss_function, train_loader, valid_loader, meta, ds):         
        super(MetaParameters, self).__init__()
        self.ds = ds 
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        if self.CROPPING is False:
            self.project_name = self.PROJ_NAME
            self.fold = self.FOLD
        elif self.CROPPING is True:
            self.project_name = self.CROPP_PROJ_NAME
            self.fold = self.CROPP_FOLD

    def get_metrics(self, loader_):
        
        self.model.eval()
        loss, dice_lv, dice_myo, dice_fib, dice_b = 0, 0, 0, 0, 0

        num_batches = len(loader_)
        
        with torch.no_grad():
            
            for inputs, labels, sub_names in loader_:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   

                predict = self.model(inputs)
                
                loss += self.loss_function(predict, labels)
                
                predict = torch.softmax(predict, dim=1)
                predict = torch.argmax(predict, dim=1)
                labels = torch.argmax(labels, dim=1)
                
                ##TODO: should delete all 1,2,3... and replace it by DICT_LAYERS[0] ....
                pred_lv = (predict == 1)
                labe_lv = (labels == 1)
                pred_myo = (predict == 2)
                labe_myo = (labels == 2)
                pred_fib = (predict == 3)
                labe_fib = (labels == 3)
                
                dice_lv += self.ds(pred_lv, labe_lv)
                dice_myo += self.ds(pred_myo, labe_myo)
                dice_fib += self.ds(pred_fib, labe_fib)
     
            mean_loss = (loss / num_batches)
            mean_dice_lv = (dice_lv / num_batches)
            mean_dice_myo = (dice_myo / num_batches)
            mean_dice_fib = (dice_fib / num_batches)
            mean_dice = (mean_dice_lv + mean_dice_myo + mean_dice_fib) / 3
            
        return mean_loss, mean_dice, mean_dice_lv, mean_dice_myo, mean_dice_fib 


    def train(self):
        trigger_times, the_last_loss = 0, 100
        
        for epoch in range(self.EPOCHS + 1):
            time_start_epoch = time.time()
            
            self.model.train()
            
            for inputs, labels, sub_names in self.train_loader:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   
          
                predict = self.model(inputs)
                train_loss = loss_function(predict, labels)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
            
            scheduler_gen.step() #g_mean_train_loss,  g_mean_valid_loss
            
            training = self.get_metrics(self.train_loader)
            validating = self.get_metrics(self.valid_loader)
            
            results = 'Train LOSS: {:.3f}; Train dice: {:.3f}; Train Dice LV: {:.3f}; Train Dice MYO: {:.3f}; Train Dice FIB: {:.3f};\n\
            Valid LOSS: {:.3f}; Valid dice: {:.3f}; Valid Dice LV: {:.3f}; Valid Dice MYO: {:.3f}; Valid Dice FIB: {:.3f};'.format(
                training[0], training[1], training[2], training[3], training[4], 
                validating[0], validating[1], validating[2], validating[3], validating[4]
            )

            log_stats(
                f'./Results/{self.fold}/{epoch},{train_loss.item()},{training[1]},{training[2]},{training[3]},{training[4]},{validating[0]},{validating[1]},{validating[2]},{validating[3]},{validating[4]}', 
                self.project_name
            )

            # results = 'Train LOSS: {:.3f}; Valid LOSS: {:.3f};'.format(training[0], validating[0])

            # log_stats(
            #     f'./Results/{self.FOLD}/{epoch},{train_loss.item()},{training[1]},{training[2]},{training[3]},{training[4]},{validating[0]},{validating[1]},{validating[2]},{validating[3]},{validating[4]}', 
            #     self.project_name
            # )

            if validating[0] > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)

                if trigger_times >= self.EARLY_STOPPING:
                    print('Early stopping!\nStart to test process.')
                    return self.model

            else:
                trigger_times = 0

            if validating[0] <= the_last_loss:
                the_last_loss = validating[0]
                torch.save(self.model, f'{self.project_name}/model_best.pth')
                print(f'{self.project_name}/model_best - {epoch} saved!')

            print(results)
            time_end_epoch = time.time()
            print(f'Epoch time: {round(time_end_epoch - time_start_epoch)} seconds') 
            
        return self.model


