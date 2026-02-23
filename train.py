# Training script to train MiVOlO model on custom data
# Modified https://github.com/KoaBou/Masked_Age_Detection/blob/main/Mivolo_model_custome.ipynb
# Author: Robin Jacob Roy

import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from mivolo.model.mi_volo import MiVOLO
from mivolo.model import mivolo_model
from timm.utils import accuracy

from landmark_crop import FaceCropper
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, gc

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
mean=IMAGENET_DEFAULT_MEAN,
std=IMAGENET_DEFAULT_STD

import logging

import configparser


config = configparser.ConfigParser()

config.read('bucket_config.ini')

device = torch.device("cuda")

PATH = "models/model_imdb_cross_person_4.22_99.46.pth.tar"
ANNOTATION_PATH = 'YOLO_annotations.csv'
batch_size = 128
n_epochs = 500
start_epoch = 0

learning_rate = 1.5e-5 #1.5e-5 default lr of ADAMW is 1e-3
weight_decay = 5e-5

val_age_maes = []
train_losses = []
val_losses = []
best_val_mae = float('inf') 
best_val_loss = float('inf')

np.random.seed(0)#torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))



def balance_sampler() -> torch.utils.data.WeightedRandomSampler:
    target = train_data['age'].astype(int)
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. /class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in target])).double()

    return torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))



class EarlyStopper:
    """ This calss will check whether the current validation loss is greater than the 
    (min val loss + delta value) for 'patience' number of epochs.
    """
    
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




def class_letterbox(im, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    """
    function to add padding with black pixes to 
    face and body crops as the MiVOLO model expects.
    """
    
    
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def resize_and_preprocess(img):
    """Normalise the image after padding"""
    try:

        img = class_letterbox(img, new_shape=(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(dtype=np.float32)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        return img
    except Exception as e:
        # print(image.shape)
        if SHOW_LOG:
            print(f"Error in resizing and preprocessing: {e}")
        return None


# -

class dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        
        file = self.df.iloc[ix]
        age = torch.tensor((file.age-avg_age)/(max_age-min_age))        
        gender = torch.tensor(file.gender).view(-1)
        img_name = file.img_name
        
        face_bbox = (file.face_x0, file.face_y0, file.face_x1, file.face_y1)
        person_bbox = (file.person_x0, file.person_y0, file.person_x1, file.person_y1)       
    
        
        #f'data/AgeGender/FaceData/data/{img_id}.jpg'
        image = cv2.imread('data/AgeGender/FaceData/data/{img_name}')       
        im = image.copy()
        
        face = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        
        
        im[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]] = [0,0,0]
        
        person = im[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]
        
        person = resize_and_preprocess(person)
        face = resize_and_preprocess(face) 

        cropped_faces = torch.cat([person, face], dim=0)
        
        # im = torch.tensor(im).permute(2,0,1)
        #im = self.normalize(im)


        return cropped_faces, age, gender



#Read the feather data file and change it to suit the problem
df = pd.read_csv(ANNOTATION_PATH)
#df = df.sample(frac=1).reset_index(drop=True)
df['gender'] = df['gender'].map({'FEMALE': 1, 'MALE': 0})


max_age = df.age.max()
min_age = df.age.min()
avg_age = (max_age + min_age)/2  #As per the MiVOLO code

######################################################################################
#Precaution for uneven data distribution
y = df['age']
age_min = 5 * round(y.min()/5)
age_max = 5 * round(y.max()/5)

# Bin the target variable into discrete intervals
bins = np.linspace(age_min, age_max, 1)  # bins
y_binned = np.digitize(y, bins)
# Use stratified sampling to get the indices of the training and validation sets
train_indices, val_indices = train_test_split(np.arange(len(df)), test_size=0.1, stratify=y_binned, random_state=42)

######################################################################################


# Divide the data into training set (70%), validation set (20%), and test set (10%)
# train_ratio = 0.7
# val_ratio = 0.3

# # Calculate the number of samples for each set
# num_samples = len(df)
# num_train = int(train_ratio * num_samples)
# num_val = num_samples - num_train

# # Create random index to divide data
# indices = np.random.permutation(num_samples)

# # Divided into training set, validation set, and test set
# train_indices = indices[:num_train]
# val_indices = indices[num_train:]


#np.save("test_indices.npy", test_indices)

########################################################################################

# Get data corresponding to each step
train_data = df.iloc[train_indices]

val_data = df.iloc[val_indices]

# Create dataset and dataloader for training set
train_new = dataset(train_data)
train_loader = DataLoader(train_new, batch_size=batch_size, num_workers=4, pin_memory=True,\
                          shuffle=False, sampler=balance_sampler())

# Create dataset and dataloader for validation set
val_new = dataset(val_data)
val_loader = DataLoader(val_new, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)

# Print out the size of each data set
print("The number of samples in the training set:", len(train_indices))
print("The number of samples in the validation set:", len(val_indices))


# #### Setup the MiVOLO model

new_model = mivolo_model.mivolo_d1_224(in_chans=6, num_classes=3).to(device)

# +
checkpoint =  torch.load(PATH, map_location=device)
# new_model = MiVOLO(PATH, device)

# new_model = new_model.model
new_model.load_state_dict(checkpoint['state_dict'])
# -

# new_model = MiVOLO("models/custome_mivolo.pth.tar", device)
# new_model = new_model.model_load()


#loss function
age_criterion = nn.MSELoss().to(device, dtype = torch.float16)  
gender_criterion = nn.BCEWithLogitsLoss().to(device, dtype = torch.float16) 


optimizer = torch.optim.AdamW(new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs/10)



# ##############################################Continue training#########################################################

# if you want to continue training from a saved checkpoint
# load the optimiser state into the optimiser function




try:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.dtype != torch.float32 :
                #print(v.dtype)
                state[k] = v.to(device=device, dtype=torch.float32)
    scheduler.load_state_dict(checkpoint['scheduler']
    start_epoch = checkpoint['epoch']
    best_val_mae = checkpoint['best_val_MAE']
    print("Resuming from saved checkpoint")
except:
    print('Saved checkpoint is not having enough values to load')
    pass

# ########################################################################################################################
# ###################################Training ################################################################


#start = time.time()
start_timer()

#initialise tensorboard
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('MiVOLO_TensorBoard{}'.format(timestamp))

#Write validation MAE to a logger file
logger = logging.getLogger()
logging.basicConfig(filename='Validation_MAEs_{}.log'.format(timestamp), filemode='w', level=logging.INFO, \
                   format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

#instantiate early stopper class
early_stopper = EarlyStopper(patience=10, min_delta=.2)

for epoch in range(start_epoch, n_epochs):
    epoch_train_loss, epoch_val_loss = 0, 0
    val_gen_loss, val_gen_acc = 0, 0
    val_age_mae, ctr = 0, 0
    #loop over train and validation
    for loader, is_train in [(train_loader, True), (val_loader, False)]:
        new_model.train() if is_train else new_model.eval()
        
        looper = tqdm(loader, desc=f"Epoch {epoch + 1}/{n_epochs} {'Train' if is_train else 'Validation'}")
        #loop over each batch
        for data in looper:
            im, age, gen = data

            im, age, gen = im.to(device, dtype = torch.float16), \
                              age.to(device, dtype = torch.float16), gen.to(device,dtype = torch.float16)
            #added as the gender loss was not getting backpropagated
            gen.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                new_model.half()
                kept = new_model(im)
                   
                pred_age = kept[:, 2]
                pred_gen = kept[:, :2].softmax(-1)
                
                gender_probs, gender_indx = pred_gen.topk(1)
                
                gender_probs[gender_indx == 0] = 1 - gender_probs[gender_indx == 0]
                # print("============")
                # print(pred_age[0])
                # print(age)
                # print("============")

                age_loss = age_criterion(pred_age.to(device, dtype = torch.float16), age)
                # print("age_loss: ", age_loss)
                gen_loss = gender_criterion(gender_probs.to(device, dtype = torch.float16), gen)
                
                gender_acc1 = (gender_probs.round() == gen).float().mean()
                # print("gen_loss: ", gen_loss)
                total_loss = age_loss + 0.03*gen_loss


                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f'Warning: NaN or Inf encountered in {"train" if is_train else "validation"} loss. Skipping update.')
                    continue

                if is_train:
                    
                    total_loss.backward()
                    new_model.float()
                    optimizer.step()
                    epoch_train_loss += total_loss.item()
                    denz_age = age*(max_age-min_age)+avg_age
                    denz_pred_age = pred_age*(max_age-min_age)+avg_age
                    age_mae = (torch.abs(denz_age - denz_pred_age).float().sum())/batch_size
                    looper.set_postfix({'mae': round(age_mae.item(),2), 'Gender accuracy' : gender_acc1.item() })
                else:
                    
                    denz_age = age*(max_age-min_age)+avg_age
                    denz_pred_age = pred_age*(max_age-min_age)+avg_age
                    age_mae = (torch.abs(denz_age - denz_pred_age).float().sum())/batch_size
                    looper.set_postfix({'mae': round(age_mae.item(),2), 'Gender accuracy' : gender_acc1.item() })
                    val_age_mae += age_mae
                    epoch_val_loss += total_loss.item()
                    val_gen_loss += gen_loss.item()
                    val_gen_acc += gender_acc1.item()
                    ctr += 1#len(data[0])
        
        if is_train:
            scheduler.step()
        torch.cuda.empty_cache()
    
    # Log the running loss averaged per batch
    # for both training and validation
    val_age_mae /= ctr
    epoch_train_loss /= len(train_loader)
    epoch_val_loss /= len(val_loader)
    val_gen_loss /= len(val_loader)
    val_gen_acc /= len(val_loader)
    elapsed = time.time() - start_time  
    best_val_loss = min(best_val_loss, epoch_val_loss)
    
    logger.info(f'Epoch = {epoch+1} : Train Age MAE = {age_mae}, Val Age MAE = {val_age_mae}, Train Age loss(MSE) = {epoch_train_loss}, Validation gender loss = {val_gen_loss}, Validation gender accuracy = {val_gen_acc}\n')
    
    
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : epoch_train_loss, 'Validation' : epoch_val_loss },
                    epoch + 1)
    writer.add_scalar('Validation MAE', val_age_mae, epoch + 1)
    writer.add_scalar('Validation Gender Accuracy', val_gen_acc, epoch + 1)
    writer.flush()
    
    if val_age_mae < best_val_mae or epoch==n_epochs-1:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if epoch==n_epochs-1:
            name = f"models/MiVOLO_last_Gndmodel_{timestamp}.pt"
        else:
            name = f"models/MiVOLO_Gndbest_epoch{epoch+1}_{timestamp}.pt"
        best_val_mae = epoch_val_loss
        
        print(f"saving the model into {name}")
        torch.save({
            'epoch': epoch+1,
            'state_dict': new_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'age_loss': age_loss,
            'gender_loss': gen_loss,
            'min_age': min_age,
            'max_age': max_age,
            'avg_age': avg_age,
            'no_gender': False,
            'with_persons_model': True,
            'best_val_MAE' : best_val_mae
            }, name)
    
    
    print(f'{epoch + 1}/{n_epochs} ({elapsed:.2f}s - {(n_epochs - epoch) * (elapsed / (epoch + 1)):.2f}s remaining)')
    info = f'''Epoch: {epoch + 1} \tTrain Loss: {epoch_train_loss:.3f} \tVal Loss: {epoch_val_loss:.3f} \tBest Val Loss: {best_val_loss:.4f}'''
    info += f'%\t Age MAE: {val_age_mae}'  
    print(info)
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    val_age_maes.append(val_age_mae)
    if early_stopper.early_stop(val_age_mae):             
        break

end_timer_and_print("Default precision:")

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Age_loss.png")

del train_loader, val_loader
