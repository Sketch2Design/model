import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import argparse
import time
import os

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils import Loss, get_num_of_classes, get_device, loader_fn, get_classes, Resize, get_train_transform, get_test_transform, show_tranformed_image
from model import create_model

plt.style.use('ggplot')

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for _, data in enumerate(prog_bar):
        
        optimizer.zero_grad()
        
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # loss
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()

        # gradient normalization
        clipping_value = 1 
        torch.nn.utils.clip_grad_norm(params, clipping_value)
        
        optimizer.step()
        
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f} Iter: {train_itr} ")
        
    return train_loss_list



# function for running validation iterations
def test(test_data_loader, model):
    print('Validating')
    global test_itr
    global mAP
    
    # initialize tqdm progress bar
    prog_bar = tqdm(test_data_loader, total=len(test_data_loader))
    
    for _, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        mAP.reset() 

        model.eval()
        with torch.inference_mode():
            prediction = model(images)
        mAP.update(prediction, targets)
        
        test_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f} Iter: {test_itr}")

    mean_ap = mAP.compute()
        
    return mean_ap


if __name__ == '__main__':

    DEVICE = get_device()

	# arg parser initailizing
    engine_parser = argparse.ArgumentParser()
    engine_parser.add_argument("--model", type=str, required= True, help="Export model name")
    engine_parser.add_argument("--export", type=str, required= True, help="Export model path")
    engine_parser.add_argument("--lr", type=float, required= False, help="Learning Rate", default=0.001)
    engine_parser.add_argument("--epochs", type=int, required= False, help="Number of epochs", default=1000)
    engine_parser.add_argument("--ckpt", type=int, required= False, help="Number of epochs", default=200)
    engine_parser.add_argument("--batch", type=int, required= False, help="Train Test split", default=4)
    args = engine_parser.parse_args()

   
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    dname_list = dname.split('/')
    dname_list.pop()
    dname_list.extend(['dataset', 'train'])
    train_path = "/".join(dname_list)
    dname_list.pop()
    dname_list.append('test')
    test_path = "/".join(dname_list)

    # tensorboard setup
    writer = SummaryWriter()

    # resize dataset
    RESIZE_TO = 512
    CLASSES = get_classes()

    train_dataset = Resize(train_path, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    test_dataset = Resize(test_path, RESIZE_TO, RESIZE_TO, CLASSES, get_test_transform())
   
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(test_dataset)}\n")

    # prepare data loaders
    train_loader = loader_fn(True,train_dataset,args.batch)
    test_loader = loader_fn(False,test_dataset,args.batch)
    
    # initialize the model and move to the computation device
    model = create_model(num_classes=get_num_of_classes())
    model = model.to(DEVICE)
    
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # define the learning rate scheduler
    warmup_epochs = 7
    warmup_factor = 0.01
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch / warmup_epochs * warmup_factor)
    
    # initialize the Loss store class
    train_loss_hist = Loss()

    train_itr = 1
    test_itr = 1
    
    # train and testing loss lists to store loss values of all...
    train_loss_list = []

    #metrics initialization
    mAP = torchmetrics.MeanAveragePrecision()
    
    # name to save the trained model with
    MODEL_NAME = args.model

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    dname_list = dname.split('/')
    root = "/".join(dname_list[:(len(dname_list) - 1)])
    os.chdir(root)
  
    # start the training epochs
    for epoch in range(args.epochs):
        print(f"\nEPOCH {epoch+1} of {args.epochs}")
        
        # reset the training and testing loss histories for the current epoch
        train_loss_hist.reset()
        test_loss_hist.reset()
        
        # create two subplots, one for each, training and testing
        figure_1, train_ax = plt.subplots()
        figure_2, test_ax = plt.subplots()
        
        # start timer and carry out training and testing
        start = time.time()

        if epoch < warmup_epochs:
            # Warm-up phase
            warmup_scheduler.step(epoch)
        else:
            scheduler.step()

        train_loss = train(train_loader, model)
        mean_ap = test(test_loader, model)

        writer.add_scalar('mAP', mean_ap, epoch) 
        
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} mAP: {mean_ap:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch+1) % args.ckpt == 0: # save model after every n epochs
            torch.save(model, f"{args.export}/{MODEL_NAME}{epoch+1}.pth")
            print('SAVING MODEL COMPLETED...\n')
        
        if (epoch+1) % args.ckpt == 0: # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            test_ax.plot(test_loss, color='red')
            test_ax.set_xlabel('iterations')
            test_ax.set_ylabel('test loss')
            figure_1.savefig(f"{args.export}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{args.export}/test_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
        
        if (epoch+1) == args.epochs: # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            test_ax.plot(test_loss, color='red')
            test_ax.set_xlabel('iterations')
            test_ax.set_ylabel('test loss')
            torch.save(model, f"{args.export}/{MODEL_NAME}{epoch+1}.pth")
        
        plt.close('all')