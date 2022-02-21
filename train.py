#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
import logging
import argparse
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Start testing ...")
    if hook:
        hook.set_mode(smd.modes.EVAL)
    model.eval()
    running_loss=0
    running_corrects=0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {100*total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer, device, args, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Start training ...")
    best_acc=0
    image_dataset={'train':train_loader, 'valid':validation_loader}
    acc_counter=0
    
    for epoch in range(1, args.epochs + 1):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                if hook:
                    hook.set_mode(smd.modes.TRAIN)
                model.train()
            else:
                if hook:
                    hook.set_mode(smd.modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if batch_idx % args.log_interval  == 0:
                    logger.info(
                        "  Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*running_corrects/running_samples,
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_acc>best_acc:
                    best_acc=epoch_acc
                else:
                    acc_counter+=1

        if acc_counter==1:
            break
            
    return model
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    logger.info("Creating model ...")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )
    return model

def create_data_loaders(data_dirs, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Creating data loaders ...")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    data_transforms["test"] = data_transforms["valid"]

    image_datasets = {x: torchvision.datasets.ImageFolder(data_dirs[x], data_transforms[x]) for x in data_transforms.keys()}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=(batch_size if x=="train" else test_batch_size), shuffle=(x=="train")) for x in data_transforms.keys()}
    return data_loaders


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    logger.info(f"Saving model to '{path}' ...")
    torch.save(model.state_dict(), path)
    

def main(args):
    
    hook = smd.Hook.create_from_json_file()

    logger.info("Hyperparameters:")
    for key, value in vars(args).items():
        logger.info(f"  {key}:{value}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Running on '{}' ...".format(device))
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model.to(device)
    if hook:
        hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    if hook:
        hook.register_loss(loss_criterion)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    data_dirs = {
        "train": args.train_dir,
        "test": args.test_dir,
        "valid": args.valid_dir        
    }
    dataloaders = create_data_loaders(data_dirs, args.batch_size, args.test_batch_size)
    
    model=train(model, dataloaders["train"], dataloaders["valid"], loss_criterion, optimizer, device, args, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, dataloaders["test"], loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model, args.model_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status (default: 10)",
    )
    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "./model"))
    parser.add_argument("--train-dir", type=str, default=os.getenv("SM_CHANNEL_TRAIN", "./data/train"))
    parser.add_argument("--test-dir", type=str, default=os.getenv("SM_CHANNEL_TEST", "./data/test"))
    parser.add_argument("--valid-dir", type=str, default=os.getenv("SM_CHANNEL_VALID", "./data/valid"))
    
    args=parser.parse_args()
    
    main(args)
