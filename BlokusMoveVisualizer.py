from solver import place_piece
from solver import yellow, blue, red, green
from uwimg import load_image
import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadTrainingImages(trainlocation, imageTransform):
    imageset = torchvision.datasets.ImageFolder(root = trainlocation, 
                                                transform=imageTransform)
    trainset, testset, validset = torch.utils.data.random_split(imageset, [0.7, 0.15, 0.15])
    trainimageloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True,
                                            num_workers=0)
    
    testimageloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True,
                                            num_workers=0)
    valimageloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True,
                                            num_workers=0)
    classes = ["F", "I1", "I2", "I3", "I4", "I5", "L4", "L5", "N", "O4", "P", "T4", "T5", "U", "V3", "V5", "W", "X", "Y", "Z4", "Z5"]
    return ({'train': trainimageloader, 'test': testimageloader, 'val': valimageloader, 'classes': classes}, len(trainset), len(validset))

def loadPieceImage(imageLocation, imageTransform):
    images = torchvision.datasets.ImageFolder(root = imageLocation, 
                                             transform = imageTransform)
    cleanImages = torchvision.datasets.ImageFolder(root = imageLocation,
                                                   transform = transforms.ToTensor())
    imageloader = torch.utils.data.DataLoader(images, batch_size = 1, shuffle = False,
                                              num_workers = 0)
    cleanImageLoader = torch.utils.data.DataLoader(cleanImages, batch_size = 1, shuffle = False,
                                              num_workers = 0)
    classes = ["F", "I1", "I2", "I3", "I4", "I5", "L4", "L5", "N", "O4", "P", "T4", "T5", "U", "V3", "V5", "W", "X", "Y", "Z4", "Z5"]
    return ({'pieces': imageloader, 'cleanpieces': cleanImageLoader, 'classes': classes})
def imshow(img):
    npimg = img.numpy()
    trans = np.transpose(npimg, (1, 2, 0))
    plt.imshow(trans)
    plt.show()

def accuracy(net, dataloader):
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in dataloader:
          images, labels = batch[0].to(device), batch[1].to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return correct/total

def verifymodel(blockmodel):
    images, trainlen, testlen = loadTrainingImages("TrainingImages/ProcessedEverythingColor", models.ResNet50_Weights.DEFAULT.transforms())

    blockmodel.eval()
    batch, labels = next(iter(images['train']))
    #Ground Truth
    print("Training Data")
    imshow(torchvision.utils.make_grid(batch))
    print('GroundTruth: ', ' '.join('%5s' % images['classes'][labels[j]] for j in range(len(labels))))

    outputs = blockmodel(batch.to(device))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % images['classes'][predicted[j]]
                                for j in range(len(labels))))

    print("Accuracy: ", accuracy(blockmodel, images['train']))

    print("Validation Data")
    batch, labels = next(iter(images['val']))
    imshow(torchvision.utils.make_grid(batch))
    print('GroundTruth: ', ' '.join('%5s' % images['classes'][labels[j]] for j in range(len(labels))))

    outputs = blockmodel(batch.to(device))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % images['classes'][predicted[j]]
                                for j in range(len(labels))))

    print("Accuracy: ", accuracy(blockmodel, images['val']))

    print("Test Data")
    batch, labels = next(iter(images['test']))
    imshow(torchvision.utils.make_grid(batch))
    print('GroundTruth: ', ' '.join('%5s' % images['classes'][labels[j]] for j in range(len(labels))))

    outputs = blockmodel(batch.to(device))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % images['classes'][predicted[j]]
                                for j in range(len(labels))))

    print("Accuracy: ", accuracy(blockmodel, images['test']))

def main():

    blockmodel = torch.load("BlockModel.pth")
    blockmodel.eval()
    board = None
    #verifymodel(blockmodel=blockmodel)
    print("Welcome to the blokus move visualizer!")
    print("This program will help you detect all the move options you have with a piece in a board configuration!\n")
    while(True):
        cmd = input("Command(type h for help): ")
        if(cmd == "h"):
            print("Commands available: [b: set board configuration], [p: check piece moves], [x: exit], [h: options]")
        elif(cmd == "p"):
            if(board is None):
                print("No current board configuration, set a board configuration with b first")
            else:
                pieces = None
                while(True):
                    piecePath = input("Path to folder holding your piece images: ")
                    try:
                        pieces = loadPieceImage(piecePath, models.ResNet50_Weights.DEFAULT.transforms())
                        break
                    except:
                        print("Invalid piece path, try again")

                cleanImageIter = iter(pieces['cleanpieces'])
                print("Going through all pieces")
                for batch, label in pieces['pieces']:
                    cleanBatch, cleanLabel = next(cleanImageIter)
                    imshow(torchvision.utils.make_grid(cleanBatch))
                    guess = blockmodel(batch.to(device))
                    trans = np.transpose(cleanBatch.numpy()[0], (1, 2, 0))
                    hsv = cv2.cvtColor(trans, cv2.COLOR_RGB2HSV)
                    color = None
                    for i in range(hsv.shape[0]):
                        for j in range(hsv.shape[1]):
                            h = hsv[i][j][0]
                            s = hsv[i][j][1]
                            v = hsv[i][j][2]
                            if(s > 0.6 and v > 0.8):
                                if(h < 60):
                                    color = "red"
                                elif(h < 120):
                                    color = "yellow"
                                elif(h < 180):
                                    color = "green"
                                elif(h < 300):
                                    color = "blue"
                                else:
                                    color = "red"
                                break
                        if color != None:
                            break
                        
                    _, predicted = torch.max(guess, 1)
                    pieceName = ''.join('%5s' % pieces['classes'][predicted[j]]
                                for j in range(len(label)))
                    pieceName = pieceName.strip()
                    while(True):
                        val = input("Is a " + color + " " + pieceName + " the piece in the picture? (y/n): ")
                        if(val == "y"):
                            print("Awesome, time to generate your options")
                            break
                        elif(val == "n"):
                            while(True):
                                newPiece = input("What is the actual piece name? (Refer to chart): ")
                                if(newPiece in pieces['classes']):
                                    pieceName = newPiece
                                    break
                                else:
                                    print("That's not a valid piece name")
                            break
                        else:
                            print("Not an option")
                    place_piece(color=color, piece=pieceName, boardPath=board)
                    moveOptions = cv2.imread("next_move.jpg")
                    rgbVer = cv2.cvtColor(moveOptions, cv2.COLOR_BGR2RGB)
                    plt.imshow(rgbVer)
                    plt.show()

                
                
        elif(cmd == "b"):
            while(True):
                boardpath = input("Path to board image: ")
                try:
                    load_image(boardpath)
                    board = boardpath
                    break
                except:
                    print("Invalid board path, try again")
        elif(cmd == "x"):
            break

    
    
    

    #place_piece("yellow", "I1")

main()



