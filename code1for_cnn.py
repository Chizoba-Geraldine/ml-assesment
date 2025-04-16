################################################################################
# Image-Text Matching Classifier: baseline system for visual question answering
# 
# This program has been adapted and rewriten from the CMP9137 materials of 2024.
# 
# It treats the task of multi-choice visual question answering as a binary
# classification task. This is possible by rewriting the questions from this format:
# v7w_2358727.jpg	When was this?  Nighttime. | Daytime. | Dawn. Sunset.
# 
# to the following format:
# v7w_2358727.jpg	When was this? Nighttime. 	match
# v7w_2358727.jpg	When was this?  Daytime. 	no-match
# v7w_2358727.jpg	When was this?  Dawn. 	no-match
# v7w_2358727.jpg	When was this?  Sunset.	no-match
#
# The list above contains the image file name, the question-answer pairs, and the labels.
# Only question types "when", "where" and "who" were used due to compute requirements. In
# this folder, files v7w.*Images.itm.txt are used and v7w.*Images.txt are ignored. The 
# two formats are provided for your information and convenience.
# 
# To enable the above this implementation provides the following classes and functions:
# - Class ITM_Dataset() to load the multimodal data (image & text (question and answer)).
# - Class Transformer_VisionEncoder() to create a pre-trained Vision Transformer, which
#   can be finetuned or trained from scratch -- update USE_PRETRAINED_MODEL accordingly.
# - Function load_sentence_embeddings() to load pre-generated sentence embeddings of questions 
#   and answers, which were generated using SentenceTransformer('sentence-transformers/gtr-t5-large').
# - Class ITM_Model() to create a model combining the vision and text encoders above. 
# - Function train_model trains/finetunes one of two possible models: CNN or ViT. The CNN 
#   model is based on resnet18, and the Vision Transformer (ViT) is based on vit_b_32.
# - Function evaluate_model() calculates the accuracy of the selected model using test data. 
# - The last block of code brings everything together calling all classes & functions above.
# 
# info of resnet18: https://pytorch.org/vision/main/models/resnet.html
# info of vit_b_32: https://pytorch.org/vision/main/models/vision_transformer.html
# info of SentenceTransformer: https://huggingface.co/sentence-transformers/gtr-t5-large
#
# This program was tested on Windows 11 using WSL and does not generate any plots. 
# Feel free to use and extend this program as part of your our assignment work.
#
# Version 1.0, main functionality in tensorflow tested with COCO data 
# Version 1.2, extended functionality for Flickr data
# Version 1.3, ported to pytorch and tested with visual7w data
# Contact: {hcuayahuitl}@lincoln.ac.uk
################################################################################

import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32  

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import label_ranking_average_precision_score


# Custom Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard for pretrained models on ImageNet
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from "+str(self.data_file))
        print("=========================================")

        random.seed(42)

        with open(self.data_file) as f:
            lines = f.readlines()

            if self.data_split == "train":
                random.shuffle(lines)
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]

            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")  
                img_path = os.path.join(self.images_path, img_name.strip())

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()

                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)

        print("|image_data|="+str(len(self.image_data)))
        print("|question_data|="+str(len(self.question_data)))
        print("|answer_data|="+str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, question_embedding, answer_embedding, label

# Load sentence embeddings
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Vision Transformer model
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()

        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
	
        self.num_features = self.vision_model.heads[0].in_features
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        features = self.vision_model(x)
        return features

# Combined Image-Text Matching Model
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING %s model, pretrained=%s' % (ARCHITECTURE, PRETRAINED))
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE

        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet50(pretrained=PRETRAINED)
            if PRETRAINED:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.vision_model.parameters():
                    param.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)

        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)

        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)

        self.question_embedding_layer = nn.Linear(768, 128)
        self.answer_embedding_layer = nn.Linear(768, 128)
        self.fc = nn.Linear(128 + 128 + 128, num_classes)

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            img_features = self.fc_vit(img_features)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=50):
    print(f'TRAINING %s model' % (ARCHITECTURE))
    model.train()

    total_training_start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time.time()

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)

            outputs = model(images, question_embeddings, answer_embeddings)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')

        avg_loss = running_loss / total_batches
        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}, {elapsed_time:.2f} seconds')

    total_training_end_time = time.time()
    print(f'\nTotal Training Time: {total_training_end_time - total_training_start_time:.2f} seconds\n')


def evaluate_model(model, ARCHITECTURE, test_loader, device):
    print(f'EVALUATING %s model' % (ARCHITECTURE))
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    start_time = time.time()

    with torch.no_grad():
        for images, question_embeddings, answer_embeddings, labels in test_loader:
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)

            outputs = model(images, question_embeddings, answer_embeddings)
            total_test_loss += criterion(outputs, labels)  

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1);
            

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

    true_ranks = np.zeros_like(all_probabilities)
    true_ranks[np.arange(len(all_labels)), all_labels] = 1
    mrr = label_ranking_average_precision_score(true_ranks, all_probabilities)

    elapsed_time = time.time() - start_time

    print(f'\nEvaluation Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Macro: {f1_macro:.4f}')
    print(f'F1 Weighted: {f1_weighted:.4f}')
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')
    print(f'Total Evaluation Time: {elapsed_time:.2f} seconds')
    print(f'Total Test Loss: {total_test_loss:.4f}')


# Main Execution
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    IMAGES_PATH = "./ITM_Classifier-baselines/visual7w-images"
    train_data_file = "./ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt"
    dev_data_file = "./ITM_Classifier-baselines/visual7w-text//v7w.DevImages.itm.txt"
    test_data_file = "./ITM_Classifier-baselines/visual7w-text//v7w.TestImages.itm.txt"
    sentence_embeddings_file = "./ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    MODEL_ARCHITECTURE = "CNN" # options are "CNN" or "ViT"
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    print("\nModel Architecture:")
    print(model)

    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()  
            total_params += num_params
            print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
    print(f"\nTotal number of parameters in the model: {total_params}")
    print(f"\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}\n")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=50)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)

