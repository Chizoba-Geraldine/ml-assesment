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
from sklearn.metrics import accuracy_score

def mean_reciprocal_rank(y_true, y_prob):
    ranks = []
    for true_label, probs in zip(y_true, y_prob):
        sorted_indices = np.argsort(probs)[::-1]
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        ranks.append(1.0 / rank)
    return np.mean(ranks)

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_data()

    def load_data(self):
        with open(self.data_file) as f:
            lines = f.readlines()
            if self.data_split == "train":
                random.shuffle(lines)
                lines = lines[:int(len(lines) * self.train_ratio)]

            for line in lines:
                img_name, text, raw_label = line.rstrip("\n").split("\t")
                img_path = os.path.join(self.images_path, img_name.strip())
                question_text, answer_text = text.split("?")
                question_text += '?'

                label = 1 if raw_label == "match" else 0

                self.image_data.append(img_path)
                self.question_data.append(question_text.strip())
                self.answer_data.append(answer_text.strip())
                self.question_embeddings_data.append(self.sentence_embeddings[question_text.strip()])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text.strip()])
                self.label_data.append(label)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = Image.open(self.image_data[idx]).convert("RGB")
        img = self.transform(img)
        q_embed = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        a_embed = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, q_embed, a_embed, label

def load_sentence_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.vision_model = vit_b_32(weights="IMAGENET1K_V1" if pretrained else None)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in list(self.vision_model.heads.parameters())[-2:]:
            param.requires_grad = True
        self.num_features = self.vision_model.heads[0].in_features
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        return self.vision_model(x)

class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        super().__init__()
        if ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in list(self.vision_model.children())[-2:]:
                for p in param.parameters():
                    p.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)
        elif ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)

        self.question_embedding_layer = nn.Linear(768, 128)
        self.answer_embedding_layer = nn.Linear(768, 128)
        self.fc = nn.Linear(128 * 3, num_classes)
        self.ARCHITECTURE = ARCHITECTURE

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            img_features = self.fc_vit(img_features)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined = torch.cat((img_features, question_features, answer_features), dim=1)
        return self.fc(combined)

def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0

        for batch_idx, (images, q_emb, a_emb, labels) in enumerate(train_loader):
            images, q_emb, a_emb, labels = images.to(device), q_emb.to(device), a_emb.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(images, q_emb, a_emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {running_loss / len(train_loader):.4f}, {time.time() - start_time:.2f} seconds")

def evaluate_model(model, ARCHITECTURE, test_loader, device):
    model.eval()
    all_labels, all_predictions, all_probs = [], [], []
    total_test_loss = 0.0
    start_time = time.time()

    with torch.no_grad():
        for images, q_emb, a_emb, labels in test_loader:
            images, q_emb, a_emb, labels = images.to(device), q_emb.to(device), a_emb.to(device), labels.to(device)
            outputs = model(images, q_emb, a_emb)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds)
            all_probs.extend(probs)

    accuracy = accuracy_score(all_labels, all_predictions)
    mrr = mean_reciprocal_rank(all_labels, all_probs)
    elapsed_time = time.time() - start_time

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    print(f"Test Loss: {total_test_loss:.4f}, Time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGES_PATH = "/workspaces/ml-assesment/Task 1/ITM_Classifier-baselines/visual7w-images"
    train_data_file = "/workspaces/ml-assesment/Task 1/ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
    dev_data_file = "/workspaces/ml-assesment/Task 1/ITM_Classifier-baselines/visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "/workspaces/ml-assesment/Task 1/ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "/workspaces/ml-assesment/Task 1/ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"

    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=120, shuffle=False)

    model = ITM_Model(num_classes=2, ARCHITECTURE="ViT", PRETRAINED=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train_model(model, "ViT", train_loader, criterion, optimiser, num_epochs=10)
    evaluate_model(model, "ViT", test_loader, device)




