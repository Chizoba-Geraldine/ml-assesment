################################################################################
# Image-Text Matching Classifier: LoRA-based system for visual question answering
# 
# Final Corrected Version - Ready to Run
# Version 1.7.1 - Optimized for Performance
################################################################################

import os
import time
import pickle
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics import f1_score, precision_score, recall_score

# Custom Dataset with corrected image dimensions
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
        
        # Corrected transforms with 224x224 size for ViT compatibility
        if self.data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        print("Class distribution - Match:", sum(self.label_data), "No-match:", len(self.label_data)-sum(self.label_data))
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

def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r, alpha):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Linear(original_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_layer.out_features, bias=False)
        self.scaling = alpha / r
        self._bias = original_layer.bias

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        output = self.original_layer(x) + self.scaling * self.lora_B(self.lora_A(x))
        if self._bias is not None:
            output += self._bias
        return output
    
    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self._bias

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class ITM_LoRA_Model(nn.Module):
    def __init__(self, num_classes, ARCHITECTURE):
        print(f'BUILDING %s model' % (ARCHITECTURE))
        super(ITM_LoRA_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE
        self.r = 8
        self.alpha = 16
  
        if self.ARCHITECTURE == "ViT-LoRA":
            vit_model = models.vit_b_32(weights="IMAGENET1K_V1")
            vit_model.heads.head = nn.Identity()
            vit_model = self.apply_lora_to_vit(vit_model)
            self.vision_model = vit_model 
            self.fc_vit = nn.Sequential(
                nn.Linear(vit_model.hidden_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )

        elif self.ARCHITECTURE == "BLIP-LoRA":
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = self.vision_model.vision_model
            self.vision_model.to(torch.bfloat16)
            self.model = self.apply_lora_to_blip(self.vision_model)
            self.fc_blip = nn.Sequential(
                nn.Linear(self.vision_model.config.hidden_size, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )

        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)

        self.question_embedding_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        self.answer_embedding_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # Enhanced final classifier
        self.fc = nn.Sequential(
            nn.Linear(256 + 256 + 256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img, question_embedding, answer_embedding):
        if self.ARCHITECTURE == "ViT-LoRA":
            img_features = self.vision_model(img)
            img_features = self.fc_vit(img_features)

        elif self.ARCHITECTURE == "BLIP-LoRA":
            img_features = self.vision_model(img).last_hidden_state[:, 0]
            img_features = self.fc_blip(img_features)

        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output
    
    def apply_lora_to_vit(self, vit_model):
        for param in vit_model.parameters():
            param.requires_grad = False
        
        for block in vit_model.encoder.layers:  
            block.self_attention.out_proj = LoRALinear(block.self_attention.out_proj, self.r, self.alpha)
        return vit_model

    def apply_lora_to_blip(self, blip_model):
        for param in blip_model.parameters():
            param.requires_grad = False

        layers_to_modify = []
        for name, module in blip_model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_modify.append((name, module))
        
        for layer_name, original_layer in layers_to_modify:
            layer = self.get_layer_by_name(blip_model, layer_name)
            if layer is not None:
                new_layer = LoRALinear(original_layer, r=self.r, alpha=self.alpha)
                self.set_layer_by_name(blip_model, layer_name, new_layer)

        return blip_model

    def get_layer_by_name(self, blip_model, name):
        layers = name.split('.')
        layer = blip_model
        for part in layers:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                return None
        return layer

    def set_layer_by_name(self, model, layer_name, new_layer):
        layers = layer_name.split('.')
        layer = model
        for name in layers[:-1]: 
            layer = getattr(layer, name)
        setattr(layer, layers[-1], new_layer)

def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, scheduler, num_epochs=40):
    print(f'TRAINING %s model' % (ARCHITECTURE))
    model.train()
    total_train_time = 0
    start_train = time.time()
    best_f1 = 0
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        all_labels = []
        all_predictions = []

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)

            optimiser.zero_grad()
            outputs = model(images, question_embeddings, answer_embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if batch_idx % 100 == 0:
                current_lr = optimiser.param_groups[0]['lr']
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        avg_loss = running_loss / total_batches
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        # Early stopping check
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                model.load_state_dict(torch.load('best_model.pth'))
                break
        
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] Summary:')
        print(f'Training Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Epoch Time: {epoch_time:.2f} seconds')
        print(f'Current LR: {optimiser.param_groups[0]["lr"]:.2e}\n')
    
    total_train_time = time.time() - start_train
    print(f'Total Training Time: {total_train_time:.2f} seconds')
    return total_train_time

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
            total_test_loss += criterion(outputs, labels).item()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    elapsed_time = time.time() - start_time
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    specificity = np.sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 0)) / np.sum(np.array(all_labels) == 0)
    
    print("\nTEST METRICS:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall/Sensitivity: {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Test Loss: {total_test_loss:.4f}')
    print(f'Testing Time: {elapsed_time:.2f} seconds')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'test_loss': total_test_loss,
        'test_time': elapsed_time
    }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    IMAGES_PATH = "visual7w-images"
    train_data_file = "visual7w-text/v7w.TrainImages.itm.txt" 
    test_data_file = "visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    
    # Create weighted sampler to handle class imbalance
    class_counts = np.bincount(train_dataset.label_data)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = class_weights[train_dataset.label_data]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    MODEL_ARCHITECTURE = "ViT-LoRA"
    model = ITM_LoRA_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE).to(device)
    
    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")

    # Enhanced training configuration
    criterion = FocalLoss(alpha=0.35, gamma=1.5)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimiser,
        base_lr=1e-5,
        max_lr=3e-5,
        step_size_up=500,
        cycle_momentum=False
    )

    train_time = train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, scheduler, num_epochs=40)
    test_metrics = evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
    
    print("\nFINAL SUMMARY:")
    print(f"Model Architecture: {MODEL_ARCHITECTURE}")
    print(f"Total Training Time: {train_time:.2f} seconds")
    print(f"Total Testing Time: {test_metrics['test_time']:.2f} seconds")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")