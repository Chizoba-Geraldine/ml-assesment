################################################################################
# Image-Text Matching Classifier: LoRA-based system for visual question answering
# 
# This program has extended from the CMP9137 workshop materials of week 5, 2025.
# It is powered the ViT and BLIP models. Here their links:
# 
# info of vit_b_32: https://pytorch.org/vision/main/models/vision_transformer.html
# info of BLIP: https://huggingface.co/docs/transformers/model_doc/blip
# 
# If you require training/test data, please get them from the materials of week 5.
#
# Version 1.0, main functionality in tensorflow tested with COCO data 
# Version 1.2, extended functionality for Flickr data
# Version 1.3, ported to pytorch and tested with visual7w data
# Version 1.4, support for Low-Rank Adaptation (LoRA) of Vit and BLIP models
# Version 1.5, added F1 score, MRR, accuracy and timing metrics
# Contact: {hcuayahuitl}@lincoln.ac.uk
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
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics import f1_score

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

            # Apply train_ratio only for training data
            if self.data_split == "train":
                random.shuffle(lines)  # Shuffle before selecting
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]

            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")  
                img_path = os.path.join(self.images_path, img_name.strip())

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()

                # Get binary labels from match/no-match answers
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

# Load sentence embeddings from an existing file -- generated a priori
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Enabler of Low-Rank Adaptation (LoRA) in image-based Transformers
class LoRALinear(nn.Module):
    def __init__(self, original_layer, r, alpha):
        super().__init__()
        self.original_layer = original_layer  # original linear layer
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

# Image-Text Matching Model
class ITM_LoRA_Model(nn.Module):
    def __init__(self, num_classes, ARCHITECTURE):
        print(f'BUILDING %s model' % (ARCHITECTURE))
        super(ITM_LoRA_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE
        self.r = 16# controls the adaptation complexity, higher r more weights to learn (more expressive adaptation)
        self.alpha=32 # controls the influence of LoRA updates, higher alpha stronger adaptation (more influence of new weights)
  
        if self.ARCHITECTURE == "ViT-LoRA":
            vit_model = models.vit_b_32(weights="IMAGENET1K_V1")
            vit_model.heads.head = nn.Identity()  # remove classifier head

            vit_model  = self.apply_lora_to_vit(vit_model)
            self.vision_model = vit_model 

            self.fc_vit = nn.Linear(vit_model.hidden_dim, 128)  # reduce feature size

        elif self.ARCHITECTURE == "BLIP-LoRA":
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            print("\nFULL model:\n",self.vision_model)
            self.vision_model = self.vision_model.vision_model
            print("\nONLY vision model:\n",self.vision_model)
            self.vision_model.to(torch.bfloat16)

            self.model = self.apply_lora_to_blip(self.vision_model)
            
            self.fc_blip = nn.Linear(self.vision_model.config.hidden_size, 128) # reduce feature size

        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)

        self.question_embedding_layer = nn.Linear(768, 128)  # question layer 
        self.answer_embedding_layer = nn.Linear(768, 128)  # answer layer
        self.fc = nn.Linear(128 + 128 + 128, num_classes)  # vision and text features

    def forward(self, img, question_embedding, answer_embedding):
        if self.ARCHITECTURE == "ViT-LoRA":
            img_features = self.vision_model(img)
            img_features = self.fc_vit(img_features) # custom linear layer for ViT

        elif self.ARCHITECTURE == "BLIP-LoRA":
            img_features = self.vision_model(img).last_hidden_state[:, 0]  # last hidden state
            img_features = self.fc_blip(img_features) # custom linear layer for BLIP

        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)

        output = self.fc(combined_features)
        return output
    
    def apply_lora_to_vit(self, vit_model):
        # Freezes all ViT parameters/weights
        for param in vit_model.parameters():
            param.requires_grad = False
        
        # Applies LoRA to all transformer blocks
        for block in vit_model.encoder.layers:  
            block.self_attention.out_proj = LoRALinear(block.self_attention.out_proj, self.r, self.alpha)
        return vit_model

    def apply_lora_to_blip(self, blip_model):
        # Freeze all BLIP parameters
        for param in blip_model.parameters():
            param.requires_grad = False

        # Identify all nn.Linear layers in the BLIP model
        layers_to_modify = []
        for name, module in blip_model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_modify.append((name, module))
        
        # Apply LoRA to each identified layer
        for layer_name, original_layer in layers_to_modify:
            layer = self.get_layer_by_name(blip_model, layer_name)
            if layer is not None:
                print(f"Applying LoRA to layer {layer_name}")
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

def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING %s model' % (ARCHITECTURE))
    model.train()
    total_train_time = 0
    start_train = time.time()
    
    # Track the overall loss for each epoch
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        all_labels = []
        all_predictions = []

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            # Move images/text/labels to the GPU (if available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)

            # Forward pass -- given input data to the model
            outputs = model(images, question_embeddings, answer_embeddings)

            # Calculate loss (error)
            loss = criterion(outputs, labels)  # output should be raw logits
            
            # Backward pass -- given loss above
            optimiser.zero_grad() # clear the gradients
            loss.backward() # computes gradient of the loss/error
            optimiser.step() # updates parameters using gradients
            running_loss += loss.item()

            # Get predictions for metrics
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Print progress every X batches
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        avg_loss = running_loss / total_batches
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        f1 = f1_score(all_labels, all_predictions)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        print(f'Epoch Time: {epoch_time:.2f} seconds')
    
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
            # Move images/text/labels to the GPU (if available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)  # Labels are single integers (0 or 1)
            
            # Perform forward pass on our data
            outputs = model(images, question_embeddings, answer_embeddings)
            
            # Accumulate loss on test data
            total_test_loss += criterion(outputs, labels).item()  

            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # Store results for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate evaluation metrics
    elapsed_time = time.time() - start_time
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    f1 = f1_score(all_labels, all_predictions)
    
    # Calculate MRR (Mean Reciprocal Rank)
    # For binary classification, MRR is equivalent to accuracy for the top prediction
    mrr = accuracy
    
    # Calculate confusion matrix components
    tp = np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 1))
    tn = np.sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 0))
    fp = np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))
    fn = np.sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 1))
    
    # Calculate precision, recall, specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\nTEST METRICS:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall/Sensitivity: {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Mean Reciprocal Rank: {mrr:.4f}')
    print(f'Test Loss: {total_test_loss:.4f}')
    print(f'Testing Time: {elapsed_time:.2f} seconds')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'mrr': mrr,
        'test_loss': total_test_loss,
        'test_time': elapsed_time
    }

# Main Execution
if __name__ == '__main__':
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Paths and files
    IMAGES_PATH = "visual7w-images"
    train_data_file = "visual7w-text/v7w.TrainImages.itm.txt" 
    dev_data_file = "visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    # Create datasets and loaders
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")  # whole test data
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create the model using one of the two supported architectures
    MODEL_ARCHITECTURE = "ViT-LoRA" # options are "ViT-LoRA" or "BLIP-LoRA"
    model = ITM_LoRA_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE).to(device)
    print("\nModel Architecture:")
    print(model)

    # Print the parameters of the model selected above
    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:  # print trainable parameters
            num_params = param.numel()  
            total_params += num_params
            print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
    print(f"\nTotal number of parameters in the model: {total_params}")

    # Define loss function and optimiser 
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    # Train and evaluate the model
    train_time = train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=40)
    test_metrics = evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print(f"Model Architecture: {MODEL_ARCHITECTURE}")
    print(f"Total Training Time: {train_time:.2f} seconds")
    print(f"Total Testing Time: {test_metrics['test_time']:.2f} seconds")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Test MRR: {test_metrics['mrr']:.4f}")