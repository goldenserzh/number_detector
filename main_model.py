import pandas as pd
import torch
import seaborn as sns
from keras.datasets import mnist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix, roc_auc_score
from conModels import ConvModel

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_t = torch.from_numpy(X_train).float().unsqueeze(1)
y_train_t = torch.from_numpy(y_train).long()

X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
y_test_t = torch.from_numpy(y_test).long()

X_train_t /= 255.0
X_test_t /= 255.0

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

num_epoche = 5
num_classes = 10
batch_size = 10
learing_rate = 0.001
  
model = ConvModel(verbose=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

total_steps = len(train_loader)
loss_list = []
accuracy_list = []

for epoch in range(num_epoche):
    for i, (images, labels) in enumerate(train_loader):

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        accuracy_list.append(accuracy)


        if model.verbose:
          if i % 100 == 0:
            print(f'Эпоха: [{epoch+1}/{num_epoche}], Шаг: [{i+1}/{total_steps}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
            
losses_test = []
for epoch in range(num_epoche):
  model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for images, labels in test_loader:
      outputs = model(images)
      loss = criterion(outputs, labels)
      val_loss += loss.item()
  avg_val_loss = val_loss / len(test_loader)
  losses_test.append(avg_val_loss)

y_predict_numbers = torch.argmax(outputs, dim=1)
y_pred_proba = torch.softmax(outputs, dim=1)

accuracy = accuracy_score(y_test, y_predict_numbers)
print(f"Accuracy: {accuracy}")
print()
multi_con_mat = multilabel_confusion_matrix(y_test, y_predict_numbers, labels=range(10))
for i in range(10):
  print(f"Для числа {i + 1}")
  print(multi_con_mat[i])
  print('-' * 20)

def micro_macro_metrics(y_true, y_pred):
  precision_macro = precision_score(y_true, y_pred, average = 'macro')
  recall_macro = recall_score(y_true, y_pred, average = 'macro')
  f1_macro = f1_score(y_true, y_pred, average = 'macro')

  precision_micro = precision_score(y_true, y_pred, average = 'micro')
  recall_micro = recall_score(y_true, y_pred, average = 'micro')
  f1_micro = f1_score(y_true, y_pred, average = 'micro')

  print('Макро метрики:')
  print()
  print(f'Precision: {precision_macro}')
  print(f'Recall: {recall_macro}')
  print(f'F1: {f1_macro}')
  print()
  print('Микро метрики:')
  print(f'Precision: {precision_micro}')
  print(f'Recall: {recall_micro}')
  print(f'F1: {f1_micro}')



micro_macro_metrics(y_test, y_predict_numbers)