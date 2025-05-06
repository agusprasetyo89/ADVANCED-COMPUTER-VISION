# **Framework Deep Learning: Jembatan Menuju Kecerdasan Buatan**


Di era digital saat ini, kita sering mendengar istilah seperti Artificial Intelligence (AI), Machine Learning, dan Deep Learning. Teknologi ini berada di balik kemampuan mobil tanpa sopir, sistem rekomendasi Netflix, sampai chatbot seperti ChatGPT.

Tapi bagaimana cara semua itu bekerja?

Jawabannya ada pada Deep Learning, dan lebih spesifiknya lagi, Framework Deep Learning.
Apa Itu Deep Learning?
Bayangkan otak manusia yang belajar dari pengalaman. Deep Learning meniru cara kerja otak ini menggunakan struktur yang disebut neural network. Jaringan ini dilatih dengan data — misalnya, ribuan gambar kucing — hingga akhirnya bisa mengenali gambar kucing baru.

Namun, membangun jaringan ini dari nol itu rumit. Di sinilah framework deep learning berperan.
Framework Deep Learning: Seperti Alat Masak di Dapur
Framework Deep Learning adalah alat bantu atau “peralatan dapur” untuk para ilmuwan data dan pengembang. Mereka mempermudah proses membuat, melatih, dan menguji model kecerdasan buatan tanpa harus membuat semuanya dari nol.
Beberapa framework terkenal adalah:
1. TensorFlow – Dikembangkan oleh Google.
2. PyTorch – Dikembangkan oleh Facebook.
3. Keras – Antarmuka ramah pengguna untuk TensorFlow.
4. MXNet – Digunakan oleh Amazon.
Mengapa Framework Penting?
Tanpa framework, kita harus memprogram jaringan saraf dari awal — seperti membuat mobil sendiri dari besi mentah. Dengan framework, kita cukup "merakit" dengan komponen yang sudah tersedia.

Manfaatnya:
- Efisiensi waktu
- Kemudahan eksperimen
- Komunitas besar
Contoh Penggunaan PyTorch
Berikut contoh sederhana penggunaan PyTorch untuk mengenali angka tulisan tangan (dataset MNIST):
```
#	Instalasi
# Instalasi Library
!pip install torch torchvision matplotlib

# Import Modul
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Persiapan Dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Definisi Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Loss Function dan Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(5):
    running_loss = 0.0
    for images, labels in trainloader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# Evaluasi Model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Akurasi: {100 * correct / total:.2f}%")
```
Penutup
Framework Deep Learning adalah jembatan penting antara ide dan implementasi dalam dunia kecerdasan buatan. Mereka membantu para peneliti, mahasiswa, hingga perusahaan besar untuk membangun sistem cerdas dengan lebih mudah dan efisien.

Referensi
Hery, H., Haryani, C. A., Widjaja, A. E., Tarigan, R. E., & Aribowo, A. (2024).
_Unsupervised Learning for MNIST with Exploratory Data Analysis for Digit Recognition._
Naik, S., Pathare, P., Qureshi, M., Kalaswad, C., Joshi, A., & Paliwal, N. (2024).
_Recognizing Handwritten Digits on MNIST Dataset using KNN Algorithm._
Ghosh, S. (2022).
_Comparative Analysis of Boosting Algorithms Over MNIST Handwritten Digit Dataset._
