# mlp-timeseries-interpolation-extrapolation


<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="assets/images/mlp_ts_in_ex.png" alt="Image 1" style="width: 100%; margin: 10px;">
</div>

## 📜 mlp-timeseries-interpolation-extrapolation
#### 📌 Summary 
Implemented MLP model A bivariate timeseries dataset cleaned, interpolated &amp; extrapolated using MLP, Fourier-Features &amp; PyTorch.

#### 🧠 Overview
This project implements a variation of the Generative Adversarial Network (GAN) called Deep Convolutional GAN (DCGAN) to generate synthetic human faces using Python and TensorFlow. The DCGAN architecture consists of two competing neural networks:
- **(1) The Generator     :** Creates fake face images from random noise.
- **(2) The Discriminator :** Tries to distinguish between real and fake images.

Both networks improve through adversarial training: the generator gets better at mimicking real faces, and the discriminator becomes more skilled at detecting fakes. This dynamic pushes the generator to produce increasingly realistic outputs. To maintain balance during training, the discriminator is deliberately kept simpler to avoid overpowering the generator.

The model is trained on the [Flickr Faces Dataset Resized](https://www.kaggle.com/datasets/matheuseduardo/flickr-faces-dataset-resized/data) from kaggle, which includes 52,000 face images in three resolutions: 64x64, 128x128, and 256x256. Due to GPU memory limitations, 64x64 resolution is recommended for training on the full dataset. However, this project experimented with 256x256 resolution on a smaller subset of the dataset to generate higher-quality outputs.  **All development and experimentation were carried out on Kaggle**, leveraging its GPU resources. This project also shows how input resolution impacts the quality of generated faces—higher-resolution training images lead to sharper, more realistic outputs, while lower resolutions introduce some blur and noise.

**DCGAN Model Architecture Summary**
- Two Neural Networks:
    1. Generator ("Artist") – creates realistic-looking images.
    2. Discriminator ("Critic") – distinguishes real images from fakes.
During training, both networks improve in opposition until the discriminator can no longer tell real from fake images.

**Generator Architecture**
- Structure:
    1. Fully connected (Dense) layer
    2. Transposed Convolution layers (upsampling)
    3. Final Output Layer

- Workflow:
    1. Starts with a latent vector (8x8x512).
    2. Upsamples through transposed convolutions:
         - 8×8 → 16×16 (256 channels) --> 16×16 → 32×32 (128) -->
         - 32×32 → 64×64 (64) --> 64×64 → 128×128 (32) --> 128×128 → 256×256 (16)
    3. Output layer: tanh activation to produce final 64×64×3 RGB image.

- Activations: ReLU for all layers (except final layer which uses tanh).

**Discriminator Architecture**
- Structure:
    1. Convolutional layers (reverse of generator)
    2. Flatten and Dropout
    3. Final Classification Layer
    4. 
- Workflow (mirrors Generator in reverse):
    1. 256×256 → 128×128 (16 channels) --> 128×128 → 64×64 (32) --> 64×64 → 32×32 (64)-->
    2. 32×32 → 16×16 (128) --> 16×16 → 8×8 (256)

- Activations: LeakyReLU (after BatchNorm), except output layer.

**Loss Functions**
- Binary Crossentropy used for both models.

- Discriminator Loss:
    - Measures accuracy of distinguishing real vs. fake (real → 1, fake → 0).
- Generator Loss:
    - Measures success at fooling the discriminator (fake → 1).

**Optimization**
- Both models use Adam Optimizer independently for training.

The project successfully demonstrates the capability of DCGANs to generate human faces, although the realism of the output images largely depends on the scale of training and the size of the input dataset.


#### 🎯 Use Cases 
- Synthetic Face Generation
- Data Augmentation
- Anonymization
- Art & Creative Media
- AI Model Benchmarking
- Educational Purpose
- Testing Face Recognition Systems

#### 🟢 Project Status
- Current Version: V1.0

#### 📂 Repository Structure
```
paper-hbert-sarcasm-detection/
├── README.md
├── LICENSE
├── .gitignore                  
├── assets/                      
│   └── images/
└── notebooks/               
    └── sarcasm-analysis.ipynb

```
### ✨ Features
- ✅ `DCGAN` model class
- ✅ Preprocessed Data
- ✅ Evaluation: Visualization of generated synthetif human face output 

🛠️ In progress:-
- On-going training with 256x256 images with more training epochs

<!--
### 🎥 Demo
 <a href="https://www.youtube.com/shorts/wexIv6X45eE?feature=share" target="_blank">
  <img src="assets/images/2_2.JPG" alt="YouTube Video" width="390" height="270">
</a> 
-->

### 🚀 Getting Started
#### 📚 Knowledge & Skills Required 
- Python programming
- ML/DL fundamentals, Neural Network Arhitecutres (CNN, GAN)
- Optimizers, Loss Functions
  
#### 💻 Software Requirements
- IDE (VS Code) or jupyter notebook or google colab, kaggle
- **Best run on Kaggle using GPU T4x2**
  
#### 🛡️ Tech Stack
- Language: python
- NLP/ML: sklearn, pandas, numpy
- Deep Learning: tensorflow
- Visualization: matplotlib

#### 🔍 Modules Breakdown
<b> 📥 (1) Data-Preprocessing :</b> wh 
- Load dataset from kaggle
- Ensure RGB (convert from grayscale if needed)
- Resize images
- Convert to float dtype
- Normalize to (-1, 1) for GAN training
- Append processed images to dataset

<b> 🤖 (2) DCGAN :</b> 
**Two Neural Networks:**
    1. Generator ("Artist") – creates realistic-looking images.
         - Activations: ReLU for all layers (except final layer which uses tanh).
    3. Discriminator ("Critic") – distinguishes real images from fakes.
         - Activations: LeakyReLU (after BatchNorm), except output layer.

<b> 📉 (3) Loss Functions :</b> 
- Binary Crossentropy used for both models.
- Discriminator Loss: Measures accuracy of distinguishing real vs. fake (real → 1, fake → 0).
- Generator Loss: Measures success at fooling the discriminator (fake → 1).

<b> 📶 (4) Optimization :</b> 
- Both models use Adam Optimizer independently for training.

##### 📊 Evaluation
- Seeing the generated fake images
- Future work : auraccy, precision , recall , f1

#### ⚙️ Installation
```
git clone https://github.com/pointer2Alvee/dcgan-human-face-generation.git
cd dcgan-face-gen

# Recommended: Use a virtual environment
pip install -r requirements.txt
```

##### 🖇️ requirements.txt (core packages):
```
pandas
numpy
tensorflow
matplotlib
```

##### 💻 Running the App Locally
1. Open Repo in VS code / Kaggle (recommended)
2. Run Command
3. See accuracy

#### 📖 Usage
- Open VS Code / kaggle

### 🧪 Sample Topics Implemented
- ✅ DCGAN
- ✅ CNN, CONVOLUTION, POOLING
- ✅ GAN, OPTIMIZERS, LOSS FUNCTIONS

### 🧭 Roadmap
- [x] Implementation of DCGAN
- [x] Generation of Fake Human Faces
- [ ] Trained with 300+ epochs 

### 🤝 Contributing
Contributions are welcomed!
1. Fork the repo. 
2. Create a branch: ```git checkout -b feature/YourFeature```
3. Commit changes: ```git commit -m 'Add some feature'```
4. Push to branch: ```git push origin feature/YourFeature```
5. Open a Pull Request.

### 📜License
Distributed under the MIT License. See LICENSE.txt for more information.

### 🙏Acknowledgements
- Special thanks to the open-source community / youtube for tools and resources.
