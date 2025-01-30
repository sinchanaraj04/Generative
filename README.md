GAN-Based Art Generation System

This project focuses on leveraging Generative Adversarial Networks (GANs) to create aesthetically pleasing and realistic artworks from random latent vectors. The system integrates a user-friendly interface for seamless interaction, making it accessible to both technical and non-technical users.

Features

Art Generation: Generate stunning artworks using a trained GAN model.

Image Upload: Users can upload images to influence the generated art.

Intuitive UI: Designed with simplicity in mind for easy use by anyone.

Evaluation: Compare generated art with real-world artworks based on visual appeal, diversity, and realism.

Multi-Platform Deployment: Available as a desktop application (via Tkinter) and a web-based platform (via Flask).

Technologies Used

Frameworks: PyTorch, Flask, Tkinter

Languages: Python

Dataset: Abstract Art Gallery Dataset

GAN Model Architecture:

Generator: Series of ConvTranspose2d layers with BatchNorm2d and ReLU activations

Output: 64x64 RGB images

System Architecture

Input:

Random latent vector (z) of size 100

Optional image upload

Processing:

GAN model transforms the latent vector into an artwork.

Outputs are evaluated against real-world art.

Output:

Generated 64x64 RGB images accessible via the interface.

Setup Instructions

1. Clone the Repository

git clone https://github.com/your-username/gan-art-generation.git
cd gan-art-generation

2. Install Dependencies

Ensure you have Python 3.8+ installed. Then run:

pip install -r requirements.txt

3. Run the Application

For Desktop (Tkinter):

python desktop_app.py

For Web (Flask):

python app.py

Visit http://127.0.0.1:5000 in your browser.

Usage

Generate Artwork:

For random art, click "Generate".

Upload Image:

Upload an image to create art influenced by it.

Download Art:

Save your favorite generated artwork locally.

Folder Structure

├── data/                   # Dataset for training
├── models/                 # Pre-trained GAN models
├── static/                 # Frontend assets (CSS, JS, images)
├── templates/              # HTML templates for Flask
├── app.py                  # Flask web application
├── desktop_app.py          # Tkinter desktop application
├── train_gan.py            # GAN training script
├── README.md               # Project documentation
└── requirements.txt        # Required Python libraries

Future Enhancements

Increase resolution of generated images.

Add support for custom style inputs.

Deploy the system on cloud platforms for broader accessibility.

Contributors

Adwika Thanmaya D

Sinchana B Raj (Lia)

Vatsala D V

License

This project is licensed under the MIT License.

Acknowledgements

Special thanks to the creators of the Abstract Art Gallery Dataset.

Inspired by research in generative art and GAN advancements.

