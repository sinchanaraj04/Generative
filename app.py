import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import uuid
import torch.nn as nn
import random

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = './models'
GENERATED_FOLDER = './generated'
os.makedirs(GENERATED_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generator Architecture
latent_size = 128
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Load model weights
try:
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'G.pth'), map_location=device))
    generator.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load generator model: {e}")

# Transformations for uploaded image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as per your model's requirement
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Generate random noise for image generation
        num_images = int(request.form.get('num_images', 1))  # Get number of images to generate
        image_urls = []
        
        for _ in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            generated_image = generator(noise).detach().cpu()
            
            # Save the generated image
            image_filename = f'{uuid.uuid4().hex}.png'
            image_path = os.path.join(GENERATED_FOLDER, image_filename)
            save_image(generated_image, image_path)
            image_urls.append(f'/generated/{image_filename}')
        
        return jsonify({'status': 'success', 'image_urls': image_urls})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/list-images', methods=['GET'])
def list_images():
    try:
        images = os.listdir(GENERATED_FOLDER)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/show-random-image', methods=['GET'])
def show_random_image():
    try:
        images = os.listdir(GENERATED_FOLDER)
        random_image = random.choice(images)
        return send_from_directory(GENERATED_FOLDER, random_image)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_and_enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['file']
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Enhance image using the generator (in place of SRGAN or another method)
        enhanced_image = generator(image_tensor).detach().cpu()

        # Save enhanced image
        image_filename = f'enhanced_{uuid.uuid4().hex}.png'
        image_path = os.path.join(GENERATED_FOLDER, image_filename)
        save_image(enhanced_image, image_path)

        return jsonify({'status': 'success', 'image_url': f'/generated/{image_filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
