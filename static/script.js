document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const uploadForm = document.getElementById('upload-form');
    const listImagesBtn = document.getElementById('list-images-btn');
    const generatedImageDiv = document.getElementById('generated-image');
    const enhancedImageDiv = document.getElementById('enhanced-image');
    const imagesListDiv = document.getElementById('images-list');

    // Generate Image
    generateBtn.addEventListener('click', async () => {
        generatedImageDiv.innerHTML = "Generating...";
        const response = await fetch('/generate', { method: 'POST' });
        const data = await response.json();
        if (data.status === 'success') {
            generatedImageDiv.innerHTML = `<img src="${data.image_url}" alt="Generated Image">`;
        } else {
            generatedImageDiv.innerHTML = `Error: ${data.message}`;
        }
    });

    // Upload and Enhance Image
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        enhancedImageDiv.innerHTML = "Enhancing...";
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.status === 'success') {
            enhancedImageDiv.innerHTML = `<img src="${data.image_url}" alt="Enhanced Image">`;
        } else {
            enhancedImageDiv.innerHTML = `Error: ${data.message}`;
        }
    });

    // List Generated Images
    listImagesBtn.addEventListener('click', async () => {
        imagesListDiv.innerHTML = "Loading...";
        const response = await fetch('/list-images', { method: 'GET' });
        const data = await response.json();
        if (data.images) {
            imagesListDiv.innerHTML = data.images
                .map(img => `<img src="/generated/${img}" alt="${img}">`)
                .join('');
        } else {
            imagesListDiv.innerHTML = "No images found.";
        }
    });
});
