# X-Ray Image Analysis and Report Generator

A Flask web application that analyzes X-ray images and generates detailed medical reports using deep learning.

## Features

- X-ray image upload and analysis
- Automated medical report generation
- Heatmap visualization of detected abnormalities
- PDF report generation
- Historical report storage and viewing

## Tech Stack

- Python 3.11
- Flask
- PyTorch
- SQLite
- FPDF
- Gunicorn

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image_Description_Generator.git
cd Image_Description_Generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
Image_Description_Generator/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── Procfile             # Process file for deployment
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── dataset/            # Training dataset
└── checkpoints/        # Model checkpoints
```

## Deployment

This project is configured for deployment on Render.com. The free tier includes:
- 512 MB RAM
- 750 hours of runtime per month
- Automatic HTTPS
- Continuous deployment from GitHub

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 