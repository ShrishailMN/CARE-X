# X-Ray Analysis System

A Flask-based web application for analyzing chest X-rays using deep learning.

## Features

- X-ray image analysis using DenseNet121
- Detailed medical report generation
- PDF report generation with images
- Patient history tracking
- Analytics dashboard
- Multi-language support

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd xray-analyzer
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

4. Create necessary directories:
```bash
mkdir -p static/uploads static/reports static/heatmaps
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment on Render

1. Push your code to GitHub

2. Create a new Web Service on Render:
   - Connect your GitHub repository
   - Select Python as the environment
   - Set the following:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
     - Python Version: 3.12.0

3. Add the following environment variables:
   - `PYTHON_VERSION`: 3.12.0
   - `PORT`: 10000

4. Deploy!

## Project Structure

```
xray-analyzer/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── render.yaml         # Render configuration
├── static/            # Static files
│   ├── uploads/       # Uploaded X-ray images
│   ├── reports/       # Generated PDF reports
│   └── heatmaps/      # Generated heatmaps
└── templates/         # HTML templates
```

## Notes

- The application uses SQLite for data storage
- Generated files (uploads, reports, heatmaps) are stored in the static directory
- Make sure to set appropriate file permissions for the static directories 