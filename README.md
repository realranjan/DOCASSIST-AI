# DocAssist AI - Medical Report Analysis System

<div align="center">
  <img src="visuals/enhancing_healthcare_with_docassist.png" alt="DocAssist System Architecture" width="800"/>
  <p><em>DocAssist AI System Architecture: Integrating Healthcare Support, Data Analysis, and Personalized Recommendations</em></p>
</div>

DocAssist AI is a sophisticated medical report analysis system that leverages machine learning to analyze blood test reports and provide intelligent medical recommendations. The system can process both PDF reports and manually entered blood test values to deliver comprehensive medical insights.

## Overview

DocAssist AI is an advanced healthcare analytics platform designed to revolutionize the way medical professionals and healthcare providers analyze and interpret blood test reports. The system combines cutting-edge machine learning with medical expertise to:

- **Automated Analysis**: Transform complex blood test reports into actionable insights within seconds
- **Intelligent Diagnosis**: Detect patterns and anomalies in blood parameters using sophisticated ML algorithms
- **Comprehensive Reporting**: Generate detailed medical reports with parameter-wise analysis and recommendations
- **Disease Pattern Recognition**: Identify potential health conditions based on blood parameter patterns
- **Treatment Guidelines**: Provide evidence-based treatment recommendations and monitoring protocols
- **PDF Processing**: Extract and analyze blood test values directly from PDF reports
- **Real-time Processing**: Deliver instant analysis for manually entered blood test values
- **PDF Report Generation**: Generate and download professional medical reports in PDF format

The system is built with a focus on accuracy, reliability, and user experience, making it an invaluable tool for:
- 👨‍⚕️ Medical Practitioners
- 🏥 Healthcare Facilities
- 🔬 Diagnostic Labs
- 📊 Medical Researchers
- 🏃‍♂️ Health & Wellness Centers

## Features

- 🔍 **PDF Report Analysis**: Automatically extract medical values from uploaded PDF reports
- 📊 **Manual Data Entry**: Input blood test values manually for instant analysis
- 🏥 **Disease Pattern Detection**: Identify potential diseases based on blood parameter patterns
- 📈 **Abnormal Value Detection**: Highlight and explain abnormal blood test results
- 💊 **Treatment Recommendations**: Provide detailed treatment plans and monitoring guidelines
- 📱 **Modern UI/UX**: Clean, responsive interface with real-time updates
- 🔒 **Secure Processing**: Local processing of medical data with no external storage
- 📄 **PDF Report Generation**: Generate and download professional medical reports in PDF format

## Tech Stack

### Frontend
- HTML5/CSS3/JavaScript
- Modern UI components with shadcn-inspired styling
- Responsive design for all devices
- Chart.js for data visualization

### Backend
- Python 3.8+
- Flask for API server
- PyPDF2 for PDF processing
- NumPy/Pandas for data processing
- Scikit-learn for ML predictions
- pdfkit for PDF report generation
- wkhtmltopdf for HTML to PDF conversion

## Prerequisites

Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git
- wkhtmltopdf (Required for PDF generation)

### Installing wkhtmltopdf

#### Windows
```bash
winget install wkhtmltopdf.wkhtmltox
```

#### macOS
```bash
brew install wkhtmltopdf
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install wkhtmltopdf
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/realranjan/DOCASSIST-AI.git
   cd DOCASSIST-AI
   ```

2. Set up the Python virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
DOCASSIST-AI/
├── backend/                 # Backend Flask application
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # ML model files
│   └── uploads/            # Temporary PDF upload directory
│
├── frontend/               # Frontend application
│   ├── public/            # Static files
│   ├── server.js          # Frontend server
│   ├── config.js          # Configuration
│   └── package.json       # Node.js dependencies
│
├── data/                  # Dataset and data processing
├── notebooks/            # Jupyter notebooks for analysis
├── visuals/             # Project visuals and diagrams
│
├── README.md            # Project documentation
├── LICENSE              # License information
├── .gitignore          # Git ignore rules
├── .gitattributes      # Git attributes
└── render.yaml         # Deployment configuration
```

## Running the Application

1. Start the Flask backend server:
   ```bash
   cd backend
   python app.py
   ```
   The backend server will start on `http://localhost:5000`

2. Open the frontend:
   - Navigate to the `frontend` directory
   - Open `index.html` in your web browser
   - For the best experience, use a modern web browser (Chrome, Firefox, Edge)

## Usage

1. **PDF Analysis**:
   - Click the "Upload PDF" button
   - Select a blood test report PDF
   - Wait for the analysis results

2. **Manual Entry**:
   - Navigate to the "Manual Entry" tab
   - Fill in the blood test parameters
   - Click "Analyze" for instant results

3. **View Results**:
   - Review the comprehensive medical report
   - Check abnormal values and their implications
   - Review disease patterns if detected
   - Follow recommended treatments and monitoring plans

4. **Download Report**:
   - After analysis, click "Download Report as PDF"
   - The report will be downloaded as a professional PDF document
   - The PDF includes all analysis results, recommendations, and visualizations

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## API Documentation

The backend provides several REST endpoints:

- `POST /predict/file`: Analyze PDF reports
- `POST /predict`: Process manual entries
- `GET /api/dashboard-data`: Get dashboard statistics
- `POST /generate-pdf`: Generate and download PDF reports

For detailed API documentation, refer to the [API Documentation](docs/API.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Medical reference ranges and disease patterns are based on standard medical guidelines
- UI design inspired by modern healthcare applications
- Special thanks to all contributors and the medical professionals who provided domain expertise

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Authors

- Ranjan Vernekar - Initial work - [realranjan](https://github.com/realranjan)

---
Made with ❤️ by the DocAssist AI Team 