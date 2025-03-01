# DocAssist AI - Medical Report Analysis System

<div align="center">
  <img src="ui%20visuals/docsvg.png" alt="DocAssist AI Logo" width="600"/>
  <br><br>
  <img src="visuals/enhancing_healthcare_with_docassist.png" alt="DocAssist System Architecture" width="800"/>
  <p><em>DocAssist AI System Architecture: Integrating Healthcare Support, Data Analysis, and Personalized Recommendations</em></p>
</div>

> **DocAssist AI** is a sophisticated medical report analysis system that leverages machine learning to analyze blood test reports and provide intelligent medical recommendations. The system can process both PDF reports and manually entered blood test values to deliver comprehensive medical insights.

## 🌟 Overview

DocAssist AI is an advanced healthcare analytics platform designed to revolutionize the way medical professionals and healthcare providers analyze and interpret blood test reports. The system combines cutting-edge machine learning with medical expertise to:

🔗 **Model Repository**: [DocAssist Model](https://github.com/realranjan/DOCASSIST-MODEL) - Contains the machine learning models and training notebooks.

### Key Capabilities

- ⚡ **Automated Analysis**: Transform complex blood test reports into actionable insights within seconds
- 🧠 **Intelligent Diagnosis**: Detect patterns and anomalies in blood parameters using sophisticated ML algorithms
- 📋 **Comprehensive Reporting**: Generate detailed medical reports with parameter-wise analysis and recommendations
- 🔍 **Disease Pattern Recognition**: Identify potential health conditions based on blood parameter patterns
- 💊 **Treatment Guidelines**: Provide evidence-based treatment recommendations and monitoring protocols
- 📄 **PDF Processing**: Extract and analyze blood test values directly from PDF reports
- ⚡ **Real-time Processing**: Deliver instant analysis for manually entered blood test values
- 📑 **PDF Report Generation**: Generate and download professional medical reports in PDF format

### Target Users
The system is built with a focus on accuracy, reliability, and user experience, making it an invaluable tool for:
- 👨‍⚕️ **Medical Practitioners**
- 🏥 **Healthcare Facilities**
- 🔬 **Diagnostic Labs**
- 📊 **Medical Researchers**
- 🏃‍♂️ **Health & Wellness Centers**

<div align="center">
  <img src="ui%20visuals/Screenshot%202025-03-02%20at%2000-15-56%20DocAssist%20AI%20-%20Medical%20Report%20Analysis.png" alt="DocAssist AI Interface" width="800"/>
  <p><em>DocAssist AI Web Interface: Modern, Intuitive, and User-Friendly Design</em></p>
</div>

## ✨ Features

### Core Functionality
- 🔍 **PDF Report Analysis**
  - Automatically extract medical values from uploaded PDF reports
  - Intelligent parsing of various report formats
- 📊 **Manual Data Entry**
  - Input blood test values manually for instant analysis
  - Real-time validation and error checking

### Analysis & Insights
- 🏥 **Disease Pattern Detection**
  - Identify potential diseases based on blood parameter patterns
  - Advanced correlation analysis
- 📈 **Abnormal Value Detection**
  - Highlight and explain abnormal blood test results
  - Comparative analysis with reference ranges

### Recommendations & Output
- 💊 **Treatment Recommendations**
  - Provide detailed treatment plans
  - Evidence-based monitoring guidelines
- 📄 **PDF Report Generation**
  - Generate professional medical reports
  - Customizable report formats

### Technical Features
- 📱 **Modern UI/UX**
  - Clean, responsive interface
  - Real-time updates and notifications
- 🔒 **Secure Processing**
  - Local processing of medical data
  - No external storage of sensitive information

## 🛠️ Tech Stack

### Frontend
- **Core Technologies**
  - HTML5/CSS3/JavaScript
  - Modern UI components with shadcn-inspired styling
- **Features**
  - Responsive design for all devices
  - Chart.js for data visualization

### Backend
- **Core Framework**
  - Python 3.8+
  - Flask for API server
- **Data Processing**
  - PyPDF2 for PDF processing
  - NumPy/Pandas for data analysis
- **Machine Learning**
  - Scikit-learn for ML predictions
- **Report Generation**
  - pdfkit for PDF generation
  - wkhtmltopdf for HTML to PDF conversion

## ⚙️ Prerequisites

Before running the application, ensure you have the following installed:
- ✅ Python 3.8 or higher
- ✅ pip (Python package manager)
- ✅ Git
- ✅ wkhtmltopdf (Required for PDF generation)

### Installing wkhtmltopdf

<details>
<summary>Click to expand installation instructions</summary>

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
</details>

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/realranjan/DOCASSIST-AI.git
   cd DOCASSIST-AI
   ```

2. **Set up the Python virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

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
├── notebooks/             # Jupyter notebooks for analysis
├── ui visuals/            # Web app interface visuals
├── visuals/              # Project visuals and diagrams
│
├── README.md             # Project documentation
├── LICENSE               # License information
├── .gitignore           # Git ignore rules
├── .gitattributes       # Git attributes
└── render.yaml          # Deployment configuration
```

## 👥 Authors

- **Ranjan Vernekar** - Initial work - [GitHub](https://github.com/realranjan) | [LinkedIn](https://www.linkedin.com/in/ranjan-vernekar-a93b08252/)

---
<div align="center">
  <p>Made with ❤️ by the DocAssist AI Team</p>
  <p><strong>© 2025 DocAssist AI. All rights reserved.</strong></p>
</div>
