# UnstructData: Advanced Unstructured Table Extraction from PDFs

UnstructData is a powerful Flask-based application designed to extract tables—including complex, borderless, or unstructured ones—from PDF files using a variety of advanced models and techniques. It is built for researchers, data scientists, and anyone who needs to accurately extract tables from challenging documents.

## Features

- **Multiple Extraction Engines**: Use state-of-the-art models including PDFPlumber, Camelot, Tabula, Unstructured VLM, Table Transformer, PyMuPDF, CascadeTabNet, and PDF2XML.
- **Handles Complex Tables**: Extracts tables even when they are borderless, merged, multi-line, or have irregular structures.
- **Model Selection**: Choose the extraction model that best fits your PDF structure and complexity.
- **Dynamic Page Range**: Extract tables from specific pages or across the whole document.
- **Enhanced Pre- and Post-Processing**: Automatic cleaning, validation, header detection, and formatting of extracted tables.
- **Fallback Strategies**: If a model fails to extract tables, the app tries clustering and other fallback strategies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mithgx/UnstructData.git
   cd UnstructData
   ```

2. **Install dependencies:**
   - It is recommended to use a virtual environment (e.g., `venv` or `conda`).
   - Install Python requirements:
     ```bash
     pip install -r requirements.txt
     ```
   - Some models require additional tools:
     - **CascadeTabNet**: Requires `mmdet`, `mmcv-full`, and `pytesseract`.
     - **pdf2xml**: Requires `pdftohtml` (part of the `poppler-utils` package).
     - **tabula**: Requires Java.

3. **Download model weights:**
   - For Table Transformer and CascadeTabNet, the required model weights will be automatically downloaded the first time they are used.
   - For CascadeTabNet, you may need to manually place the config and checkpoint files in the `CascadeTabNet/Config` directory.

4. **Start the application:**
   ```bash
   python app.py
   ```


## Usage

1. Open your browser and go to `your localhost URL where the app is`.
2. Upload a PDF file using the provided form.
3. Select the desired table extraction model from the dropdown menu:
    - **pdfplumber**: Fast, basic table extraction.
    - **camelot**: Good for lattice-based tables.
    - **tabula**: Java-based, works for many standard tables.
    - **table_transformer_advanced**: Microsoft Table Transformer, excels at complex layouts.
    - **unstructured**: Uses the Unstructured VLM for high-resolution and borderless tables.
    - **cascade_tabnet**: Deep learning model for robust detection.
    - **pdf2xml**: XML-based, good for preserving document structure.
    - **pymupdf**: PyMuPDF-based extraction for challenging layouts.
4. (Optional) Specify start and end pages for extraction.
5. Click "Process" to extract tables. Results will be displayed on the page.

## Example

![screenshot](docs/screenshot.png)

## Supported Models & Methods

| Model                     | Description                                                      | Strengths                                 |
|---------------------------|------------------------------------------------------------------|-------------------------------------------|
| pdfplumber                | Lightweight, fast PDF text and table extraction                  | Simple, quick, standard tables            |
| camelot                   | Lattice and stream-based extraction using OpenCV                 | Bordered and simple stream tables         |
| tabula                    | Java-based parser, good for many table types                     | Cross-platform, standard tables           |
| table_transformer_advanced| Microsoft Table Transformer Detection + Structure Recognition     | Complex, borderless, multi-line tables    |
| unstructured              | Unstructured VLM, high-res, OCR, clustering fallback             | Borderless, irregular, multi-format PDFs  |
| cascade_tabnet            | Deep learning, MMDetection-based table detection                 | Robust detection, difficult layouts       |
| pdf2xml                   | XML-based, preserves visual layout                               | Complex, multi-column documents           |
| pymupdf                   | Cluster-based, for irregular and borderless tables               | Challenging, unstructured tables          |

## Advanced Options

- **Image & OCR Enhancement**: The app automatically sharpens images and applies adaptive histogram equalization for better OCR and detection.
- **Header Detection**: Multi-row and spanned headers are detected and merged.
- **Financial Data Formatting**: Specific rules for currency, dates, and numbers.

## Troubleshooting

- **CascadeTabNet errors**: Make sure you have installed `mmdet`, `mmcv-full`, and `pytesseract`, and placed the correct config and checkpoint files.
- **pdf2xml errors**: Install `poppler-utils` (Linux: `sudo apt-get install poppler-utils`; macOS: `brew install poppler`).
- **tabula errors**: Ensure Java is installed and available in your PATH.
- **Out-of-memory**: For large files, reduce DPI or restrict page ranges.

## Contributing

Pull requests, feature suggestions, and bug reports are welcome! Please create an issue or submit a PR.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)
- [Unstructured.io](https://github.com/Unstructured-IO/unstructured)
- [Camelot](https://github.com/camelot-dev/camelot)
- [Tabula](https://github.com/tabulapdf/tabula)
- [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

---

For issues or questions, please contact [mithgx](https://github.com/mithgx).
