import os
from flask import Flask, request, render_template
import pdfplumber
import camelot
import tabula
from werkzeug.utils import secure_filename
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Text
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf_or_image
from unstructured.partition.image import partition_image
from unstructured.staging.base import elements_to_json
import numpy as np
from collections import defaultdict
import logging
import traceback
import re
from typing import List, Tuple, Dict, Union, Optional, Any
from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection, DetrImageProcessor
from pdf2image import convert_from_path
import torch
from torchvision.ops import nms
from PIL import Image
import io
import warnings
import math
import cv2
from itertools import groupby
from operator import itemgetter
import sys
import subprocess
from pathlib import Path
from PyPDF2 import PdfReader
from collections import OrderedDict
# pip install lxml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.module')
warnings.filterwarnings('ignore', message='The `max_size` parameter is deprecated')
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').propagate = False
logging.getLogger('transformers.configuration_utils').propagate = False

# Combined Table Transformer initialization
class CombinedTableTransformer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detection_model = None
        self.detection_processor = None
        self.structure_model = None
        self.structure_processor = None
        self._init_models()

    def _init_models(self):
        # Load detection model
        self.detection_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection",
            use_fast=True,
            do_resize=True,
            size={"height": 800, "width": 1333},
            do_pad=True
        )
        self.detection_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection",
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32
        ).to(self.device).eval()
        # Load structure model
        self.structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition",
            use_fast=True,
            do_resize=True,
            size={"shortest_edge": 800, "longest_edge": 1333},
            do_pad=True
        )
        self.structure_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition",
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32
        ).to(self.device).eval()

    def detect_tables(self, image):
        # Use detection model to get table boxes
        inputs = self.detection_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": 800, "width": 1333},
            do_pad=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        results = self.detection_processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[(image.size[1], image.size[0])]
        )[0]
        return results

    def extract_table_structure(self, image, box):
        # Crop the table region and use structure model
        x0, y0, x1, y1 = map(int, box)
        table_img = image.crop((x0, y0, x1, y1))
        inputs = self.structure_processor(
            images=table_img,
            return_tensors="pt",
            do_resize=True,
            size={"shortest_edge": 800, "longest_edge": 1333},
            do_pad=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        results = self.structure_processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[(table_img.size[1], table_img.size[0])]
        )[0]
        return results, table_img

combined_transformer = CombinedTableTransformer()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    tables = []
    uploaded_file = request.files.get('pdf_file')
    model = request.form.get('model')
    start_page = request.form.get('start_page')
    end_page = request.form.get('end_page')

    # Convert page numbers to integers if provided
    try:
        start_page = int(start_page) if start_page else None
        end_page = int(end_page) if end_page else None
    except ValueError:
        start_page = None
        end_page = None

    if uploaded_file and uploaded_file.filename.endswith('.pdf'):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)

        if model == 'table_transformer_advanced':
            tables = extract_tables_combined_transformer(filepath, start_page=start_page, end_page=end_page)
        elif model == 'pdfplumber':
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages[start_page-1 if start_page else 0:end_page if end_page else None], start=1):
                    words = page.extract_words()
                    tables_found = page.find_tables()
                    for tbl in tables_found:
                        table_data = tbl.extract()
                        if table_data:
                            top_y = tbl.bbox[1]
                            nearby_text = [
                                w['text'] for w in words
                                if w['top'] < top_y and abs(w['top'] - top_y) < 50
                            ]
                            title = " ".join(nearby_text).strip() or f"Table (Page {page_num})"
                            tables.append((title, table_data, [page_num]))
        elif model == 'camelot':
            try:
                pages = f"{start_page}-{end_page}" if start_page and end_page else 'all'
                camelot_tables = camelot.read_pdf(filepath, pages=pages)
                for idx, t in enumerate(camelot_tables):
                    title = f"Table {idx + 1} (Page {t.page})"
                    tables.append((title, t.df.values.tolist(), [t.page]))
            except Exception as e:
                tables.append(("Camelot Error", [[str(e)]], [1]))
        elif model == 'tabula':
            try:
                # Use the new text extraction function
                tables = extract_text_with_tabula(filepath, start_page=start_page, end_page=end_page)
            except Exception as e:
                tables.append(("Tabula Error", [[str(e)]], [1]))
        elif model == 'cascade_tabnet':
            tables = extract_tables_cascadetabnet(filepath, start_page=start_page, end_page=end_page)
        elif model == 'pdf2xml':
            try:
                tables = extract_tables_from_pdf2xml(filepath, start_page=start_page, end_page=end_page)
            except Exception as e:
                tables = [("PDF2XML Error", [[str(e)]], [1])]
        elif model == 'pymupdf':
            try:
                tables = extract_tables_pymupdf(filepath, start_page=start_page, end_page=end_page)
            except Exception as e:
                tables = [("PyMuPDF Error", [[str(e)]], [1])]

    return render_template('index.html', tables=tables)

import fitz  # PyMuPDF
from collections import defaultdict

import fitz  # PyMuPDF
import pprint

def clean_table_data(table_data):
    """
    Remove empty rows and columns from table data.
    Args:
        table_data: List of lists containing table cells
    Returns:
        Cleaned table data with empty rows and columns removed
    """
    if not table_data or not table_data[0]:
        return table_data

    # Remove empty rows
    table_data = [row for row in table_data if any(cell.strip() for cell in row)]
    
    if not table_data:
        return table_data

    # Find non-empty columns
    non_empty_cols = []
    for col_idx in range(len(table_data[0])):
        if any(row[col_idx].strip() for row in table_data):
            non_empty_cols.append(col_idx)
    
    # Keep only non-empty columns
    if non_empty_cols:
        table_data = [[row[col_idx] for col_idx in non_empty_cols] for row in table_data]

    return table_data

def extract_tables_pymupdf(pdf_path, y_tol=8, x_tol=6, min_cols=2, start_page=None, end_page=None):
    """
    Advanced extraction of complex, borderless, unstructured tables using PyMuPDF.
    Handles multi-line, merged, and irregular tables by clustering text blocks.
    """
    doc = fitz.open(pdf_path)
    all_tables = []
    total_pages = len(doc)
    start_page = max(0, (start_page or 1) - 1)
    end_page = min(total_pages, (end_page or total_pages))

    def calculate_dynamic_tolerances(blocks):
        # minimal: get spacing between blocks
        x_diffs = []
        y_diffs = []
        sorted_blocks = sorted(blocks, key=lambda b: (b['y'], b['x']))
        for i in range(1, len(sorted_blocks)):
            dx = abs(sorted_blocks[i]['x'] - sorted_blocks[i-1]['x'])
            dy = abs(sorted_blocks[i]['y'] - sorted_blocks[i-1]['y'])
            if dx != 0:
                x_diffs.append(dx)
            if dy != 0:
                y_diffs.append(dy)
        x_tol = int(np.percentile(x_diffs, 40)) if x_diffs else 8
        y_tol = int(np.percentile(y_diffs, 40)) if y_diffs else 6
        x_tol = max(4, min(x_tol, 20))
        y_tol = max(3, min(y_tol, 15))
        return x_tol, y_tol

    def cluster_rows(blocks, y_tol):
        # minimal: group by y
        blocks = sorted(blocks, key=lambda b: (b['y'], b['x']))
        rows = []
        current_row = []
        last_y = None
        for b in blocks:
            if last_y is None or abs(b['y'] - last_y) <= y_tol:
                current_row.append(b)
                last_y = b['y'] if last_y is None else (last_y + b['y']) / 2
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [b]
                last_y = b['y']
        if current_row:
            rows.append(current_row)
        return rows

    def cluster_columns(row, x_tol):
        # minimal: group by x
        row = sorted(row, key=lambda b: b['x'])
        columns = []
        current_col = []
        last_x = None
        for b in row:
            if last_x is None or abs(b['x'] - last_x) <= x_tol:
                current_col.append(b)
                last_x = b['x'] if last_x is None else (last_x + b['x']) / 2
            else:
                if current_col:
                    columns.append(current_col)
                current_col = [b]
                last_x = b['x']
        if current_col:
            columns.append(current_col)
        return columns

    def merge_multiline_cells(columns):
        # minimal: join text in a column
        merged = []
        for col in columns:
            text = ' '.join(b['text'] for b in col)
            merged.append(text)
        return merged

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        text_blocks = []
        for b in blocks:
            if 'lines' in b:
                for line in b['lines']:
                    for span in line['spans']:
                        if span['text'].strip():
                            text_blocks.append({
                                'text': span['text'].strip(),
                                'x': span['bbox'][0],
                                'y': span['bbox'][1],
                                'font': span.get('font', ''),
                                'size': span.get('size', 0),
                                'flags': span.get('flags', 0),
                                'color': span.get('color', 0),
                                'bbox': span['bbox']
                            })
        if not text_blocks:
            continue
        # minimal: get dynamic tolerances for this page
        x_tol, y_tol = calculate_dynamic_tolerances(text_blocks)
        # minimal: cluster into rows
        rows = cluster_rows(text_blocks, y_tol)
        # minimal: get max columns
        col_counts = [len(cluster_columns(row, x_tol)) for row in rows]
        max_cols = max(col_counts) if col_counts else 0
        if max_cols < min_cols:
            continue
        # minimal: build table data
        table_data = []
        for row in rows:
            columns = cluster_columns(row, x_tol)
            merged = merge_multiline_cells(columns)
            merged = (merged + [''] * max_cols)[:max_cols]
            table_data.append(merged)
        table_data = [row for row in table_data if any(cell.strip() for cell in row)]
        if not table_data or len(table_data[0]) < min_cols:
            continue
        # minimal: try to detect header
        header_idx = 0
        for i, row in enumerate(rows):
            if any(b['size'] > 10 or b['flags'] & 2 for b in row):
                header_idx = i
                break
        title = f"PyMuPDF Table (Page {page_num+1})"
        all_tables.append((title, table_data, [page_num]))
    doc.close()
    return all_tables if all_tables else [("No Tables Found", [["No valid tables detected"]], [1])]

from lxml import etree
from collections import defaultdict
import subprocess
import os

def extract_tables_from_pdf2xml(pdf_path, y_tolerance=5, x_tolerance=10, min_cols=2, min_rows=2, start_page=None, end_page=None):
    # advanced: preserve table layout using grid logic
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    xml_file = f"{base}.xml"
    subprocess.run([
        "pdftohtml",
        "-c",
        "-hidden",
        "-nodrm",
        "-xml",
        "-i",
        pdf_path,
        xml_file
    ])
    tree = etree.parse(xml_file)
    pages = tree.xpath("//page")
    all_tables = []
    for page_num, page in enumerate(pages[start_page-1 if start_page else 0:end_page if end_page else None], start=1):
        text_elements = []
        for elem in page.xpath(".//text"):
            try:
                text_elements.append({
                    'text': (elem.text or "").strip(),
                    'top': float(elem.attrib.get('top', 0)),
                    'left': float(elem.attrib.get('left', 0)),
                    'width': float(elem.attrib.get('width', 0)),
                    'height': float(elem.attrib.get('height', 0)),
                    'font': elem.attrib.get('font', ''),
                    'size': float(elem.attrib.get('size', 0))
                })
            except (ValueError, KeyError):
                continue
        if not text_elements:
            continue
        # advanced: sort by y then x
        text_elements.sort(key=lambda x: (x['top'], x['left']))
        # advanced: cluster into rows
        rows = []
        current_row = []
        last_y = None
        for elem in text_elements:
            if last_y is None or abs(elem['top'] - last_y) <= y_tolerance:
                current_row.append(elem)
                last_y = elem['top'] if last_y is None else (last_y + elem['top']) / 2
            else:
                if len(current_row) >= min_cols:
                    rows.append(current_row)
                current_row = [elem]
                last_y = elem['top']
        if len(current_row) >= min_cols:
            rows.append(current_row)
        if len(rows) < min_rows:
            continue
        # advanced: build grid using unique x positions
        all_x = set()
        for row in rows:
            for elem in row:
                all_x.add(elem['left'])
                all_x.add(elem['left'] + elem['width'])
        x_coords = sorted(list(all_x))
        # advanced: merge close x coords
        merged_x = [x_coords[0]] if x_coords else []
        for x in x_coords[1:]:
            if abs(x - merged_x[-1]) > x_tolerance:
                merged_x.append(x)
        if len(merged_x) < min_cols + 1:
            continue
        # advanced: build table grid (row x col)
        table_grid = []
        for row in rows:
            grid_row = [''] * (len(merged_x) - 1)
            for elem in row:
                # find col index by left position
                for col in range(len(merged_x) - 1):
                    if merged_x[col] - 1e-2 <= elem['left'] < merged_x[col+1] + 1e-2:
                        # check if cell spans multiple columns
                        right = elem['left'] + elem['width']
                        span = 1
                        for k in range(col+1, len(merged_x)-1):
                            if right > merged_x[k+1] - 1e-2:
                                span += 1
                            else:
                                break
                        # fill cell (merge if already present)
                        cell_text = elem['text']
                        for s in range(span):
                            if grid_row[col+s]:
                                grid_row[col+s] += ' ' + cell_text
                            else:
                                grid_row[col+s] = cell_text
                        break
            table_grid.append(grid_row)
        # advanced: clean up empty rows/cols
        table_grid = clean_table_data(table_grid)
        if table_grid and len(table_grid[0]) >= min_cols and len(table_grid) >= min_rows:
            # advanced: try to find title
            title_candidates = []
            min_top = min(elem['top'] for elem in rows[0])
            for elem in text_elements:
                if elem['top'] < min_top - elem['height'] and elem['size'] > rows[0][0]['size']:
                    title_candidates.append(elem['text'])
            table_title = " ".join(title_candidates) if title_candidates else f"Table {len(all_tables) + 1}"
            # advanced: check for data variation
            col_values = [set() for _ in range(len(table_grid[0]))]
            for row in table_grid:
                for i, cell in enumerate(row):
                    if cell.strip():
                        col_values[i].add(cell.strip())
            if any(len(values) > 1 for values in col_values):
                all_tables.append((table_title, table_grid, [page_num]))
    try:
        os.remove(xml_file)
    except OSError:
        pass
    return all_tables

def enhance_image_for_detection(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better table detection
    """
    # Convert PIL to CV2
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply adaptive histogram equalization
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
    
    # Convert back to PIL
    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    return enhanced_pil

def detect_table_structure(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect table structure using enhanced computer vision techniques
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Detect lines with different scales
    scales = [(30, 1), (20, 1), (40, 1)]  # Different horizontal scales
    horizontal_lines = np.zeros_like(binary)
    
    for scale_x, scale_y in scales:
        size_x = binary.shape[1] // scale_x
        struct_x = cv2.getStructuringElement(cv2.MORPH_RECT, (size_x, scale_y))
        temp_x = cv2.erode(binary, struct_x)
        temp_x = cv2.dilate(temp_x, struct_x)
        horizontal_lines = cv2.bitwise_or(horizontal_lines, temp_x)
    
    scales = [(1, 30), (1, 20), (1, 40)]  # Different vertical scales
    vertical_lines = np.zeros_like(binary)
    
    for scale_x, scale_y in scales:
        size_y = binary.shape[0] // scale_y
        struct_y = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_x, size_y))
        temp_y = cv2.erode(binary, struct_y)
        temp_y = cv2.dilate(temp_y, struct_y)
        vertical_lines = cv2.bitwise_or(vertical_lines, temp_y)
    
    # Combine lines
    table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        table_structure, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and validate contours
    table_boxes = []
    min_area = 1000  # Minimum area threshold
    min_aspect_ratio = 0.2  # Minimum aspect ratio
    max_aspect_ratio = 5.0  # Maximum aspect ratio
    
    for i, contour in enumerate(contours):
        # Check if contour has children (indicating table structure)
        has_children = hierarchy[0][i][2] != -1
        
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Validate contour properties
        if (area > min_area and 
            min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
            (has_children or cv2.countNonZero(table_structure[y:y+h, x:x+w]) > area * 0.1)):
            table_boxes.append((x, y, x + w, y + h))
    
    # Merge overlapping boxes
    if table_boxes:
        table_boxes = np.array(table_boxes)
        indices = nms(
            torch.tensor(table_boxes, dtype=torch.float32),
            torch.ones(len(table_boxes)),  # All boxes have same confidence
            iou_threshold=0.3
        )
        table_boxes = table_boxes[indices.numpy()]
    
    return table_boxes.tolist() if len(table_boxes) > 0 else []

def extract_tables_unstructured(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Tuple[str, List[List[str]], List[int]]]:
    """
    Enhanced table extraction using Unstructured VLM with high-resolution processing and fallback clustering for complex, borderless tables.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import Table, Text
        import fitz
        import re
        import logging
        # Configure high-resolution settings
        extraction_settings = {
            "pdf_image_dpi": 300,
            "strategy": "hi_res",
            "ocr_languages": ["eng"],
            "infer_table_structure": True,
            "preserve_whitespace": True,
            "include_page_breaks": True,
            "max_partition": 1000,
            "image_processing": {
                "extract_images": True,
                "extract_image_tables": True,
                "extract_tables": True
            }
        }
        # Extract elements
        elements = partition_pdf(
            filename=pdf_path,
            start=start_page - 1 if start_page else None,
            end=end_page if end_page else None,
            **extraction_settings
        )
        tables = []
        current_page = 1
        table_counter = 1
        table_context = []
        found_table = False
        for element in elements:
            try:
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                    current_page = element.metadata.page_number
                if isinstance(element, Text):
                    text = element.text.strip()
                    if text:
                        table_context.append({
                            'text': text,
                            'page': current_page,
                            'type': 'text',
                            'position': getattr(element.metadata, 'coordinates', None)
                        })
                        if len(table_context) > 10:
                            table_context.pop(0)
                if isinstance(element, Table):
                    found_table = True
                    table_data = []
                    if hasattr(element, 'cells') and element.cells:
                        table_data = process_structured_cells(element.cells)
                    elif hasattr(element, 'text') and element.text:
                        table_data = process_text_based_table(element.text)
                    if table_data:
                        cleaned_data = clean_and_validate_table(table_data)
                        if cleaned_data and is_valid_table(cleaned_data):
                            table_metadata = extract_table_metadata(element)
                            title = find_table_title(
                                element,
                                table_context,
                                current_page,
                                table_counter,
                                table_metadata
                            )
                            tables.append((
                                title,
                                cleaned_data,
                                [current_page],
                                table_metadata
                            ))
                            table_counter += 1
            except Exception as e:
                logging.warning(f"Error processing element on page {current_page}: {str(e)}")
                continue
        # Fallback: If no Table elements found, try clustering Text elements into a table
        if not tables:
            try:
                doc = fitz.open(pdf_path)
                for page_num in range((start_page or 1) - 1, (end_page or doc.page_count)):
                    page = doc[page_num]
                    text_blocks = []
                    for b in page.get_text("dict")['blocks']:
                        if 'lines' in b:
                            for line in b['lines']:
                                for span in line['spans']:
                                    if span['text'].strip():
                                        text_blocks.append({
                                            'text': span['text'].strip(),
                                            'x': span['bbox'][0],
                                            'y': span['bbox'][1],
                                            'font': span.get('font', ''),
                                            'size': span.get('size', 0),
                                            'flags': span.get('flags', 0),
                                            'color': span.get('color', 0),
                                            'bbox': span['bbox']
                                        })
                    def cluster_rows(blocks, y_tol=8):
                        blocks = sorted(blocks, key=lambda b: (b['y'], b['x']))
                        rows = []
                        current_row = []
                        last_y = None
                        for b in blocks:
                            if last_y is None or abs(b['y'] - last_y) <= y_tol:
                                current_row.append(b)
                                last_y = b['y'] if last_y is None else (last_y + b['y']) / 2
                            else:
                                if current_row:
                                    rows.append(current_row)
                                current_row = [b]
                                last_y = b['y']
                        if current_row:
                            rows.append(current_row)
                        return rows
                    def cluster_columns(row, x_tol=6):
                        row = sorted(row, key=lambda b: b['x'])
                        columns = []
                        current_col = []
                        last_x = None
                        for b in row:
                            if last_x is None or abs(b['x'] - last_x) <= x_tol:
                                current_col.append(b)
                                last_x = b['x'] if last_x is None else (last_x + b['x']) / 2
                            else:
                                if current_col:
                                    columns.append(current_col)
                                current_col = [b]
                                last_x = b['x']
                        if current_col:
                            columns.append(current_col)
                        return columns
                    def merge_multiline_cells(columns):
                        merged = []
                        for col in columns:
                            text = ' '.join(b['text'] for b in col)
                            merged.append(text)
                        return merged
                    rows = cluster_rows(text_blocks, y_tol=8)
                    col_counts = [len(cluster_columns(row, x_tol=6)) for row in rows]
                    max_cols = max(col_counts) if col_counts else 0
                    if max_cols < 2:
                        continue
                    table_data = []
                    for row in rows:
                        columns = cluster_columns(row, x_tol=6)
                        merged = merge_multiline_cells(columns)
                        merged = (merged + [''] * max_cols)[:max_cols]
                        table_data.append(merged)
                    table_data = [row for row in table_data if any(cell.strip() for cell in row)]
                    if not table_data or len(table_data[0]) < 2:
                        continue
                    title = f"Unstructured Table (Fallback, Page {page_num+1})"
                    tables.append((title, table_data, [page_num+1], {}))
                doc.close()
            except Exception as e:
                logging.warning(f"Unstructured fallback clustering error: {str(e)}")
        # Post-process tables for better quality
        processed_tables = post_process_tables(tables)
        return [(title, data, pages) for title, data, pages, _ in processed_tables] if processed_tables else [("No Tables Found", [["No valid tables detected"]], [1])]
    except Exception as e:
        error_msg = f"Error in unstructured extraction: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return [("Extraction Error", [[error_msg]], [1])]

def process_structured_cells(cells: List[List[Any]]) -> List[List[str]]:
    """Process structured table cells with enhanced cleaning"""
    processed_data = []
    for row in cells:
        processed_row = []
        for cell in row:
            # Handle different cell types
            if isinstance(cell, (list, tuple)):
                cell_text = ' '.join(str(item).strip() for item in cell if str(item).strip())
            else:
                cell_text = str(cell).strip()
            
            # Clean cell text
            cell_text = re.sub(r'\s+', ' ', cell_text)  # Normalize whitespace
            cell_text = cell_text.replace('\n', ' ')  # Remove newlines
            processed_row.append(cell_text)
        
        if any(cell for cell in processed_row):  # Skip empty rows
            processed_data.append(processed_row)
    
    return processed_data

def process_text_based_table(text: Union[str, List[str]]) -> List[List[str]]:
    """Process text-based table with enhanced parsing"""
    if isinstance(text, str):
        # Split into rows with multiple delimiters
        rows = text.strip().split('\n')
        table_data = []
        
        for row in rows:
            # Try different delimiters
            for delimiter in ['\t', '  ', '|', ';']:
                cells = [cell.strip() for cell in row.split(delimiter) if cell.strip()]
                if len(cells) >= 2:  # Valid row should have at least 2 cells
                    table_data.append(cells)
                    break
        
        return table_data
    elif isinstance(text, list):
        return [[str(cell).strip() for cell in row] for row in text if any(str(cell).strip() for cell in row)]
    
    return []

def clean_and_validate_table(table_data: List[List[str]]) -> List[List[str]]:
    """Clean and validate table data with enhanced rules"""
    if not table_data:
        return []
    
    # Remove empty rows and normalize columns
    cleaned_data = []
    max_cols = max(len(row) for row in table_data)
    
    if max_cols < 2:  # Require at least 2 columns
        return []
    
    for row in table_data:
        # Clean cells
        cleaned_row = []
        for cell in row:
            # Enhanced cell cleaning
            cell = re.sub(r'\s+', ' ', cell.strip())  # Normalize whitespace
            cell = re.sub(r'[^\S\n]+', ' ', cell)  # Remove multiple spaces
            cell = cell.replace('\n', ' ')  # Remove newlines
            cleaned_row.append(cell)
        
        # Pad or truncate row to match max columns
        cleaned_row = (cleaned_row + [''] * max_cols)[:max_cols]
        
        if any(cell.strip() for cell in cleaned_row):  # Skip empty rows
            cleaned_data.append(cleaned_row)
    
    return cleaned_data if len(cleaned_data) >= 2 else []  # Require at least 2 rows

def is_valid_table(table_data: List[List[str]]) -> bool:
    """Enhanced table validation with multiple criteria"""
    if not table_data or len(table_data) < 2:
        return False
    
    # Check column consistency
    col_counts = [len(row) for row in table_data]
    if not all(count == col_counts[0] for count in col_counts) or col_counts[0] < 2:
        return False
    
    # Check data variation
    unique_values = set()
    for row in table_data:
        for cell in row:
            if cell.strip():
                unique_values.add(cell.strip())
    
    # Require sufficient unique values
    if len(unique_values) < 3:
        return False
    
    # Check for header-like first row
    first_row = table_data[0]
    other_rows = table_data[1:]
    
    # Header should be different from data rows
    header_cells = set(cell.strip() for cell in first_row if cell.strip())
    data_cells = set(cell.strip() for row in other_rows for cell in row if cell.strip())
    
    return len(header_cells.intersection(data_cells)) < len(header_cells) * 0.5

def extract_table_metadata(element: Any) -> Dict[str, Any]:
    """Extract enhanced metadata from table element"""
    metadata = {
        'coordinates': None,
        'confidence': None,
        'table_type': None,
        'style': None
    }
    
    try:
        if hasattr(element, 'metadata'):
            # Extract coordinates if available
            if hasattr(element.metadata, 'coordinates'):
                metadata['coordinates'] = element.metadata.coordinates
            
            # Extract text style information
            if hasattr(element.metadata, 'text_as_html'):
                metadata['style'] = 'html'
            
            # Determine table type
            if hasattr(element, 'cells'):
                metadata['table_type'] = 'structured'
            elif hasattr(element, 'text'):
                metadata['table_type'] = 'text_based'
            
            # Set confidence based on available data
            metadata['confidence'] = 'high' if metadata['table_type'] == 'structured' else 'medium'
    except Exception as e:
        logger.warning(f"Error extracting table metadata: {str(e)}")
    
    return metadata

def find_table_title(element: Any, context: List[Dict], current_page: int, table_counter: int, metadata: Dict) -> str:
    """Find table title using enhanced context analysis"""
    title_candidates = []
    
    # Check element's own metadata first
    if hasattr(element, 'metadata'):
        if hasattr(element.metadata, 'text_as_html'):
            html_text = element.metadata.text_as_html
            if html_text and ('table' in html_text.lower() or 'figure' in html_text.lower()):
                title_candidates.append(('high', html_text.strip()))
    
    # Analyze context
    for ctx in reversed(context):  # Recent context first
        if ctx['page'] == current_page:
            text = ctx['text'].lower()
            if 'table' in text or 'figure' in text:
                # Score the title candidate
                score = 'high' if text.startswith(('table', 'figure')) else 'medium'
                title_candidates.append((score, ctx['text']))
    
    # Select best title
    if title_candidates:
        # Prefer high confidence titles
        high_conf_titles = [t for s, t in title_candidates if s == 'high']
        if high_conf_titles:
            return high_conf_titles[0]
        return title_candidates[0][1]
    
    # Fallback to generic title with metadata
    confidence = metadata.get('confidence', 'low')
    return f"Table {table_counter} (Confidence: {confidence})"

def post_process_tables(tables: List[Tuple[str, List[List[str]], List[int], Dict[str, Any]]]) -> List[Tuple[str, List[List[str]], List[int], Dict[str, Any]]]:
    """Post-process tables for better quality"""
    processed_tables = []
    
    for title, data, pages, metadata in tables:
        # Skip invalid tables
        if not data or len(data) < 2 or not all(row for row in data):
            continue
        
        # Normalize column widths
        max_col_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
        
        # Format data for better readability
        formatted_data = []
        for row in data:
            formatted_row = []
            for i, cell in enumerate(row):
                # Pad or truncate cells for consistency
                cell_str = str(cell)
                if len(cell_str) > max_col_widths[i] * 2:  # Truncate very long cells
                    cell_str = cell_str[:max_col_widths[i] * 2 - 3] + '...'
                formatted_row.append(cell_str)
            formatted_data.append(formatted_row)
        
        processed_tables.append((title, formatted_data, pages, metadata))
    
    return processed_tables

def process_image_for_detection(image, processor, device):
    """Helper function to process images for the model"""
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Prepare image for model
    inputs = processor(
        images=image,
        return_tensors="pt",
        do_resize=True,
        size={"shortest_edge": 800, "longest_edge": 1333},
        do_pad=True
    )
    
    # Move to correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def detect_header_structure(blocks, y_tolerance=10):
    """Detect multi-row header structure and column spans"""
    if not blocks:
        return [], {}

    # Sort blocks by y-coordinate
    sorted_blocks = sorted(blocks, key=lambda b: b['y'])
    first_y = sorted_blocks[0]['y']
    
    # Collect potential header rows
    header_rows = []
    current_row = []
    last_y = first_y
    
    for block in sorted_blocks:
        if abs(block['y'] - last_y) <= y_tolerance:
            current_row.append(block)
        else:
            if current_row:
                header_rows.append(current_row)
            current_row = [block]
            last_y = block['y']
            # Stop after finding non-header content
            if len(header_rows) > 0 and block['y'] - first_y > 3 * y_tolerance:
                break
    
    if current_row:
        header_rows.append(current_row)
    
    # Detect column spans
    spans = {}
    for row_idx, row in enumerate(header_rows):
        sorted_row = sorted(row, key=lambda b: b['x'])
        for i, block in enumerate(sorted_row):
            text = block['text'].strip()
            if text:
                # Check if this cell spans multiple columns
                if i < len(sorted_row) - 1:
                    next_block = sorted_row[i + 1]
                    gap = next_block['x'] - (block['x'] + block['bbox'][2] - block['bbox'][0])
                    if gap > 20:  # Large gap indicates potential span
                        spans[(row_idx, i)] = {'text': text, 'span': 2}  # Assume span of 2 for now
    
    return header_rows, spans

def merge_header_cells(header_rows, spans):
    """Merge header cells based on detected spans"""
    merged_headers = []
    for row_idx, row in enumerate(header_rows):
        merged_row = []
        skip_next = False
        sorted_row = sorted(row, key=lambda b: b['x'])
        
        for i, block in enumerate(sorted_row):
            if skip_next:
                skip_next = False
                continue
                
            text = block['text'].strip()
            if (row_idx, i) in spans:
                # This cell spans multiple columns
                span_info = spans[(row_idx, i)]
                merged_row.append(span_info['text'])
                skip_next = True
            else:
                merged_row.append(text)
        
        merged_headers.append(merged_row)
    
    return merged_headers

def extract_tables_combined_transformer(pdf_path, start_page=None, end_page=None):
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF
    import numpy as np
    all_tables = []
    images = convert_from_path(
        pdf_path,
        first_page=start_page or 1,
        last_page=end_page or None,
        dpi=300,
        grayscale=False,
        thread_count=4
    )
    doc = fitz.open(pdf_path)
    
    def cluster_by_column_alignment(blocks, x_tolerance=5):
        """Cluster text blocks by their x-coordinates to identify potential columns"""
        x_positions = defaultdict(list)
        for block in blocks:
            # Round x position to nearest multiple of tolerance
            x_pos = round(block['x'] / x_tolerance) * x_tolerance
            x_positions[x_pos].append(block)
        return sorted(x_positions.items())

    def detect_table_structure(blocks):
        """Detect table structure using column alignment and data type patterns"""
        # Sort blocks by y-coordinate first
        blocks = sorted(blocks, key=lambda b: b['y'])
        
        # Detect header structure
        header_rows, spans = detect_header_structure(blocks)
        merged_headers = merge_header_cells(header_rows, spans)
        
        # Analyze data types in columns
        data_patterns = defaultdict(set)
        data_blocks = [b for b in blocks if b not in sum(header_rows, [])]
        
        for block in data_blocks:
            text = block['text'].strip()
            x_pos = round(block['x'] / 5) * 5
            
            # Detect data types
            if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', text):  # Date
                data_patterns[x_pos].add('date')
            elif re.match(r'^\d+\.\d{2}$', text):  # Currency
                data_patterns[x_pos].add('currency')
            elif re.match(r'^\d+$', text):  # Integer
                data_patterns[x_pos].add('number')
            else:
                data_patterns[x_pos].add('text')
        
        return merged_headers, data_patterns, data_blocks

    def process_financial_data(text):
        """Process financial data with specific formatting"""
        # Remove currency symbols and commas
        text = re.sub(r'[$,]', '', text.strip())
        
        # Convert percentage strings
        if text.endswith('%'):
            try:
                return str(float(text.rstrip('%')) / 100)
            except ValueError:
                return text
        
        # Try to convert to float for numerical values
        try:
            return f"{float(text):.2f}"
        except ValueError:
            return text

    for page_num, image in enumerate(images, start=(start_page if start_page else 1)):
        try:
            # Step 1: Detect table regions
            detection_results = combined_transformer.detect_tables(image)
            boxes = detection_results["boxes"]
            scores = detection_results["scores"]
            page = doc[page_num - 1]
            page_width, page_height = page.rect.width, page.rect.height
            
            # Get all text blocks from the page
            text_blocks = []
            for b in page.get_text("dict")['blocks']:
                if 'lines' in b:
                    for line in b['lines']:
                        for span in line['spans']:
                            if span['text'].strip():
                                text_blocks.append({
                                    'text': span['text'].strip(),
                                    'x': span['bbox'][0],
                                    'y': span['bbox'][1],
                                    'font': span.get('font', ''),
                                    'size': span.get('size', 0),
                                    'flags': span.get('flags', 0),
                                    'color': span.get('color', 0),
                                    'bbox': span['bbox']
                                })

            for idx, (box, score) in enumerate(zip(boxes, scores)):
                if score < 0.5:
                    continue
                
                x0, y0, x1, y1 = map(int, box.tolist())
                # Map coordinates
                pdf_x0 = x0 / image.width * page_width
                pdf_y0 = y0 / image.height * page_height
                pdf_x1 = x1 / image.width * page_width
                pdf_y1 = y1 / image.height * page_height
                
                # Filter blocks within the detected region
                region = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                region_blocks = []
                for block in text_blocks:
                    cx = (block['bbox'][0] + block['bbox'][2]) / 2
                    cy = (block['bbox'][1] + block['bbox'][3]) / 2
                    if region.contains(fitz.Point(cx, cy)):
                        region_blocks.append(block)

                if not region_blocks:
                    continue

                # Detect table structure including headers
                merged_headers, data_patterns, data_blocks = detect_table_structure(region_blocks)
                
                # Start building table data with headers
                table_data = merged_headers.copy()  # Start with merged headers
                
                # Process data rows
                current_row = []
                last_y = None
                y_tolerance = 6  # Adjust based on text spacing
                
                # Sort remaining blocks by y then x
                sorted_blocks = sorted(data_blocks, key=lambda b: (b['y'], b['x']))
                
                for block in sorted_blocks:
                    if last_y is None or abs(block['y'] - last_y) > y_tolerance:
                        if current_row:
                            table_data.append(current_row)
                        current_row = []
                        last_y = block['y']
                    
                    # Process text based on data patterns
                    x_pos = round(block['x'] / 5) * 5
                    text = block['text'].strip()
                    if 'currency' in data_patterns[x_pos]:
                        text = process_financial_data(text)
                    elif 'date' in data_patterns[x_pos]:
                        # Keep date format as is
                        pass
                    
                    current_row.append(text)
                
                if current_row:
                    table_data.append(current_row)
                
                # Normalize table data (ensure all rows have same number of columns)
                if table_data:
                    max_cols = max(len(row) for row in table_data)
                    table_data = [row + [''] * (max_cols - len(row)) for row in table_data]
                    
                    # Remove empty rows and columns
                    table_data = [row for row in table_data if any(cell.strip() for cell in row)]
                    if table_data and len(table_data[0]) >= 2:
                        title = f"Table {idx + 1} (Page {page_num}, Confidence: {score:.2f})"
                        all_tables.append((title, table_data, [page_num]))

        except Exception as e:
            logger.warning(f"Error processing page {page_num}: {str(e)}")
            continue

    doc.close()
    return all_tables if all_tables else [("No Tables Found", [["No valid tables detected"]], [1])]

def extract_text_with_tabula(filepath: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Tuple[str, List[List[str]], List[int]]]:
    """
    Extract text from PDF using Tabula with enhanced text processing and ordering
    """
    try:
        # Use Tabula to extract tables
        tables = tabula.read_pdf(filepath, pages=f"{start_page}-{end_page}" if start_page and end_page else 'all')
        
        # Convert extracted tables to structured format
        structured_tables = []
        for table in tables:
            title = f"Tabula Table (Page {table['page']})"
            structured_tables.append((title, table['data'], [table['page']]))
        
        return structured_tables
    except Exception as e:
        error_msg = f"Tabula extraction error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return [("Tabula Error", [[error_msg]], [1])]

def check_and_install_dependencies():
    """Check and install required dependencies for CascadeTabNet"""
    try:
        import mmdet
        import pytesseract
        return True
    except ImportError:
        print("Installing required dependencies for CascadeTabNet...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "mmdet", "pytesseract", "mmcv-full"])
        try:
            import mmdet
            import pytesseract
            return True
        except ImportError as e:
            print(f"Failed to install dependencies: {str(e)}")
            return False

# Update the CascadeTabNet imports with better error handling
try:
    from mmdet.apis import init_detector, inference_detector
    import pytesseract
    import mmcv
    CASCADETABNET_AVAILABLE = True
except ImportError:
    CASCADETABNET_AVAILABLE = False
    init_detector = None
    inference_detector = None
    pytesseract = None
    mmcv = None

class CascadeTabNetModel:
    """Wrapper class for CascadeTabNet model management"""
    def __init__(self):
        self.model = None
        self.config_file = None
        self.checkpoint_file = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    def initialize(self, config_dir='CascadeTabNet/Config'):
        """Initialize the CascadeTabNet model"""
        try:
            if not CASCADETABNET_AVAILABLE:
                if not check_and_install_dependencies():
                    raise ImportError("Failed to install required dependencies")
            
            config_dir = Path(config_dir)
            self.config_file = str(config_dir / 'cascade_mask_rcnn_hrnetv2.py')
            self.checkpoint_file = str(config_dir / 'cascade_mask_rcnn_hrnetv2.pth')
            
            # Verify files exist
            if not Path(self.config_file).exists():
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            if not Path(self.checkpoint_file).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_file}")
            
            self.model = init_detector(
                self.config_file,
                self.checkpoint_file,
                device=self.device
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CascadeTabNet: {str(e)}")
            return False
    
    def detect_tables(self, image):
        """Detect tables in an image using CascadeTabNet"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Run inference
            result = inference_detector(self.model, image)
            
            # Process results
            table_bboxes = []
            if isinstance(result, tuple):
                bbox_result, _ = result
            else:
                bbox_result = result
            
            # Get table detections (assuming first class is table)
            if bbox_result and len(bbox_result) > 0:
                for bbox in bbox_result[0]:
                    x1, y1, x2, y2, score = bbox
                    if score >= 0.7:  # Confidence threshold
                        table_bboxes.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(score)
                        })
            
            return table_bboxes
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            return []

# Initialize CascadeTabNet model
cascadetabnet = CascadeTabNetModel()

def extract_tables_cascadetabnet(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Tuple[str, List[List[str]], List[int]]]:
    """
    Enhanced table extraction using CascadeTabNet with improved detection and OCR
    """
    try:
        if not CASCADETABNET_AVAILABLE:
            return [("CascadeTabNet Error", [["MMDetection or pytesseract not installed. Please install required dependencies."]], [1])]
        
        # Initialize model if not already initialized
        if not cascadetabnet.model and not cascadetabnet.initialize():
            return [("CascadeTabNet Error", [["Failed to initialize CascadeTabNet model"]], [1])]
        
        # Convert PDF pages to images
        images = convert_from_path(
            pdf_path,
            first_page=start_page or 1,
            last_page=end_page or None,
            dpi=300,
            grayscale=False,
            thread_count=4
        )
        
        all_tables = []
        
        for page_num, image in enumerate(images, start=(start_page if start_page else 1)):
            try:
                # Convert PIL image to numpy array (BGR)
                img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Enhance image quality
                img_np = enhance_image_for_detection(Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))
                img_np = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                
                # Detect tables
                table_detections = cascadetabnet.detect_tables(img_np)
                
                for idx, detection in enumerate(table_detections):
                    try:
                        bbox = detection['bbox']
                        score = detection['score']
                        
                        # Extract table region
                        x1, y1, x2, y2 = map(int, bbox)
                        table_img = image.crop((x1, y1, x2, y2))
                        
                        # Enhance table image for OCR
                        table_img = enhance_image_for_ocr(table_img)
                        
                        # Extract text using OCR
                        table_data = extract_table_with_ocr(table_img)
                        
                        
                        if table_data and len(table_data) >= 2 and len(table_data[0]) >= 2:
                            # Clean and validate table data
                            table_data = clean_and_validate_table(table_data)
                            
                            if table_data:
                                title = f"Table {idx + 1} (Page {page_num}, Confidence: {score:.2f})"
                                all_tables.append((title, table_data, [page_num]))
                    
                    except Exception as e:
                        logger.warning(f"Error processing table {idx + 1} on page {page_num}: {str(e)}")
                        continue
            
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {str(e)}")
                continue
        
        return all_tables if all_tables else [("No Tables Found", [["No valid tables detected"]], [1])]
    
    except Exception as e:
        error_msg = f"CascadeTabNet extraction error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return [("CascadeTabNet Error", [[error_msg]], [1])]

def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance image quality for better OCR results"""
    try:
        # Convert to CV2 format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply adaptive histogram equalization
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.warning(f"Error enhancing image for OCR: {str(e)}")
        return image

def extract_table_with_ocr(image: Image.Image) -> List[List[str]]:
    """Extract table data using improved OCR"""
    try:
        # Configure OCR settings
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        # Get OCR data with detailed information
        ocr_data = pytesseract.image_to_data(
            image, 
            config=custom_config, 
            output_type=pytesseract.Output.DICT
        )
        
        # Group text by lines using y-coordinates
        lines = defaultdict(list)
        for i, (text, conf, x, y, w, h) in enumerate(zip(
            ocr_data['text'], 
            ocr_data['conf'], 
            ocr_data['left'], 
            ocr_data['top'], 
            ocr_data['width'], 
            ocr_data['height']
        )):
            if float(conf) > 30 and text.strip():  # Filter low confidence and empty text
                lines[y].append((x, text.strip(), w))
        
        # Sort lines by y-coordinate
        sorted_lines = []
        for y in sorted(lines.keys()):
            # Sort text within each line by x-coordinate
            line = sorted(lines[y], key=lambda x: x[0])
            sorted_lines.append([item[1] for item in line])
        
        return sorted_lines
    except Exception as e:
        logger.warning(f"Error in OCR extraction: {str(e)}")
        return []

def extract_text_with_pypdf2(filepath: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Tuple[str, List[List[str]], List[int]]]:
    """
    Extract text from PDF using PyPDF2 with enhanced text processing and ordering
    """
    try:
        # Open the PDF file
        with open(filepath, 'rb') as file:
            # Create PDF reader object
            reader = PdfReader(file)
            
            # Get total number of pages
            total_pages = len(reader.pages)
            
            # Validate and adjust page range
            start_idx = (start_page or 1) - 1
            end_idx = min(end_page or total_pages, total_pages)
            
            if start_idx < 0 or start_idx >= total_pages:
                return [("PyPDF2 Error", [["Invalid start page"]], [1])]
            
            text_blocks = []
            
            # Process each page in the range
            for page_num in range(start_idx, end_idx):
                try:
                    # Get the page
                    page = reader.pages[page_num]
                    
                    # Extract text with layout preservation
                    text = page.extract_text()
                    
                    if text.strip():
                        # Process the text
                        processed_blocks = process_pdf_text(text, page_num + 1)
                        text_blocks.extend(processed_blocks)
                
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            if not text_blocks:
                return [("No Text Found", [["No text could be extracted"]], [1])]
            
            return text_blocks
        
    except Exception as e:
        error_msg = f"PyPDF2 extraction error: {str(e)}"
        logger.error(error_msg)
        return [("PyPDF2 Error", [[error_msg]], [1])]

def process_pdf_text(text: str, page_num: int) -> List[Tuple[str, List[List[str]], List[int]]]:
    """
    Process and organize extracted text into structured blocks
    """
    blocks = []
    
    try:
        # Split text into lines
        lines = text.split('\n')
        
        # Initialize variables for block processing
        current_block = []
        current_block_type = None
        block_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                # Process completed block
                if current_block:
                    block_count += 1
                    blocks.append(process_text_block(current_block, block_count, page_num))
                    current_block = []
                continue
            
            # Detect block type (paragraph, list, heading, etc.)
            block_type = detect_block_type(line)
            
            if block_type != current_block_type and current_block:
                # Start new block if type changes
                block_count += 1
                blocks.append(process_text_block(current_block, block_count, page_num))
                current_block = []
            
            current_block.append(line)
            current_block_type = block_type
        
        # Process last block
        if current_block:
            block_count += 1
            blocks.append(process_text_block(current_block, block_count, page_num))
        
        return blocks
    
    except Exception as e:
        logger.warning(f"Error processing text on page {page_num}: {str(e)}")
        return [(f"Text Block (Page {page_num})", [[text]], [page_num])]

def detect_block_type(line: str) -> str:
    """
    Detect the type of text block based on its characteristics
    """
    line = line.strip()
    
    # Check for heading patterns
    if re.match(r'^[A-Z][^a-z]{0,2}\.?\s+[A-Z]', line):  # e.g., "1. TITLE" or "A. SECTION"
        return 'heading'
    
    # Check for list items
    if re.match(r'^[\d\-•\*]\s+', line):  # Numbers, bullets, etc.
        return 'list'
    
    # Check for table-like content
    if '\t' in line or '    ' in line:
        return 'table'
    
    # Default to paragraph
    return 'paragraph'

def process_text_block(lines: List[str], block_num: int, page_num: int) -> Tuple[str, List[List[str]], List[int]]:
    """
    Process a block of text lines into a structured format
    """
    block_type = detect_block_type(lines[0])
    
    # Format title based on block type
    title = f"{block_type.title()} Block {block_num} (Page {page_num})"
    
    # Process lines based on block type
    if block_type == 'table':
        # Split by tabs or multiple spaces
        data = [re.split(r'\t+|\s{4,}', line.strip()) for line in lines]
    elif block_type == 'list':
        # Remove list markers and create single-column list
        data = [[re.sub(r'^[\d\-•\*]\s+', '', line.strip())] for line in lines]
    else:
        # Keep lines as is for paragraphs and headings
        data = [[line.strip()] for line in lines]
    
    return title, data, [page_num]

if __name__ == '__main__':
    app.run(debug=True)
