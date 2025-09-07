import fitz
import easyocr
import tempfile
import os
import re
import json
import logging
from pathlib import Path
from PIL import Image, ImageEnhance
from collections import defaultdict, Counter
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import torch

class AdaptivePDFReorderer:
    def __init__(self, gpu_optimization=True, max_workers=None):
        self.setup_logging()
        self.gpu_optimization = gpu_optimization
        self.max_workers = max_workers or min(8, os.cpu_count())
        
        # Initialize EasyOCR with GPU optimization
        self.reader = self._initialize_ocr()
        
        # Analysis results
        self.page_findings = {}
        self.confidence_scores = {}
        self.disambiguation_data = {}
        
        self.logger.info(f"Initialized with GPU optimization: {gpu_optimization}")
        self.logger.info(f"Max workers: {self.max_workers}")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_ocr(self):
        try:
            if self.gpu_optimization and torch.cuda.is_available():
                torch.cuda.empty_cache()
                reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.logger.info(f"GPU initialized. CUDA memory: {gpu_memory:.1f}GB")
            else:
                reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.logger.info("Using CPU for OCR")
            return reader
        except Exception as e:
            self.logger.error(f"OCR initialization failed: {e}")
            return easyocr.Reader(['en'], gpu=False, verbose=False)
    
    def analyze_pdf(self, pdf_path):
        """Comprehensive PDF analysis with adaptive page detection"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        self.logger.info(f"Analyzing PDF: {pdf_path}")
        self.logger.info(f"Total pages: {total_pages}")
        
        # Parallel processing for speed
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for page_num in range(total_pages):
                future = executor.submit(self._analyze_single_page, doc, page_num)
                futures.append((page_num, future))
            
            # Collect results
            for page_num, future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        self.page_findings[page_num] = result
                except Exception as e:
                    self.logger.error(f"Failed to analyze page {page_num + 1}: {e}")
        
        doc.close()
        
        # Clean up GPU memory
        if self.gpu_optimization:
            torch.cuda.empty_cache()
        
        self.logger.info(f"Analysis complete. Found page numbers on {len(self.page_findings)} pages")
        return self._create_page_mapping()
    
    def _analyze_single_page(self, doc, page_num):
        """Analyze single page with multiple OCR approaches"""
        page = doc[page_num]
        
        # Get page dimensions for region-based analysis
        rect = page.rect
        page_width, page_height = rect.width, rect.height
        
        all_found_numbers = set()
        detection_methods = {}
        
        try:
            # Method 1: Standard full-page OCR
            full_page_numbers = self._standard_ocr(page)
            all_found_numbers.update(full_page_numbers)
            if full_page_numbers:
                detection_methods['full_page'] = full_page_numbers
            
            # Method 2: Region-based OCR (header/footer areas)
            region_numbers = self._region_based_ocr(page, page_width, page_height)
            all_found_numbers.update(region_numbers)
            if region_numbers:
                detection_methods['regions'] = region_numbers
            
            # Method 3: Multi-resolution OCR
            multi_res_numbers = self._multi_resolution_ocr(page)
            all_found_numbers.update(multi_res_numbers)
            if multi_res_numbers:
                detection_methods['multi_res'] = multi_res_numbers
            
            # Filter to reasonable page numbers
            valid_numbers = [n for n in all_found_numbers if 1 <= n <= 200]
            
            if valid_numbers:
                return {
                    'found_numbers': sorted(valid_numbers),
                    'detection_methods': detection_methods,
                    'primary_candidates': self._identify_primary_candidates(valid_numbers, detection_methods)
                }
        
        except Exception as e:
            self.logger.debug(f"Error analyzing page {page_num + 1}: {e}")
        
        return None
    
    def _standard_ocr(self, page):
        """Standard OCR approach"""
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_path = tempfile.mktemp(suffix='.png')
            pix.save(img_path)
            
            # Preprocess image
            img = Image.open(img_path)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            img = img.convert('L')
            
            # OCR
            results = self.reader.readtext(img_path, detail=True)
            numbers = self._extract_numbers_from_ocr(results)
            
            os.remove(img_path)
            return numbers
            
        except Exception:
            return []
    
    def _region_based_ocr(self, page, page_width, page_height):
        """OCR specific regions where page numbers typically appear"""
        numbers = set()
        
        # Define key regions
        regions = [
            fitz.Rect(0, 0, page_width, page_height * 0.08),  # Top header
            fitz.Rect(0, page_height * 0.92, page_width, page_height),  # Bottom footer
            fitz.Rect(page_width * 0.4, page_height * 0.9, page_width * 0.6, page_height),  # Center bottom
        ]
        
        for region in regions:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=region)
                
                if pix.width > 20 and pix.height > 10:
                    img_path = tempfile.mktemp(suffix='.png')
                    pix.save(img_path)
                    
                    # Enhanced preprocessing for small regions
                    img = Image.open(img_path)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(2.0)
                    img = img.convert('L')
                    
                    results = self.reader.readtext(img_path, detail=True)
                    region_numbers = self._extract_numbers_from_ocr(results, min_confidence=0.3)
                    numbers.update(region_numbers)
                    
                    os.remove(img_path)
                    
            except Exception:
                continue
        
        return list(numbers)
    
    def _multi_resolution_ocr(self, page):
        """OCR at multiple resolutions"""
        numbers = set()
        resolutions = [2, 4, 6]
        
        for zoom in resolutions:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                img_path = tempfile.mktemp(suffix='.png')
                pix.save(img_path)
                
                results = self.reader.readtext(img_path, detail=True)
                res_numbers = self._extract_numbers_from_ocr(results, min_confidence=0.4)
                numbers.update(res_numbers)
                
                os.remove(img_path)
                
            except Exception:
                continue
        
        return list(numbers)
    
    def _extract_numbers_from_ocr(self, ocr_results, min_confidence=0.5):
        """Extract page numbers from OCR results using multiple patterns"""
        numbers = set()
        
        patterns = [
            (r'^\s*(\d+)\s*$', 1.0),          # Isolated number (high confidence)
            (r'-\s*(\d+)\s*-', 0.9),          # Dash format
            (r'page\s*(\d+)', 0.8),           # "Page N"
            (r'(\d+)\s*$', 0.7),              # Number at line end
            (r'^\s*(\d+)', 0.6),              # Number at line start
        ]
        
        for bbox, text, confidence in ocr_results:
            if confidence < min_confidence:
                continue
            
            text_clean = text.strip()
            
            for pattern, pattern_weight in patterns:
                matches = re.findall(pattern, text_clean, re.IGNORECASE)
                for match in matches:
                    if match.isdigit():
                        num = int(match)
                        if 1 <= num <= 200:
                            numbers.add(num)
        
        return list(numbers)
    
    def _identify_primary_candidates(self, numbers, methods):
        """Identify the most likely page number candidates"""
        frequency = Counter()
        for method_numbers in methods.values():
            frequency.update(method_numbers)
        
        primary = []
        for num, count in frequency.most_common():
            if count >= 2 or len([m for m in methods.values() if num in m]) >= 2:
                primary.append(num)
        
        return primary[:3]
    
    def _create_page_mapping(self):
        """Create intelligent page number to position mapping"""
        self.logger.info("Creating page mapping with disambiguation...")
        
        detections = defaultdict(list)
        for pdf_pos, data in self.page_findings.items():
            for page_num in data['primary_candidates']:
                detections[page_num].append(pdf_pos)
        
        final_mapping = {}
        used_positions = set()
        
        for page_num in sorted(detections.keys()):
            positions = detections[page_num]
            available_positions = [p for p in positions if p not in used_positions]
            
            if available_positions:
                best_pos = self._choose_best_position(page_num, available_positions)
                final_mapping[page_num] = best_pos
                used_positions.add(best_pos)
        
        self.logger.info(f"Final mapping created: {len(final_mapping)} pages mapped")
        
        for page_num in sorted(final_mapping.keys()):
            pdf_pos = final_mapping[page_num]
            self.logger.info(f"Page {page_num} -> PDF position {pdf_pos + 1}")
        
        return final_mapping
    
    def _choose_best_position(self, page_num, positions):
        """Choose the best PDF position for a page number using heuristics"""
        if len(positions) == 1:
            return positions[0]
        
        position_scores = {}
        
        for pos in positions:
            score = 0
            
            total_detections = len(self.page_findings[pos]['found_numbers'])
            score += 1.0 / (total_detections + 1)
            
            if page_num in self.page_findings[pos]['primary_candidates']:
                score += 1.0
            
            expected_range_start = max(0, page_num - 3)
            expected_range_end = min(len(self.page_findings), page_num + 3)
            if expected_range_start <= pos <= expected_range_end:
                score += 0.5
            
            position_scores[pos] = score
        
        return max(positions, key=lambda p: position_scores.get(p, 0))
    
    def create_reordered_pdf(self, input_path, output_path, page_mapping):
        """Create reordered PDF based on page mapping"""
        if not page_mapping:
            raise ValueError("No page mapping available for reordering")
        
        self.logger.info(f"Creating reordered PDF: {output_path}")
        
        doc = fitz.open(input_path)
        reordered_doc = fitz.open()
        
        sorted_pages = sorted(page_mapping.items())
        
        for page_num, pdf_position in sorted_pages:
            reordered_doc.insert_pdf(doc, from_page=pdf_position, to_page=pdf_position)
            self.logger.debug(f"Added page {page_num} from PDF position {pdf_position + 1}")
        
        reordered_doc.save(output_path)
        doc.close()
        reordered_doc.close()
        
        self.logger.info(f"Reordered PDF saved: {output_path}")
        self.logger.info(f"Pages reordered: {len(sorted_pages)}")
        
        return len(sorted_pages)
    
    def generate_report(self, output_path, page_mapping):
        """Generate detailed analysis report"""
        report = {
            'summary': {
                'total_pdf_pages': len(self.page_findings) if hasattr(self, 'page_findings') else 0,
                'pages_with_numbers': len([p for p in self.page_findings.values() if p['found_numbers']]),
                'successful_mappings': len(page_mapping),
                'page_range': f"{min(page_mapping.keys())}-{max(page_mapping.keys())}" if page_mapping else "None"
            },
            'page_mappings': page_mapping,
            'detailed_findings': self.page_findings
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Analysis report saved: {output_path}")
    
    def process_pdf(self, input_path, output_dir=None):
        """Complete PDF processing pipeline"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_path}")
        
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        base_name = input_path.stem
        output_pdf = output_dir / f"{base_name}_reordered.pdf"
        report_path = output_dir / f"{base_name}_analysis_report.json"
        
        try:
            # Step 1: Analyze PDF
            self.logger.info("Step 1: Analyzing PDF structure...")
            page_mapping = self.analyze_pdf(input_path)
            
            if not page_mapping:
                raise ValueError("No page numbers could be detected and mapped")
            
            # Step 2: Create reordered PDF
            self.logger.info("Step 2: Creating reordered PDF...")
            pages_reordered = self.create_reordered_pdf(input_path, output_pdf, page_mapping)
            
            # Step 3: Generate report
            self.logger.info("Step 3: Generating analysis report...")
            self.generate_report(report_path, page_mapping)
            
            return {
                'success': True,
                'input_path': str(input_path),
                'output_pdf': str(output_pdf),
                'report_path': str(report_path),
                'pages_reordered': pages_reordered,
                'page_mapping': page_mapping
            }
        
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_path': str(input_path)
            }
