import streamlit as st
import tempfile
import os
from pathlib import Path
import json
import torch
from methods import AdaptivePDFReorderer
import time

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Page Reorderer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_gpu_availability():
    """Check GPU availability and display status"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"🚀 GPU Available: {gpu_name} ({gpu_memory:.1f}GB)"
    else:
        return False, "⚠️ GPU not available - using CPU (slower processing)"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 PDF Page Reorderer</h1>
        <p>Automatically detect and reorder PDF pages based on page numbers using advanced OCR technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GPU Status
    gpu_available, gpu_status = check_gpu_availability()
    if gpu_available:
        st.markdown(f'<div class="success-box">{gpu_status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">{gpu_status}</div>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown("""
        **Step 1:** Upload your jumbled PDF file
        
        **Step 2:** AI analyzes each page using OCR to detect page numbers
        
        **Step 3:** Pages are automatically reordered based on detected numbers
        
        **Step 4:** Download your reordered PDF
        """)
        
        st.header("🔧 Settings")
        gpu_optimization = st.checkbox("Enable GPU Acceleration", value=gpu_available, disabled=not gpu_available)
        max_workers = st.slider("Max Workers", min_value=1, max_value=16, value=8)
        
        st.header("📚 Supported Features")
        st.markdown("""
        ✅ Multiple OCR detection methods
        
        ✅ Header/footer region analysis
        
        ✅ Multi-resolution processing
        
        ✅ Intelligent page number disambiguation
        
        ✅ Parallel processing for speed
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload PDF File")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF with jumbled pages that need reordering"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ File uploaded successfully!</strong><br>
                📄 <strong>Filename:</strong> {uploaded_file.name}<br>
                📊 <strong>Size:</strong> {file_size_mb:.1f} MB
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🔄 Process PDF", type="primary", use_container_width=True):
                process_pdf(uploaded_file, gpu_optimization, max_workers)
    
    with col2:
        st.header("📋 Instructions")
        st.markdown("""
        **Best Performance:**
        - Use Google Colab or Kaggle for GPU acceleration
        - Recommended for PDFs with clearly visible page numbers
        - Works best with standard academic papers, books, and reports
        
        **Tips:**
        - Ensure page numbers are clearly visible
        - Works with numbers in headers, footers, or margins
        - Supports various page number formats (1, -1-, Page 1, etc.)
        """)

def process_pdf(uploaded_file, gpu_optimization, max_workers):
    """Process the uploaded PDF file"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Initialize reorderer
        with st.spinner("🔧 Initializing PDF processor..."):
            reorderer = AdaptivePDFReorderer(
                gpu_optimization=gpu_optimization,
                max_workers=max_workers
            )
        
        st.success("✅ Processor initialized successfully!")
        
        # Process PDF
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Analyzing PDF structure and detecting page numbers...")
        progress_bar.progress(25)
        
        # Process the PDF
        result = reorderer.process_pdf(tmp_path, tempfile.gettempdir())
        
        progress_bar.progress(75)
        status_text.text("🔄 Creating reordered PDF...")
        
        if result['success']:
            progress_bar.progress(100)
            status_text.text("✅ Processing completed successfully!")
            
            # Display results
            st.markdown(f"""
            <div class="success-box">
                <h3>🎉 Success!</h3>
                <strong>Pages reordered:</strong> {result['pages_reordered']}<br>
                <strong>Original file:</strong> {uploaded_file.name}<br>
                <strong>Output file:</strong> {Path(result['output_pdf']).name}
            </div>
            """, unsafe_allow_html=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Reordered PDF download
                with open(result['output_pdf'], 'rb') as f:
                    pdf_data = f.read()
                st.download_button(
                    label="📄 Download Reordered PDF",
                    data=pdf_data,
                    file_name=Path(result['output_pdf']).name,
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col2:
                # Analysis report download
                with open(result['report_path'], 'r') as f:
                    report_data = f.read()
                st.download_button(
                    label="📊 Download Analysis Report",
                    data=report_data,
                    file_name=Path(result['report_path']).name,
                    mime="application/json",
                    use_container_width=True
                )
            
            # Display page mapping
            st.header("📋 Page Mapping Details")
            if result['page_mapping']:
                mapping_df = []
                for page_num, pdf_pos in sorted(result['page_mapping'].items()):
                    mapping_df.append({
                        "Detected Page Number": page_num,
                        "Original PDF Position": pdf_pos + 1,
                        "New Position": len(mapping_df) + 1
                    })
                
                st.dataframe(mapping_df, use_container_width=True)
            else:
                st.warning("No page mapping data available")
        
        else:
            progress_bar.progress(0)
            st.markdown(f"""
            <div class="error-box">
                <h3>❌ Processing Failed</h3>
                <strong>Error:</strong> {result['error']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Possible solutions:**
            - Ensure your PDF has visible page numbers
            - Try a different PDF file
            - Check if the page numbers are clear and readable
            """)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ Unexpected Error</h3>
            <strong>Error:</strong> {str(e)}
        </div>
        """, unsafe_allow_html=True)
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
