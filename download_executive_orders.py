#!/usr/bin/env python3
"""
Executive Order Downloader
Downloads executive orders from the Federal Register API
"""

import requests
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
import re

def clean_filename(filename):
    """Clean filename for safe file system storage"""
    # Remove/replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename[:200]  # Limit length

def download_executive_orders(limit=1000):
    """Download executive orders from Federal Register API"""
    
    # Create output directory
    output_dir = Path("data/executive_orders")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading executive orders to {output_dir}")
    
    # Federal Register API endpoint
    api_url = "https://www.federalregister.gov/api/v1/documents.json"
    
    # API parameters
    params = {
        "conditions[type][]": "PRESDOCU",
        "conditions[presidential_document_type][]": "executive_order",
        "per_page": limit,
        "fields[]": ["title", "pdf_url", "executive_order_number", "publication_date", "html_url"],
        "order": "newest"
    }
    
    try:
        print("Fetching executive order metadata from Federal Register API...")
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        documents = data.get('results', [])
        
        if not documents:
            print("No executive orders found.")
            return []
        
        print(f"Found {len(documents)} executive orders to download")
        
        downloaded_files = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown Title')
            eo_number = doc.get('executive_order_number', 'Unknown')
            pub_date = doc.get('publication_date', 'Unknown')
            pdf_url = doc.get('pdf_url')
            
            print(f"\n{i}/{len(documents)}: {title}")
            print(f"  EO Number: {eo_number}")
            print(f"  Date: {pub_date}")
            
            if not pdf_url:
                print(f"  ⚠️  No PDF URL available")
                continue
                
            try:
                # Create filename
                if eo_number != 'Unknown':
                    filename = f"EO_{eo_number}_{pub_date}_{clean_filename(title)}.pdf"
                else:
                    filename = f"EO_{pub_date}_{clean_filename(title)}.pdf"
                
                filepath = output_dir / filename
                
                # Skip if already exists
                if filepath.exists():
                    print(f"  ✅ Already exists: {filename}")
                    downloaded_files.append(str(filepath))
                    continue
                
                # Download PDF
                print(f"  📥 Downloading: {pdf_url}")
                pdf_response = requests.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()
                
                # Save PDF
                with open(filepath, 'wb') as f:
                    f.write(pdf_response.content)
                
                file_size = len(pdf_response.content) / 1024  # KB
                print(f"  ✅ Downloaded: {filename} ({file_size:.1f} KB)")
                downloaded_files.append(str(filepath))
                
                # Be respectful to the server
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error downloading {title}: {e}")
                continue
        
        print(f"\n🎉 Successfully downloaded {len(downloaded_files)} executive orders")
        return downloaded_files
        
    except Exception as e:
        print(f"❌ Error fetching from API: {e}")
        return []

if __name__ == "__main__":
    downloaded = download_executive_orders(limit=1000)
    
    if downloaded:
        print(f"\nDownloaded files:")
        for file_path in downloaded:
            print(f"  - {file_path}")
    else:
        print("\nNo files were downloaded.")