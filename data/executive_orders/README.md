# Executive Orders Directory

This directory is designed to contain PDF documents of US Presidential Executive Orders for regulatory analysis.

## Setup Instructions

1. **Download Executive Order PDFs**: Add Executive Order PDF files to this directory
2. **Naming Convention**: Use descriptive filenames like:
   - `EO-14028-Cybersecurity.pdf`
   - `EO-14110-AI-Safety.pdf` 
   - `EO-13636-Critical-Infrastructure.pdf`

## Supported File Types

- PDF documents (`.pdf`)
- The system will automatically process all PDF files in this directory
- Each PDF will be chunked and embedded into the Executive Order vector store

## Vector Store Management

- Vector store will be created automatically when PDFs are added
- Located at: `../exec_order_vector_store/`
- Regenerated when new PDFs are added

## Example Executive Orders to Include

Consider adding these key Executive Orders relevant to business compliance:

### Cybersecurity & Technology
- **EO 14028**: Improving the Nation's Cybersecurity (May 2021)
- **EO 13636**: Improving Critical Infrastructure Cybersecurity (February 2013)
- **EO 14110**: Safe, Secure, and Trustworthy AI (October 2023)

### Data Protection & Privacy
- **EO 14086**: Enhancing Safeguards for US Signals Intelligence (October 2022)
- Cross-border data transfer orders

### Federal Contracting
- **EO 14042**: Ensuring Adequate COVID Safety Protocols for Federal Contractors
- Supply chain security orders
- Federal contractor cybersecurity requirements

### Critical Infrastructure
- **EO 13800**: Strengthening the Cybersecurity of Federal Networks and Critical Infrastructure
- **EO 14017**: America's Supply Chains (February 2021)

## Usage

Once PDFs are added to this directory:

1. Restart the RegTech AI system
2. The Executive Order vector store will be rebuilt automatically
3. Questions about US regulatory compliance will now include Executive Order analysis
4. The system will classify questions for Executive Order applicability

## Notes

- The system supports multiple Executive Orders simultaneously
- Each Executive Order is processed and chunked separately
- Metadata includes document name and framework type
- Executive Orders complement existing EU regulatory frameworks (GDPR, NIS2, DORA, CER)