\# Intelligent Invoice Extraction System



This project is an Intelligent Document Processing system for automated invoice and receipt data extraction.



\## Features



\- OCR text extraction

\- LayoutLM token classification

\- Rule-based fallback extraction

\- Field verification

\- Batch evaluation

\- Streamlit web interface

\- JSON and CSV export



\## Extracted Fields



\- Company name

\- Address

\- Date

\- Total amount



\## Evaluation Result



The system was evaluated on 341 invoice and receipt images.



| Field | Accuracy |

|---|---:|

| Company Name | 87.68% |

| Address | 91.50% |

| Date | 49.56% |

| Total Amount | 76.83% |

| Average Field Accuracy | 76.39% |

| All Fields Exact Match | 46.63% |



\## Run Web App



```bash

streamlit run invoice\_data/app.py


\## ## AI Assistance Disclosure

Claude was used only for Git/GitHub technical support, specifically to assist with pushing the completed project files to this repository. The project idea, system implementation, evaluation, and documentation were prepared by the author.


