#!/usr/bin/env python3
"""
Phase 4 Data Scraper for GraphRAG
Scrapes MSF (Ministry of Social and Family Development) and ComCare content.

Covers:
- ComCare assistance schemes (short/medium/long-term)
- Silver Support Scheme (elderly cash supplement)
- CDC Vouchers
- Family services (foster care, adoption, eldercare)
- Disability support
"""

import os
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Configuration
CORPUS_DIR = Path("corpus/raw")
DELAY_BETWEEN_REQUESTS = 1.5  # seconds

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# =============================================================================
# MSF URLs
# =============================================================================

MSF_URLS = {
    "comcare": [
        ("msf-comcare-overview", "https://www.msf.gov.sg/what-we-do/comcare"),
        # Old pages now use different structure - use media room articles instead
        ("msf-comcare-smta-eligibility", "https://www.msf.gov.sg/media-room/article/eligibility-for-comcare-short-to-medium-term-assistance"),
        ("msf-comcare-smta-online", "https://www.msf.gov.sg/media-room/article/online-applications-for-comcare-short-to-medium-term-assistance-(smta)"),
        ("msf-comcare-pchi-raise", "https://www.msf.gov.sg/media-room/article/raising-of-per-capita-income-benchmark-for-comcare-assistance"),
        ("msf-comcare-safety-nets", "https://www.msf.gov.sg/media-room/article/further-progress-in-strengthening-social-safety-nets"),
    ],
    "silver-support": [
        ("msf-silver-support", "https://www.msf.gov.sg/what-we-do/silver-support-scheme"),
        ("msf-silver-support-faq", "https://www.msf.gov.sg/what-we-do/silver-support-scheme/faqs"),
    ],
    "families": [
        ("msf-what-we-do", "https://www.msf.gov.sg/what-we-do"),
        ("msf-divorce-support", "https://www.msf.gov.sg/what-we-do/famatfsc/divorce-support"),
        ("msf-caregiver-support", "https://www.msf.gov.sg/what-we-do/swd/key-areas-of-support/recognition-support-for-caregivers"),
        ("msf-disability-caregiver", "https://www.msf.gov.sg/media-room/article/support-measures-for-caregivers-of-persons-with-disabilities"),
    ],
    "children-youth": [
        ("msf-children-protection", "https://www.msf.gov.sg/what-we-do/children-and-youth/child-protection"),
        ("msf-foster-care", "https://www.msf.gov.sg/what-we-do/children-and-youth/foster-care"),
        ("msf-adoption", "https://www.msf.gov.sg/what-we-do/children-and-youth/adoption"),
        ("msf-youth-services", "https://www.msf.gov.sg/what-we-do/children-and-youth/youth-services"),
    ],
    "disability": [
        ("msf-disability-overview", "https://www.msf.gov.sg/what-we-do/disability"),
        ("msf-disability-services", "https://www.msf.gov.sg/what-we-do/disability/disability-services"),
        ("msf-disability-assistive-tech", "https://www.msf.gov.sg/what-we-do/disability/assistive-technology-fund"),
    ],
    "elderly": [
        ("msf-elderly-overview", "https://www.msf.gov.sg/what-we-do/elderly"),
        ("msf-eldercare", "https://www.msf.gov.sg/what-we-do/elderly/eldercare-services"),
    ],
    "about": [
        ("msf-home", "https://www.msf.gov.sg/"),
    ],
}

# MSF PDF reports
MSF_PDFS = [
    ("msf-comcare-schemes-summary", "https://www.msf.gov.sg/docs/default-source/mediaroom-document/summary-of-comcare-lta-and-smta-schemes.pdf"),
    ("msf-lower-income-trends-2025", "https://www.msf.gov.sg/docs/default-source/research-data/supporting-lower-income-households-trends-report-2025.pdf"),
    ("msf-family-trends-2025", "https://www.msf.gov.sg/docs/default-source/research-data/family-trends-report-2025.pdf"),
    ("msf-cos-2025-factsheet", "https://www.msf.gov.sg/docs/default-source/cos-2025/msf-cos-2025---media-factsheet---a-singapore-made-for-all-families.pdf"),
]

# CDC Vouchers (community development councils)
CDC_URLS = {
    "vouchers": [
        ("cdc-vouchers-main", "https://vouchers.cdc.gov.sg/"),
        ("cdc-vouchers-residents", "https://vouchers.cdc.gov.sg/residents"),
        ("cdc-vouchers-merchants", "https://vouchers.cdc.gov.sg/merchants"),
        ("cdc-vouchers-faq", "https://vouchers.cdc.gov.sg/faq"),
    ],
}

# SG Enable (disability services)
SGENABLE_URLS = {
    "services": [
        ("sgenable-home", "https://www.sgenable.sg/"),
        ("sgenable-services", "https://www.sgenable.sg/pages/content.aspx?path=/get-support/"),
        ("sgenable-caregiver", "https://www.sgenable.sg/pages/content.aspx?path=/get-support/for-caregivers/"),
    ],
}

# Agency for Integrated Care (eldercare)
AIC_URLS = {
    "eldercare": [
        ("aic-home", "https://www.aic.sg/"),
        ("aic-care-services", "https://www.aic.sg/care-services"),
        ("aic-financial-assistance", "https://www.aic.sg/financial-assistance"),
        ("aic-caregiver-support", "https://www.aic.sg/caregiver-support"),
    ],
}

# NCSS (National Council of Social Service)
NCSS_URLS = {
    "assistance": [
        ("ncss-home", "https://www.ncss.gov.sg/"),
        ("ncss-social-services", "https://www.ncss.gov.sg/social-services"),
    ],
}

# Supporting Families Portal
SUPPORT_URLS = {
    "supportgowhere": [
        ("supportgowhere-home", "https://supportgowhere.life.gov.sg/"),
        ("supportgowhere-schemes", "https://supportgowhere.life.gov.sg/schemes"),
    ],
}


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF file."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return True
    except requests.RequestException as e:
        print(f"  ERROR downloading PDF: {e}")
        return False


def scrape_pdfs() -> int:
    """Download MSF PDF documents."""
    output_dir = CORPUS_DIR / "msf" / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Downloading PDFs ({len(MSF_PDFS)} files)")
    print("=" * 60)

    downloaded = 0
    for name, url in MSF_PDFS:
        output_file = output_dir / f"{name}.pdf"

        if output_file.exists():
            print(f"  Skipped (exists): {name}.pdf")
            downloaded += 1
            continue

        print(f"  Downloading PDF: {url[:70]}...")
        if download_pdf(url, output_file):
            print(f"  Saved: {name}.pdf")
            downloaded += 1
        else:
            print(f"  FAILED: {name}.pdf")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"  Completed: {downloaded}/{len(MSF_PDFS)} PDFs")
    return downloaded


def fetch_page(url: str) -> str | None:
    """Fetch a page and return its content."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  ERROR: {e}")
        return None


def html_to_markdown(html: str, url: str) -> str:
    """Convert HTML to clean markdown."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "iframe", "noscript"]):
        tag.decompose()

    # Try to find main content
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="content")
        or soup.find("div", id="content")
        or soup.find("div", class_="main-content")
        or soup.body
    )

    if main is None:
        main = soup

    # Convert to markdown
    markdown = md(str(main), heading_style="ATX", bullets="-")

    # Clean up
    lines = markdown.split("\n")
    cleaned = []
    for line in lines:
        line = line.rstrip()
        if line and not line.isspace():
            cleaned.append(line)

    # Add source URL
    result = f"Source: {url}\n\n" + "\n".join(cleaned)
    return result


def scrape_category(category: str, subcategory: str, urls: list[tuple[str, str]]) -> int:
    """Scrape a category of URLs."""
    output_dir = CORPUS_DIR / category / subcategory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Scraping: {category}/{subcategory} ({len(urls)} URLs)")
    print("=" * 60)

    scraped = 0
    for name, url in urls:
        output_file = output_dir / f"{name}.md"

        # Skip if already exists
        if output_file.exists():
            print(f"  Skipped (exists): {name}")
            scraped += 1
            continue

        print(f"  Fetching: {url[:70]}...")
        html = fetch_page(url)

        if html:
            markdown = html_to_markdown(html, url)
            if len(markdown) > 500:  # Minimum content threshold
                output_file.write_text(markdown, encoding="utf-8")
                print(f"  Saved: {name}.md")
                scraped += 1
            else:
                print(f"  SKIPPED (too little content): {name}")
        else:
            print(f"  FAILED: {name}")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"  Completed: {scraped}/{len(urls)} pages")
    return scraped


def create_directories():
    """Create directory structure for Phase 4 data."""
    dirs = []
    for subcategory in MSF_URLS.keys():
        dirs.append(CORPUS_DIR / "msf" / subcategory)
    for subcategory in CDC_URLS.keys():
        dirs.append(CORPUS_DIR / "msf" / subcategory)
    for subcategory in SGENABLE_URLS.keys():
        dirs.append(CORPUS_DIR / "msf" / "disability" / subcategory)
    for subcategory in AIC_URLS.keys():
        dirs.append(CORPUS_DIR / "msf" / "eldercare")
    for subcategory in NCSS_URLS.keys():
        dirs.append(CORPUS_DIR / "msf" / subcategory)
    for subcategory in SUPPORT_URLS.keys():
        dirs.append(CORPUS_DIR / "msf" / subcategory)

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")


def count_urls():
    """Count total URLs to scrape."""
    total_msf = sum(len(urls) for urls in MSF_URLS.values())
    total_cdc = sum(len(urls) for urls in CDC_URLS.values())
    total_sgenable = sum(len(urls) for urls in SGENABLE_URLS.values())
    total_aic = sum(len(urls) for urls in AIC_URLS.values())
    total_ncss = sum(len(urls) for urls in NCSS_URLS.values())
    total_support = sum(len(urls) for urls in SUPPORT_URLS.values())
    total_pdfs = len(MSF_PDFS)
    return total_msf, total_cdc, total_sgenable, total_aic, total_ncss, total_support, total_pdfs


def main():
    """Main scraping function."""
    total_msf, total_cdc, total_sgenable, total_aic, total_ncss, total_support, total_pdfs = count_urls()

    print("=" * 60)
    print("Phase 4 Data Scraper for GraphRAG")
    print("Scraping MSF/ComCare Social Services data")
    print("=" * 60)
    print(f"MSF URLs: {total_msf}")
    print(f"CDC Vouchers URLs: {total_cdc}")
    print(f"SG Enable URLs: {total_sgenable}")
    print(f"AIC URLs: {total_aic}")
    print(f"NCSS URLs: {total_ncss}")
    print(f"Support GoWhere URLs: {total_support}")
    print(f"MSF PDFs: {total_pdfs}")
    grand_total = total_msf + total_cdc + total_sgenable + total_aic + total_ncss + total_support + total_pdfs
    print(f"Total: {grand_total}")

    # Create directories
    print("\nCreating directories...")
    create_directories()

    # Scrape MSF data
    scraped_msf = 0
    for subcategory, urls in MSF_URLS.items():
        scraped_msf += scrape_category("msf", subcategory, urls)

    # Scrape CDC data (under msf)
    scraped_cdc = 0
    for subcategory, urls in CDC_URLS.items():
        scraped_cdc += scrape_category("msf", subcategory, urls)

    # Scrape SG Enable data
    scraped_sgenable = 0
    for subcategory, urls in SGENABLE_URLS.items():
        scraped_sgenable += scrape_category("msf/disability", subcategory, urls)

    # Scrape AIC data
    scraped_aic = 0
    for subcategory, urls in AIC_URLS.items():
        scraped_aic += scrape_category("msf", "eldercare", urls)

    # Scrape NCSS data
    scraped_ncss = 0
    for subcategory, urls in NCSS_URLS.items():
        scraped_ncss += scrape_category("msf", subcategory, urls)

    # Scrape Support GoWhere data
    scraped_support = 0
    for subcategory, urls in SUPPORT_URLS.items():
        scraped_support += scrape_category("msf", subcategory, urls)

    # Download PDFs
    scraped_pdfs = scrape_pdfs()

    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"MSF pages scraped: {scraped_msf}/{total_msf}")
    print(f"CDC Vouchers pages scraped: {scraped_cdc}/{total_cdc}")
    print(f"SG Enable pages scraped: {scraped_sgenable}/{total_sgenable}")
    print(f"AIC pages scraped: {scraped_aic}/{total_aic}")
    print(f"NCSS pages scraped: {scraped_ncss}/{total_ncss}")
    print(f"Support GoWhere pages scraped: {scraped_support}/{total_support}")
    print(f"MSF PDFs downloaded: {scraped_pdfs}/{total_pdfs}")
    total_scraped = scraped_msf + scraped_cdc + scraped_sgenable + scraped_aic + scraped_ncss + scraped_support + scraped_pdfs
    print(f"Total files: {total_scraped}/{grand_total}")
    print("\nNext steps:")
    print("1. Review the scraped data in corpus/raw/msf/")
    print("2. Rebuild the graph: python main.py build")
    print("3. Test queries: python main.py serve")


if __name__ == "__main__":
    main()
