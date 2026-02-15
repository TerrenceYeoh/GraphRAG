"""
Phase 3 Data Scraper for GraphRAG
Scrapes Business Grants, SkillsFuture, and Education data from Singapore government websites.

Categories:
- Grants: Enterprise SG, GoBusiness, IMDA digital programs
- SkillsFuture: Credit, Career Transition/Conversion Programmes, Subsidies
- Education: MOE Financial Assistance, PSEA, Bursaries, Tertiary

Usage:
    python scripts/scrape_phase3.py
"""

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Base directories
CORPUS_DIR = Path("corpus/raw")

# Request headers to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests


# ============================================================================
# COMPREHENSIVE URLs FOR PHASE 3
# ============================================================================

GRANTS_URLS = {
    "gobusiness": [
        ("gobusiness-grants-overview", "https://www.gobusiness.gov.sg/gov-assist/grants/"),
        ("gobusiness-grants-portal", "https://grants.gobusiness.gov.sg/"),
        ("gobusiness-psg-main", "https://www.gobusiness.gov.sg/productivity-solutions-grant/"),
        ("gobusiness-psg-solutions", "https://www.gobusiness.gov.sg/productivity-solutions-grant/all-psg-solutions/"),
        ("gobusiness-psg-faq", "https://grants.gobusiness.gov.sg/faq/psg"),
        ("gobusiness-claims-faq", "https://www.gobusiness.gov.sg/business-grants-portal-faq/claims/"),
        ("gobusiness-eeg", "https://www.gobusiness.gov.sg/business-grants-portal-faq/eeg/"),
        # Business licensing and starting
        ("gobusiness-home", "https://www.gobusiness.gov.sg/"),
        ("gobusiness-licences", "https://www.gobusiness.gov.sg/licences/"),
        ("gobusiness-licences-permits", "https://www.gobusiness.gov.sg/start-a-business/get-licences-permits/"),
        ("gobusiness-licence-directory", "https://licensing.gobusiness.gov.sg/licence-directory"),
        ("gobusiness-licence-by-sector", "https://www.gobusiness.gov.sg/licences/find-licence-by-sector/"),
        ("gobusiness-licence-by-industry", "https://www.gobusiness.gov.sg/licences/find-licence-by-industry/"),
        ("gobusiness-register-business", "https://www.gobusiness.gov.sg/start-a-business/register-your-business/"),
        ("gobusiness-home-based", "https://www.gobusiness.gov.sg/start-a-business/faqs/home-based-businesses"),
        ("gobusiness-start-business", "https://www.gobusiness.gov.sg/start-a-business/"),
    ],
    "imda": [
        ("imda-sme-digital", "https://www.imda.gov.sg/how-we-can-help/smes-go-digital"),
        ("imda-advanced-digital", "https://www.imda.gov.sg/how-we-can-help/smes-go-digital/advanced-digital-solutions"),
        ("imda-business", "https://www.imda.gov.sg/business"),
        ("imda-digital-leaders", "https://www.imda.gov.sg/how-we-can-help/smes-go-digital/digital-leaders-programme"),
        ("imda-cyber-essentials", "https://www.imda.gov.sg/how-we-can-help/smes-go-digital/cyber-essentials-and-lite"),
        ("imda-home", "https://www.imda.gov.sg/"),
        ("imda-how-we-help", "https://www.imda.gov.sg/how-we-can-help"),
    ],
    "startup-sg": [
        ("startupsg-home", "https://www.startupsg.gov.sg/"),
        # Note: Most Startup SG content is on Enterprise SG which blocks scraping
    ],
}

SKILLSFUTURE_URLS = {
    "credit": [
        ("sf-individuals", "https://www.skillsfuture.gov.sg/initiatives/individuals"),
        ("sf-credit", "https://www.skillsfuture.gov.sg/initiatives/mid-career/credit"),
        ("sf-level-up", "https://www.skillsfuture.gov.sg/level-up-programme"),
        ("sf-home", "https://www.skillsfuture.gov.sg/"),
        ("sf-enhanced-subsidy", "https://www.skillsfuture.gov.sg/initiatives/individuals/enhancedsubsidy"),
        ("sf-enhanced-subsidy-faq", "https://www.skillsfuture.gov.sg/enhancedsubsidy-faq"),
    ],
    "career-transition": [
        ("sf-career-transition", "https://www.skillsfuture.gov.sg/careertransition"),
        ("sf-sctp", "https://www.skillsfuture.gov.sg/sctp"),
        ("sf-work-study", "https://www.skillsfuture.gov.sg/workstudy"),
    ],
    "enterprise": [
        ("sf-sfec", "https://www.skillsfuture.gov.sg/sfec"),
        ("sf-employers", "https://www.skillsfuture.gov.sg/initiatives/employers"),
        ("sf-etss", "https://www.skillsfuture.gov.sg/initiatives/employers/enhanced-training-support-for-smes"),
    ],
}

WSG_URLS = {
    "ccp": [
        ("wsg-ccp-employers", "https://www.wsg.gov.sg/home/employers-industry-partners/workforce-development-job-redesign/career-conversion-programmes-employers"),
        ("wsg-ccp-individuals", "https://www.wsg.gov.sg/home/individuals/attachment-placement-programmes/career-conversion-programmes-for-individuals"),
        ("wsg-ccp-sme", "https://www.wsg.gov.sg/home/campaigns/ccp-sme-professionals"),
        ("wsg-ccp-business-case", "https://www.wsg.gov.sg/home/campaigns/the-business-case-for-career-conversion-programme"),
    ],
    "jobseekers": [
        ("wsg-individuals", "https://www.wsg.gov.sg/home/individuals"),
        ("wsg-job-seekers-main", "https://www.wsg.gov.sg/home/individuals/attachment-placement-programmes"),
        ("wsg-jobseeker-support", "https://www.wsg.gov.sg/home/individuals/jobseeker-support"),
        ("wsg-career-trial-jobseekers", "https://www.wsg.gov.sg/home/individuals/attachment-placement-programmes/career-trial-for-jobseekers"),
    ],
    "employers": [
        ("wsg-employers", "https://www.wsg.gov.sg/home/employers-industry-partners"),
        ("wsg-workforce-dev", "https://www.wsg.gov.sg/home/employers-industry-partners/workforce-development-job-redesign"),
        ("wsg-mid-career-pathways", "https://www.wsg.gov.sg/home/employers-industry-partners/workforce-development-job-redesign/mid-career-pathways-programme-for-host-organisations"),
        ("wsg-career-trial-employers", "https://www.wsg.gov.sg/home/employers-industry-partners/workforce-development-job-redesign/career-trial-for-employers"),
    ],
    "media": [
        ("wsg-jobseeker-support-press", "https://www.wsg.gov.sg/home/media-room/media-releases-speeches/skillsfuture-jobseeker-support-scheme-to-benefit-around-60-000-singaporean-residents-per-year"),
    ],
}

EDUCATION_URLS = {
    "financial-assistance": [
        ("moe-financial-matters", "https://www.moe.gov.sg/financial-matters"),
        ("moe-financial-assistance", "https://www.moe.gov.sg/financial-matters/financial-assistance"),
        ("moe-fas-2025-press", "https://www.moe.gov.sg/news/press-releases/20251016-moe-financial-assistance-schemes-to-benefit-an-additional-31000-students"),
        ("moe-higher-ed-cost", "https://www.moe.gov.sg/news/edtalks/heading-for-higher-education-and-worried-about-the-cost"),
        ("moe-fas-psei", "https://www.moe.gov.sg/financial-matters/financial-assistance/financial-assistance-information-for-pseis"),
    ],
    "psea": [
        ("moe-psea-overview", "https://www.moe.gov.sg/financial-matters/psea/overview"),
        ("moe-psea-main", "https://www.moe.gov.sg/financial-matters/psea"),
        ("moe-psea-funds", "https://www.moe.gov.sg/financial-matters/psea/funds-balance-and-usage"),
        ("cpf-psea", "https://www.cpf.gov.sg/service/article/what-is-the-moe-post-secondary-education-account-psea"),
    ],
    "awards-scholarships": [
        ("moe-awards", "https://www.moe.gov.sg/financial-matters/awards-scholarships"),
        ("moe-edusave", "https://www.moe.gov.sg/financial-matters/awards-scholarships/edusave-awards"),
        ("moe-edusave-account", "https://www.moe.gov.sg/financial-matters/edusave-account"),
    ],
    "tertiary": [
        ("moe-post-secondary", "https://www.moe.gov.sg/post-secondary"),
        ("moe-tuition-grant", "https://www.moe.gov.sg/financial-matters/tuition-grant-scheme"),
        ("moe-tuition-grant-overview", "https://www.moe.gov.sg/financial-matters/tuition-grant-scheme/overview"),
    ],
    "school-fees": [
        ("moe-fees-overview", "https://www.moe.gov.sg/financial-matters/fees"),
    ],
}

# Polytechnic financial aid
POLYTECHNIC_URLS = {
    "singapore-poly": [
        ("sp-financial-aid", "https://www.sp.edu.sg/admissions/financial-aid"),
        ("sp-financial-schemes", "https://www.sp.edu.sg/admissions/financial-schemes"),
        ("sp-financial-assistance", "https://www.sp.edu.sg/admissions/financial-aid/financial-assistance"),
        ("sp-govt-bursaries", "https://www.sp.edu.sg/admissions/financial-aid/government-bursaries"),
        ("sp-external-bursaries", "https://www.sp.edu.sg/admissions/financial-aid/external-bursaries"),
    ],
    "ngee-ann-poly": [
        ("np-financial-aid", "https://www.np.edu.sg/admissions-enrolment/guide-for-prospective-students/aid"),
    ],
    "temasek-poly": [
        ("tp-fees-financial", "https://www.tp.edu.sg/admissions-and-finance/fees-financial-matters.html"),
    ],
    "republic-poly": [
        ("rp-financial-assistance", "https://www.rp.edu.sg/admissions/financial-assistance"),
    ],
    "nanyang-poly": [
        ("nyp-financial-assistance", "https://www.nyp.edu.sg/admissions/financial-assistance.html"),
    ],
}

# University financial aid
UNIVERSITY_URLS = {
    "nus": [
        ("nus-financial-aid", "https://www.nus.edu.sg/oam/financial-aid"),
        ("nus-bursaries", "https://www.nus.edu.sg/oam/financial-aid/bursaries"),
    ],
    "ntu": [
        ("ntu-financial-aid", "https://www.ntu.edu.sg/admissions/undergraduate/financial-matters/financial-aid"),
        ("ntu-psea", "https://www.ntu.edu.sg/admissions/undergraduate/financial-matters/financial-aid/post-secondary-education-account-(psea)"),
    ],
    "smu": [
        ("smu-financial-aid", "https://admissions.smu.edu.sg/financial-matters/financial-aid"),
    ],
    "sutd": [
        ("sutd-financial-aid", "https://www.sutd.edu.sg/Admissions/Undergraduate/Financing-Your-Studies/Financial-Aid"),
    ],
    "sit": [
        ("sit-financial-aid", "https://www.singaporetech.edu.sg/admissions/financial-aid"),
        ("sit-financial-faq", "https://www.singaporetech.edu.sg/admissions/financial-aid/financial-assistance-faqs"),
    ],
    "suss": [
        ("suss-financial-aid", "https://www.suss.edu.sg/part-time-undergraduate/admissions/financial-aid"),
    ],
}

# ITE financial aid
ITE_URLS = {
    "ite": [
        ("ite-financial-assistance", "https://www.ite.edu.sg/admissions/financial-matters"),
    ],
}

# PDF documents to download
PDF_URLS = [
    # WSG CCPs
    ("skillsfuture", "ccp", "ccp-factsheet-2025", "https://www.wsg.gov.sg/docs/default-source/programme/career-conversion-programmes/1_ccp_factsheet_3mar2025.pdf"),
    ("skillsfuture", "ccp", "ccp-faqs-2025", "https://www.wsg.gov.sg/docs/default-source/programme/career-conversion-programmes/2_ccps_faqs_3mar2025.pdf"),
    # SkillsFuture
    ("skillsfuture", "enterprise", "sfec-faqs", "https://www.skillsfuture.gov.sg/docs/default-source/initiatives/sfec-skillsfuture-enterprise-credit/faqs-for-sfec_22-mar-202464df1204-dc84-4f77-81df-d7226348435b.pdf"),
    ("skillsfuture", "career-transition", "sctp-faqs", "https://www.skillsfuture.gov.sg/docs/default-source/initiatives/sctp/faqs-for-sctp_public_v3-(final).pdf"),
    ("skillsfuture", "credit", "midcareer-support-faqs", "https://www.skillsfuture.gov.sg/docs/default-source/initiatives/faqs-on-skillsfuture-mid-career-support-package.pdf"),
    # MOE
    ("education", "awards-scholarships", "edusave-report-2024", "https://www.moe.gov.sg/-/media/files/financial-matters/edusave-report-2024.pdf"),
]


def create_directories():
    """Create directory structure for Phase 3 data."""
    dirs = []
    for subcategory in GRANTS_URLS.keys():
        dirs.append(CORPUS_DIR / "grants" / subcategory)
    for subcategory in SKILLSFUTURE_URLS.keys():
        dirs.append(CORPUS_DIR / "skillsfuture" / subcategory)
    for subcategory in WSG_URLS.keys():
        dirs.append(CORPUS_DIR / "skillsfuture" / subcategory)  # WSG under skillsfuture
    for subcategory in EDUCATION_URLS.keys():
        dirs.append(CORPUS_DIR / "education" / subcategory)
    for subcategory in POLYTECHNIC_URLS.keys():
        dirs.append(CORPUS_DIR / "education" / "polytechnic" / subcategory)
    for subcategory in UNIVERSITY_URLS.keys():
        dirs.append(CORPUS_DIR / "education" / "university" / subcategory)
    for subcategory in ITE_URLS.keys():
        dirs.append(CORPUS_DIR / "education" / subcategory)

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove common navigation elements
    text = re.sub(r"Skip to main content", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Back to top", "", text, flags=re.IGNORECASE)
    return text.strip()


def html_to_markdown(soup: BeautifulSoup, url: str) -> str:
    """Convert HTML to simple markdown format."""
    # Remove script, style, nav, footer elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Find main content area
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(class_=re.compile(r"content|main|article", re.I))
        or soup.find("body")
    )

    if not main_content:
        main_content = soup

    lines = []
    lines.append(f"# {soup.title.string if soup.title else 'Untitled'}")
    lines.append(f"\nSource: {url}\n")

    # Extract text with basic structure
    for element in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = element.get_text(strip=True)
        if not text:
            continue

        if element.name == "h1":
            lines.append(f"\n# {text}\n")
        elif element.name == "h2":
            lines.append(f"\n## {text}\n")
        elif element.name == "h3":
            lines.append(f"\n### {text}\n")
        elif element.name == "h4":
            lines.append(f"\n#### {text}\n")
        elif element.name == "li":
            lines.append(f"- {text}")
        elif element.name in ["td", "th"]:
            lines.append(f"| {text} ")
        else:
            lines.append(text)

    return clean_text("\n".join(lines))


def fetch_webpage(url: str) -> str | None:
    """Fetch a webpage and return its content as markdown."""
    try:
        print(f"  Fetching: {url[:80]}...")
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        markdown = html_to_markdown(soup, url)

        time.sleep(REQUEST_DELAY)
        return markdown

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        return None


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF file."""
    try:
        print(f"  Downloading PDF: {url[:60]}...")
        response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        time.sleep(REQUEST_DELAY)
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ERROR downloading PDF: {e}")
        return False


def scrape_category(category: str, subcategory: str, urls: list[tuple[str, str]]):
    """Scrape all URLs for a category."""
    output_dir = CORPUS_DIR / category / subcategory

    print(f"\n{'='*60}")
    print(f"Scraping: {category}/{subcategory} ({len(urls)} URLs)")
    print(f"{'='*60}")

    success_count = 0
    for name, url in urls:
        output_path = output_dir / f"{name}.md"

        # Skip if already exists
        if output_path.exists():
            print(f"  Skipped (exists): {name}")
            success_count += 1
            continue

        content = fetch_webpage(url)
        if content:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Saved: {output_path.name}")
            success_count += 1
        else:
            print(f"  FAILED: {name}")

    print(f"  Completed: {success_count}/{len(urls)} pages")
    return success_count


def scrape_pdfs():
    """Download all PDF documents."""
    print(f"\n{'='*60}")
    print(f"Downloading PDFs ({len(PDF_URLS)} files)")
    print(f"{'='*60}")

    success_count = 0
    for category, subcategory, name, url in PDF_URLS:
        output_dir = CORPUS_DIR / category / subcategory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.pdf"

        # Skip if already exists
        if output_path.exists():
            print(f"  Skipped (exists): {name}.pdf")
            success_count += 1
            continue

        if download_pdf(url, output_path):
            print(f"  Saved: {output_path}")
            success_count += 1

    print(f"  Completed: {success_count}/{len(PDF_URLS)} PDFs")
    return success_count


def count_urls():
    """Count total URLs to scrape."""
    total_grants = sum(len(urls) for urls in GRANTS_URLS.values())
    total_sf = sum(len(urls) for urls in SKILLSFUTURE_URLS.values())
    total_wsg = sum(len(urls) for urls in WSG_URLS.values())
    total_edu = sum(len(urls) for urls in EDUCATION_URLS.values())
    total_poly = sum(len(urls) for urls in POLYTECHNIC_URLS.values())
    total_uni = sum(len(urls) for urls in UNIVERSITY_URLS.values())
    total_ite = sum(len(urls) for urls in ITE_URLS.values())
    total_pdfs = len(PDF_URLS)
    return total_grants, total_sf, total_wsg, total_edu, total_poly, total_uni, total_ite, total_pdfs


def main():
    """Main scraping function."""
    total_grants, total_sf, total_wsg, total_edu, total_poly, total_uni, total_ite, total_pdfs = count_urls()

    print("=" * 60)
    print("Phase 3 Data Scraper for GraphRAG (Expanded)")
    print("Scraping Business Grants, SkillsFuture, and Education data")
    print("=" * 60)
    print(f"Grants URLs: {total_grants}")
    print(f"SkillsFuture URLs: {total_sf}")
    print(f"WSG URLs: {total_wsg}")
    print(f"MOE Education URLs: {total_edu}")
    print(f"Polytechnic URLs: {total_poly}")
    print(f"University URLs: {total_uni}")
    print(f"ITE URLs: {total_ite}")
    print(f"PDFs: {total_pdfs}")
    grand_total = total_grants + total_sf + total_wsg + total_edu + total_poly + total_uni + total_ite + total_pdfs
    print(f"Total: {grand_total}")

    # Create directories
    print("\nCreating directories...")
    create_directories()

    # Scrape Grants data
    scraped_grants = 0
    for subcategory, urls in GRANTS_URLS.items():
        scraped_grants += scrape_category("grants", subcategory, urls)

    # Scrape SkillsFuture data
    scraped_sf = 0
    for subcategory, urls in SKILLSFUTURE_URLS.items():
        scraped_sf += scrape_category("skillsfuture", subcategory, urls)

    # Scrape WSG data (under skillsfuture category)
    scraped_wsg = 0
    for subcategory, urls in WSG_URLS.items():
        scraped_wsg += scrape_category("skillsfuture", subcategory, urls)

    # Scrape Education data
    scraped_edu = 0
    for subcategory, urls in EDUCATION_URLS.items():
        scraped_edu += scrape_category("education", subcategory, urls)

    # Scrape Polytechnic data
    scraped_poly = 0
    for subcategory, urls in POLYTECHNIC_URLS.items():
        scraped_poly += scrape_category("education/polytechnic", subcategory, urls)

    # Scrape University data
    scraped_uni = 0
    for subcategory, urls in UNIVERSITY_URLS.items():
        scraped_uni += scrape_category("education/university", subcategory, urls)

    # Scrape ITE data
    scraped_ite = 0
    for subcategory, urls in ITE_URLS.items():
        scraped_ite += scrape_category("education", subcategory, urls)

    # Download PDFs
    scraped_pdfs = scrape_pdfs()

    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Grants pages scraped: {scraped_grants}/{total_grants}")
    print(f"SkillsFuture pages scraped: {scraped_sf}/{total_sf}")
    print(f"WSG pages scraped: {scraped_wsg}/{total_wsg}")
    print(f"MOE Education pages scraped: {scraped_edu}/{total_edu}")
    print(f"Polytechnic pages scraped: {scraped_poly}/{total_poly}")
    print(f"University pages scraped: {scraped_uni}/{total_uni}")
    print(f"ITE pages scraped: {scraped_ite}/{total_ite}")
    print(f"PDFs downloaded: {scraped_pdfs}/{total_pdfs}")
    total_scraped = scraped_grants + scraped_sf + scraped_wsg + scraped_edu + scraped_poly + scraped_uni + scraped_ite + scraped_pdfs
    print(f"Total files: {total_scraped}/{grand_total}")
    print("\nNext steps:")
    print("1. Review the scraped data in corpus/raw/grants, corpus/raw/skillsfuture, corpus/raw/education")
    print("2. Rebuild the graph: python main.py rebuild")
    print("3. Test queries: python main.py serve")


if __name__ == "__main__":
    main()
