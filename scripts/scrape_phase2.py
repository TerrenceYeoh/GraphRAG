"""
Phase 2 Data Scraper for GraphRAG (Enhanced)
Scrapes MOH (Healthcare) and MOM (Employment) data from Singapore government websites.

Usage:
    python scripts/scrape_phase2.py
"""

import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

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
# COMPREHENSIVE URLs FOR PHASE 2
# ============================================================================

MOH_URLS = {
    "medisave": [
        ("cpf-medisave-overview", "https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings"),
        ("cpf-medisave-hospitalisation", "https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings/using-medisave-for-hospitalisation"),
        ("cpf-medisave-outpatient", "https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings/using-medisave-for-outpatient-treatments"),
        ("cpf-medisave-topup", "https://www.cpf.gov.sg/member/growing-your-savings/saving-more-with-cpf/top-up-your-medisave-savings"),
        ("cpf-basic-healthcare-sum", "https://www.cpf.gov.sg/service/article/what-is-the-basic-healthcare-sum"),
        ("cpf-medisave-max", "https://www.cpf.gov.sg/service/article/is-there-a-maximum-amount-that-i-can-save-in-my-medisave-account"),
        ("cpf-healthcare-changes-2025", "https://www.cpf.gov.sg/member/infohub/educational-resources/cpf-changes-announced-in-budget-and-cos-2025-and-what-it-means-for-you"),
        ("cpf-healthcare-updates", "https://www.cpf.gov.sg/member/infohub/educational-resources/healthcare-in-singapore-updates-you-might-have-missed"),
        ("moh-medisave-ltc", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medisave/long-term-care/"),
    ],
    "medishield-life": [
        ("moh-medishield-life", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medishield-life/medishield-life/"),
        ("moh-medishield-premiums", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medishield-life/medishield-life-premiums-and-subsidies/"),
        ("moh-medishield-premium-tables", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medishield-life/medishield-life-premium-and-subsidy-tables/"),
        ("moh-managing-bills", "https://www.moh.gov.sg/managing-expenses/keeping-healthcare-affordable/managing-medical-bills/"),
        ("moh-healthcare-schemes", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/"),
        ("moh-healthcare-affordable", "https://www.moh.gov.sg/cost-financing/healthcare-schemes-subsidies"),
    ],
    "integrated-shield-plans": [
        ("moh-isp-about", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/integrated-shield-plans/about-integrated-shield-plans/"),
        ("moh-isp-main", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/integrated-shield-plans/"),
        ("moh-isp-comparison", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/integrated-shield-plans/comparision-of-integrated-shield-plans/"),
    ],
    "careshield-eldershield": [
        ("moh-careshield-main", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/careshield-life/"),
        ("moh-careshield-life", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/careshield-life/careshield-life/"),
        ("moh-careshield-2025-review", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/careshield-life/careshield-life-2025-review/"),
        ("moh-careshield-2025-faqs", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/careshield-life/careshield-life-2025-review-faqs/"),
        ("moh-careshield-act", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/careshield-life/careshield-life-and-long-term-care-act/"),
        ("moh-careshield-supplements", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/careshield-life/careshield-life-and-eldershield-supplements/"),
        ("cpf-careshield", "https://www.cpf.gov.sg/member/healthcare-financing/careshield-life"),
        ("cpf-careshield-planning", "https://www.cpf.gov.sg/member/infohub/educational-resources/careshieldlife-and-why-you-should-start-long-term-care-planning"),
        ("cpf-careshield-2025", "https://www.cpf.gov.sg/member/infohub/news/cpf-related-announcements/careshieldlife-2025-review"),
    ],
    "chas": [
        ("moh-chas", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/chas/"),
        ("chas-subsidies", "https://www.chas.sg/chas-subsidies"),
        ("chas-chronic-tier", "https://www.chas.sg/healthier-sg-chronic-tier-subsidies"),
    ],
    "pioneer-merdeka": [
        ("moh-pioneer-generation", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/pioneer-generation-package/"),
        ("moh-merdeka-generation", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/merdeka-generation-package/"),
    ],
    "medifund": [
        ("moh-medifund", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medifund/"),
    ],
    "long-term-care": [
        ("moh-ltc-overview", "https://www.moh.gov.sg/managing-expenses/keeping-healthcare-affordable/long-term-care/"),
        ("moh-ltc-residential-subsidies", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/subsidies-for-residential-long-term-care-services/"),
        ("moh-ltc-nonresidential-subsidies", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/subsidy-framework-for-non-residential-long-term-care-services"),
        ("moh-iltc-services", "https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/intermediate-and-long-term-care-services/"),
    ],
    "healthier-sg": [
        ("healthiersg-main", "https://www.healthiersg.gov.sg/"),
        ("healthiersg-benefits", "https://www.healthiersg.gov.sg/enrolment/benefits/"),
        ("healthiersg-chronic-tier", "https://www.healthiersg.gov.sg/enrolment/healthier-sg-chronic-tier/about/"),
        ("moh-healthiersg-vaccinations", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/healthier-sg-vaccinations/"),
        ("moh-childhood-vaccinations", "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/childhood-developmental-screening-and-childhood-vaccinations/"),
    ],
}

MOM_URLS = {
    "work-passes": [
        ("mom-passes-overview", "https://www.mom.gov.sg/passes-and-permits"),
        ("mom-employment-pass", "https://www.mom.gov.sg/passes-and-permits/employment-pass"),
        ("mom-ep-eligibility", "https://www.mom.gov.sg/passes-and-permits/employment-pass/eligibility"),
        ("mom-ep-apply", "https://www.mom.gov.sg/passes-and-permits/employment-pass/apply-for-a-pass"),
        ("mom-ep-consider-fairly", "https://www.mom.gov.sg/passes-and-permits/employment-pass/consider-all-candidates-fairly"),
        ("mom-spass", "https://www.mom.gov.sg/passes-and-permits/s-pass"),
        ("mom-spass-eligibility", "https://www.mom.gov.sg/passes-and-permits/s-pass/eligibility"),
        ("mom-spass-quota-levy", "https://www.mom.gov.sg/passes-and-permits/s-pass/quota-and-levy"),
        ("mom-spass-levy-requirements", "https://www.mom.gov.sg/passes-and-permits/s-pass/quota-and-levy/levy-and-quota-requirements"),
        ("mom-one-pass", "https://www.mom.gov.sg/passes-and-permits/overseas-networks-expertise-pass"),
        ("mom-one-pass-eligibility", "https://www.mom.gov.sg/passes-and-permits/overseas-networks-expertise-pass/eligibility"),
        ("mom-pep", "https://www.mom.gov.sg/passes-and-permits/personalised-employment-pass"),
        ("mom-entrepass", "https://www.mom.gov.sg/passes-and-permits/entrepass"),
        ("mom-training-ep", "https://www.mom.gov.sg/passes-and-permits/training-employment-pass"),
        ("mom-work-holiday", "https://www.mom.gov.sg/passes-and-permits/work-holiday-programme"),
    ],
    "compass-fcf": [
        ("mom-compass", "https://www.mom.gov.sg/passes-and-permits/employment-pass/upcoming-changes-to-employment-pass-eligibility/complementarity-assessment-framework-compass"),
        ("mom-compass-salary", "https://www.mom.gov.sg/passes-and-permits/employment-pass/eligibility/compass-c1-salary-benchmarks"),
        ("mom-compass-skills", "https://www.mom.gov.sg/passes-and-permits/employment-pass/eligibility/compass-c5-skills-bonus-shortage-occupation-list-sol"),
        ("mom-fcf", "https://www.mom.gov.sg/employment-practices/fair-consideration-framework"),
    ],
    "employment-act": [
        ("mom-employment-practices", "https://www.mom.gov.sg/employment-practices"),
        ("mom-leave", "https://www.mom.gov.sg/employment-practices/leave"),
        ("mom-annual-leave", "https://www.mom.gov.sg/employment-practices/leave/annual-leave"),
        ("mom-annual-leave-eligibility", "https://www.mom.gov.sg/employment-practices/leave/annual-leave/eligibility-and-entitlement"),
        ("mom-annual-leave-special", "https://www.mom.gov.sg/employment-practices/leave/annual-leave/special-situations"),
        ("mom-sick-leave", "https://www.mom.gov.sg/employment-practices/leave/sick-leave/eligibility-and-entitlement"),
        ("mom-maternity-leave", "https://www.mom.gov.sg/employment-practices/leave/maternity-leave/eligibility-and-entitlement"),
        ("mom-paternity-leave", "https://www.mom.gov.sg/employment-practices/leave/paternity-leave"),
        ("mom-childcare-leave", "https://www.mom.gov.sg/employment-practices/leave/childcare-leave/eligibility-and-entitlement"),
        ("mom-shared-parental-leave", "https://www.mom.gov.sg/employment-practices/leave/shared-parental-leave"),
        ("mom-parttime-leave", "https://www.mom.gov.sg/employment-practices/part-time-employment/leave"),
        ("mom-termination", "https://www.mom.gov.sg/employment-practices/termination-of-employment"),
        ("mom-termination-notice", "https://www.mom.gov.sg/employment-practices/termination-of-employment/termination-with-notice"),
    ],
    "foreign-workers": [
        ("mom-work-permit", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker"),
        ("mom-fw-levy", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/foreign-worker-levy"),
        ("mom-fw-levy-what", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/foreign-worker-levy/what-is-the-foreign-worker-levy"),
        ("mom-fw-quota-calc", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/foreign-worker-levy/calculate-foreign-worker-quota"),
        ("mom-construction", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/sector-specific-rules/construction-sector-requirements"),
        ("mom-mdw", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-domestic-worker"),
        ("mom-mdw-levy", "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-domestic-worker/foreign-domestic-worker-levy/paying-levy"),
    ],
    "workplace-safety": [
        ("mom-wsh-main", "https://www.mom.gov.sg/workplace-safety-and-health"),
        ("mom-wsh-legislation", "https://www.mom.gov.sg/legislation/workplace-safety-and-health"),
        ("mom-wsh-act", "https://www.mom.gov.sg/workplace-safety-and-health/workplace-safety-and-health-act"),
        ("mom-wsh-act-coverage", "https://www.mom.gov.sg/workplace-safety-and-health/workplace-safety-and-health-act/what-it-covers"),
        ("mom-wsh-training", "https://www.mom.gov.sg/workplace-safety-and-health/workplace-safety-and-health-training"),
        ("mom-wsh-best-practices", "https://www.mom.gov.sg/workplace-safety-and-health/wsh-best-practices"),
        ("mom-wsh-officer", "https://www.mom.gov.sg/workplace-safety-and-health/wsh-professionals/workplace-safety-and-health-officer"),
    ],
    "work-injury": [
        ("mom-wica-main", "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation"),
        ("mom-wica-what", "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/what-is-wica"),
        ("mom-wica-covered", "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/who-is-covered"),
        ("mom-wica-claims", "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/eligible-claims"),
        ("mom-wica-types", "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/types-of-compensation"),
        ("mom-wica-vs-common-law", "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/wica-versus-common-law"),
        ("mom-platform-workers-wica", "https://www.mom.gov.sg/employment-practices/platform-workers-act/work-injury-compensation-for-platform-workers"),
    ],
    "retirement-age": [
        ("mom-re-employment", "https://www.mom.gov.sg/employment-practices/re-employment"),
    ],
}

# PDF documents to download
PDF_URLS = [
    ("moh", "medisave", "medisave-withdrawal-limits-2025", "https://www.cpf.gov.sg/content/dam/web/member/healthcare/documents/MediSave%20Withdrawal%20Limits_1%20April%202025.pdf"),
    ("mom", "work-passes", "cos-2025-foreign-workforce-policies", "https://www.mom.gov.sg/-/media/mom/documents/budget2025/cos-2025-factsheet-on-foreign-workforce-policies.pdf"),
    ("mom", "compass-fcf", "compass-booklet", "https://www.mom.gov.sg/-/media/mom/documents/work-passes-and-permits/compass/compass-booklet.pdf"),
    ("mom", "compass-fcf", "ep-framework-updates-2022", "https://www.mom.gov.sg/-/media/mom/documents/budget2022/updates-to-ep-framework.pdf"),
    ("mom", "work-injury", "wica-guide-employers", "https://www.mom.gov.sg/-/media/mom/documents/safety-health/publications/wica/wic-guide-for-employers-english.pdf"),
    ("mom", "foreign-workers", "quota-levy-guide", "https://www.mom.gov.sg/~/media/mom/documents/services-forms/passes/guide_on_comp_of_company_quota_balance.pdf"),
    ("moh", "healthier-sg", "healthiersg-whitepaper", "https://file.go.gov.sg/healthiersg-whitepaper-pdf.pdf"),
    ("moh", "chas", "chas-brochure-2024", "https://www.chas.sg/Documents/Form%20and%20Other%20Materials/CHAS%20Brochures/CHAS%20English%20Brochure%20(Dec%202024).pdf"),
    ("moh", "chas", "chas-application-form-2025", "https://www.chas.sg/Documents/Form%20and%20Other%20Materials/CHAS%20Application%20Form_Apr%202025%20Version.pdf"),
]


def create_directories():
    """Create directory structure for Phase 2 data."""
    dirs = []
    for subcategory in MOH_URLS.keys():
        dirs.append(CORPUS_DIR / "moh" / subcategory)
    for subcategory in MOM_URLS.keys():
        dirs.append(CORPUS_DIR / "mom" / subcategory)

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
    total_moh = sum(len(urls) for urls in MOH_URLS.values())
    total_mom = sum(len(urls) for urls in MOM_URLS.values())
    total_pdfs = len(PDF_URLS)
    return total_moh, total_mom, total_pdfs


def main():
    """Main scraping function."""
    total_moh_urls, total_mom_urls, total_pdfs = count_urls()

    print("=" * 60)
    print("Phase 2 Data Scraper for GraphRAG (Enhanced)")
    print("Scraping MOH (Healthcare) and MOM (Employment) data")
    print("=" * 60)
    print(f"MOH URLs: {total_moh_urls}")
    print(f"MOM URLs: {total_mom_urls}")
    print(f"PDFs: {total_pdfs}")
    print(f"Total: {total_moh_urls + total_mom_urls + total_pdfs}")

    # Create directories
    print("\nCreating directories...")
    create_directories()

    # Scrape MOH data
    total_moh = 0
    for subcategory, urls in MOH_URLS.items():
        total_moh += scrape_category("moh", subcategory, urls)

    # Scrape MOM data
    total_mom = 0
    for subcategory, urls in MOM_URLS.items():
        total_mom += scrape_category("mom", subcategory, urls)

    # Download PDFs
    total_pdfs_downloaded = scrape_pdfs()

    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"MOH pages scraped: {total_moh}/{total_moh_urls}")
    print(f"MOM pages scraped: {total_mom}/{total_mom_urls}")
    print(f"PDFs downloaded: {total_pdfs_downloaded}/{total_pdfs}")
    print(f"Total files: {total_moh + total_mom + total_pdfs_downloaded}")
    print("\nNext steps:")
    print("1. Review the scraped data in corpus/raw/moh and corpus/raw/mom")
    print("2. Rebuild the graph: python main.py build")
    print("3. Test queries: python main.py serve")


if __name__ == "__main__":
    main()
