"""
Web scraper for Singapore government policy websites.

Features:
- Spider mode: Recursively follow links to discover pages
- PDF downloads: Automatically download PDF documents
- Sitemap parsing: Discover pages from sitemap.xml
- FAQ scraping: Detect and extract FAQ content

Usage:
    # Basic scrape with predefined URLs
    python scripts/scrape_gov_sites.py --phase 1

    # Spider mode - crawl from seed URLs
    python scripts/scrape_gov_sites.py --spider --max-depth 2 --domain cpf.gov.sg

    # Parse sitemap
    python scripts/scrape_gov_sites.py --sitemap --domain cpf.gov.sg

    # Download PDFs only
    python scripts/scrape_gov_sites.py --pdfs --domain hdb.gov.sg
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "scraper.log", rotation="10 MB")

# Base directories
CORPUS_DIR = Path(__file__).parent.parent / "corpus" / "raw"
PDF_DIR = Path(__file__).parent.parent / "corpus" / "pdfs"

# Request settings
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Document extensions to download
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}


@dataclass
class ScraperConfig:
    """Configuration for the web scraper."""

    delay: float = 2.0
    max_depth: int = 2
    max_pages: int = 500
    respect_robots: bool = True
    download_documents: bool = True
    parse_faqs: bool = True
    allowed_domains: set[str] = field(default_factory=set)
    excluded_patterns: list[str] = field(
        default_factory=lambda: [
            r"/search",
            r"/login",
            r"/register",
            r"/logout",
            r"/api/",
            r"/cdn-cgi/",
            r"\?",  # Query strings often lead to duplicates
            r"#",  # Anchors
            r"/print/",
            r"/share/",
            r"/email/",
        ]
    )


@dataclass
class PageData:
    """Structured data extracted from a web page."""

    url: str
    category: str
    title: str
    text_chunks: list[str]
    faqs: list[dict[str, str]]  # List of {"question": ..., "answer": ...}
    links: list[str]
    document_links: list[str]
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.url,
            "category": self.category,
            "title": self.title,
            "scraped_at": self.scraped_at,
            "text_chunks": self.text_chunks,
            "faqs": self.faqs if self.faqs else None,
        }

    def has_content(self) -> bool:
        """Check if page has meaningful content."""
        return len(self.text_chunks) >= 3 or len(self.faqs) >= 1


# =============================================================================
# Phase URL configurations
# =============================================================================

PHASE_1_SEEDS = {
    "cpf.gov.sg": [
        "https://www.cpf.gov.sg/member/cpf-overview",
        "https://www.cpf.gov.sg/member/home-ownership",
        "https://www.cpf.gov.sg/member/retirement-income",
        "https://www.cpf.gov.sg/member/growing-your-savings",
        "https://www.cpf.gov.sg/member/healthcare-financing",
        "https://www.cpf.gov.sg/employer/employer-obligations",
        "https://www.cpf.gov.sg/member/faq",
    ],
    "hdb.gov.sg": [
        "https://www.hdb.gov.sg/residential/buying-a-flat",
        "https://www.hdb.gov.sg/residential/selling-a-flat",
        "https://www.hdb.gov.sg/residential/renting-a-flat",
        "https://www.hdb.gov.sg/residential/living-in-an-hdb-flat",
        "https://www.hdb.gov.sg/cs/infoweb/residential/financing-a-flat-purchase",
    ],
    "iras.gov.sg": [
        "https://www.iras.gov.sg/taxes/individual-income-tax",
        "https://www.iras.gov.sg/taxes/corporate-income-tax",
        "https://www.iras.gov.sg/taxes/goods-services-tax-(gst)",
        "https://www.iras.gov.sg/taxes/property-tax",
        "https://www.iras.gov.sg/taxes/stamp-duty",
    ],
}

PHASE_2_SEEDS = {
    "moh.gov.sg": [
        "https://www.moh.gov.sg/healthcare-schemes-subsidies",
    ],
    "mom.gov.sg": [
        "https://www.mom.gov.sg/passes-and-permits",
        "https://www.mom.gov.sg/employment-practices",
    ],
}

PHASE_3_SEEDS = {
    "enterprisesg.gov.sg": [
        "https://www.enterprisesg.gov.sg/financial-support",
    ],
    "skillsfuture.gov.sg": [
        "https://www.skillsfuture.gov.sg/initiatives",
    ],
}


# =============================================================================
# HTTP utilities
# =============================================================================


class RateLimiter:
    """Simple rate limiter for HTTP requests."""

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.last_request_time: dict[str, float] = {}

    def wait(self, domain: str) -> None:
        """Wait if needed to respect rate limit for domain."""
        now = time.time()
        last_time = self.last_request_time.get(domain, 0)
        elapsed = now - last_time

        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        self.last_request_time[domain] = time.time()


def fetch_url(
    url: str, rate_limiter: RateLimiter | None = None
) -> requests.Response | None:
    """
    Fetch a URL with error handling and optional rate limiting.

    Args:
        url: URL to fetch
        rate_limiter: Optional rate limiter instance

    Returns:
        Response object or None if failed
    """
    try:
        domain = urlparse(url).netloc
        if rate_limiter:
            rate_limiter.wait(domain)

        response = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        response.raise_for_status()
        return response

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def download_file(
    url: str, output_path: Path, rate_limiter: RateLimiter | None = None
) -> bool:
    """
    Download a file (PDF, etc.) to disk.

    Args:
        url: URL of the file
        output_path: Path to save the file
        rate_limiter: Optional rate limiter instance

    Returns:
        True if successful, False otherwise
    """
    try:
        domain = urlparse(url).netloc
        if rate_limiter:
            rate_limiter.wait(domain)

        response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(
            f"Downloaded: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


# =============================================================================
# HTML parsing utilities
# =============================================================================


def extract_page_data(html: str, url: str, category: str) -> PageData:
    """
    Extract structured data from HTML page.

    Args:
        html: Raw HTML content
        url: Source URL
        category: Category path for organization

    Returns:
        PageData object with extracted content
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "iframe",
            "noscript",
            "svg",
            "form",
        ]
    ):
        tag.decompose()

    # Remove cookie banners, modals, etc.
    for tag in soup.find_all(
        class_=re.compile(
            r"cookie|modal|popup|banner|advertisement|sidebar|menu|nav", re.I
        )
    ):
        tag.decompose()

    # Find main content area
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main", re.I))
        or soup.find(class_=re.compile(r"content|main|article", re.I))
        or soup.find("body")
    )

    if not main_content:
        main_content = soup.body or soup

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
    if not title:
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)

    # Extract text chunks
    text_chunks = extract_text_chunks(main_content)

    # Extract FAQs
    faqs = extract_faqs(soup)

    # Extract links
    links = extract_links(soup, url)

    # Extract document links
    document_links = extract_document_links(soup, url)

    return PageData(
        url=url,
        category=category,
        title=title,
        text_chunks=text_chunks,
        faqs=faqs,
        links=links,
        document_links=document_links,
    )


def extract_text_chunks(content: BeautifulSoup) -> list[str]:
    """Extract clean text chunks from content."""
    chunks = []

    for element in content.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "ul",
            "ol",
            "table",
            "blockquote",
            "dl",
        ]
    ):
        text = element.get_text(strip=True)
        if not text or len(text) < 15:
            continue

        # Format based on element type
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            chunks.append(f"Section: {text}")

        elif element.name == "p":
            chunks.append(text)

        elif element.name in ["ul", "ol"]:
            items = []
            for li in element.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text and len(li_text) > 5:
                    items.append(li_text)
            if items:
                chunks.append("Items: " + "; ".join(items))

        elif element.name == "dl":
            # Definition lists (common in gov sites)
            pairs = []
            dts = element.find_all("dt")
            dds = element.find_all("dd")
            for dt, dd in zip(dts, dds):
                term = dt.get_text(strip=True)
                definition = dd.get_text(strip=True)
                if term and definition:
                    pairs.append(f"{term}: {definition}")
            if pairs:
                chunks.append("Definitions: " + "; ".join(pairs))

        elif element.name == "table":
            table_text = convert_table_to_text(element)
            if table_text:
                chunks.append(table_text)

        elif element.name == "blockquote":
            chunks.append(f"Note: {text}")

    # Deduplicate consecutive chunks
    deduped = []
    prev = None
    for chunk in chunks:
        if chunk != prev:
            deduped.append(chunk)
            prev = chunk

    return deduped


def extract_faqs(soup: BeautifulSoup) -> list[dict[str, str]]:
    """
    Extract FAQ question-answer pairs from page.

    Looks for common FAQ patterns:
    - Accordion/collapsible elements
    - Definition lists
    - Q&A sections
    """
    faqs = []

    # Pattern 1: Accordion/collapsible FAQ sections
    # Common classes: accordion, faq, collapse, expandable
    faq_containers = soup.find_all(
        class_=re.compile(r"faq|accordion|collapse|qa", re.I)
    )

    for container in faq_containers:
        # Look for question-answer pairs
        questions = container.find_all(
            class_=re.compile(r"question|header|title|toggle", re.I)
        )
        answers = container.find_all(
            class_=re.compile(r"answer|content|body|panel", re.I)
        )

        for q, a in zip(questions, answers):
            q_text = q.get_text(strip=True)
            a_text = a.get_text(strip=True)
            if q_text and a_text and len(a_text) > 20:
                faqs.append({"question": q_text, "answer": a_text})

    # Pattern 2: Definition lists used as FAQ
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            q_text = dt.get_text(strip=True)
            a_text = dd.get_text(strip=True)
            # Check if it looks like a question
            if (
                q_text
                and a_text
                and (
                    "?" in q_text
                    or q_text.lower().startswith(
                        (
                            "how",
                            "what",
                            "when",
                            "where",
                            "why",
                            "who",
                            "can",
                            "do",
                            "is",
                            "are",
                        )
                    )
                )
            ):
                faqs.append({"question": q_text, "answer": a_text})

    # Pattern 3: Headers followed by content
    for header in soup.find_all(["h2", "h3", "h4"]):
        header_text = header.get_text(strip=True)
        if "?" in header_text:
            # Find next sibling content
            answer_parts = []
            for sibling in header.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3", "h4"]:
                    break
                if sibling.name in ["p", "ul", "ol"]:
                    answer_parts.append(sibling.get_text(strip=True))

            if answer_parts:
                faqs.append(
                    {
                        "question": header_text,
                        "answer": " ".join(answer_parts)[:1000],  # Limit length
                    }
                )

    # Deduplicate FAQs
    seen = set()
    unique_faqs = []
    for faq in faqs:
        key = faq["question"].lower()[:50]
        if key not in seen:
            seen.add(key)
            unique_faqs.append(faq)

    return unique_faqs


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract all internal links from page."""
    links = []
    base_domain = urlparse(base_url).netloc

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        # Skip empty, javascript, and anchor-only links
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only keep links to same domain or subdomains
        if base_domain in parsed.netloc or parsed.netloc in base_domain:
            # Normalize URL (remove fragment, trailing slash)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            normalized = normalized.rstrip("/")

            if normalized not in links:
                links.append(normalized)

    return links


def extract_document_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract links to downloadable documents (PDF, DOC, etc.)."""
    doc_links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        path_lower = parsed.path.lower()

        # Check if it's a document
        for ext in DOCUMENT_EXTENSIONS:
            if path_lower.endswith(ext):
                doc_links.append(full_url)
                break

    return list(set(doc_links))


def convert_table_to_text(table: BeautifulSoup) -> str:
    """Convert HTML table to readable text format."""
    rows = table.find_all("tr")
    if not rows:
        return ""

    # Extract headers
    headers = []
    first_row = rows[0]
    header_cells = first_row.find_all(["th", "td"])
    headers = [cell.get_text(strip=True) for cell in header_cells]

    # Extract data rows
    data_rows = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        if any(cell_texts):
            data_rows.append(cell_texts)

    if not headers and not data_rows:
        return ""

    # Format as readable text
    lines = []
    if headers:
        lines.append(f"Table columns: {', '.join(h for h in headers if h)}")

    for row_data in data_rows[:20]:  # Limit rows
        if len(row_data) == len(headers):
            pairs = [f"{h}: {v}" for h, v in zip(headers, row_data) if h and v]
            if pairs:
                lines.append(" | ".join(pairs))
        else:
            non_empty = [v for v in row_data if v]
            if non_empty:
                lines.append(" | ".join(non_empty))

    return "Table data: " + "; ".join(lines) if lines else ""


# =============================================================================
# Sitemap parsing
# =============================================================================


def parse_sitemap(
    sitemap_url: str, rate_limiter: RateLimiter | None = None
) -> list[str]:
    """
    Parse sitemap.xml and extract all URLs.

    Handles both regular sitemaps and sitemap indexes.

    Args:
        sitemap_url: URL of the sitemap
        rate_limiter: Optional rate limiter

    Returns:
        List of URLs found in sitemap
    """
    urls = []

    response = fetch_url(sitemap_url, rate_limiter)
    if not response:
        # Try common sitemap locations
        base_url = f"{urlparse(sitemap_url).scheme}://{urlparse(sitemap_url).netloc}"
        for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemaps/sitemap.xml"]:
            response = fetch_url(base_url + path, rate_limiter)
            if response:
                break

    if not response:
        logger.warning(f"Could not find sitemap for {sitemap_url}")
        return urls

    try:
        root = ET.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Check if it's a sitemap index
        sitemap_elements = root.findall(".//ns:sitemap/ns:loc", namespace)
        if sitemap_elements:
            # It's an index, recursively parse each sitemap
            logger.info(f"Found sitemap index with {len(sitemap_elements)} sitemaps")
            for sitemap in sitemap_elements[:10]:  # Limit to first 10 sub-sitemaps
                sub_urls = parse_sitemap(sitemap.text, rate_limiter)
                urls.extend(sub_urls)
        else:
            # Regular sitemap, extract URLs
            url_elements = root.findall(".//ns:url/ns:loc", namespace)
            for url_elem in url_elements:
                if url_elem.text:
                    urls.append(url_elem.text)
            logger.info(f"Found {len(urls)} URLs in sitemap")

    except ET.ParseError as e:
        logger.warning(f"Failed to parse sitemap XML: {e}")

    return urls


def filter_sitemap_urls(urls: list[str], config: ScraperConfig) -> list[str]:
    """Filter sitemap URLs based on configuration."""
    filtered = []

    for url in urls:
        # Check excluded patterns
        excluded = False
        for pattern in config.excluded_patterns:
            if re.search(pattern, url):
                excluded = True
                break

        if not excluded:
            filtered.append(url)

    return filtered


# =============================================================================
# Spider (recursive crawler)
# =============================================================================


class Spider:
    """Recursive web crawler with depth limiting."""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.delay)
        self.visited: set[str] = set()
        self.document_urls: set[str] = set()

    def normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return normalized.rstrip("/").lower()

    def should_visit(self, url: str) -> bool:
        """Check if URL should be visited."""
        normalized = self.normalize_url(url)

        # Already visited
        if normalized in self.visited:
            return False

        # Check domain restrictions
        domain = urlparse(url).netloc
        if self.config.allowed_domains:
            allowed = False
            for allowed_domain in self.config.allowed_domains:
                if allowed_domain in domain or domain in allowed_domain:
                    allowed = True
                    break
            if not allowed:
                return False

        # Check excluded patterns
        for pattern in self.config.excluded_patterns:
            if re.search(pattern, url):
                return False

        # Check if it's a document (we'll download these separately)
        path_lower = urlparse(url).path.lower()
        for ext in DOCUMENT_EXTENSIONS:
            if path_lower.endswith(ext):
                self.document_urls.add(url)
                return False

        return True

    def crawl(
        self,
        seed_urls: list[str],
        category: str,
    ) -> Generator[PageData, None, None]:
        """
        Crawl from seed URLs up to max_depth.

        Uses BFS (breadth-first search) with depth tracking.

        Args:
            seed_urls: Starting URLs
            category: Category for organizing output

        Yields:
            PageData for each successfully scraped page
        """
        # Queue: (url, depth)
        queue: deque[tuple[str, int]] = deque()

        for url in seed_urls:
            if self.should_visit(url):
                queue.append((url, 0))
                self.visited.add(self.normalize_url(url))

        pages_scraped = 0

        while queue and pages_scraped < self.config.max_pages:
            url, depth = queue.popleft()

            logger.info(f"[Depth {depth}] Crawling: {url}")

            response = fetch_url(url, self.rate_limiter)
            if not response:
                continue

            # Check content type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                continue

            # Extract page data
            page_data = extract_page_data(response.text, url, category)

            if page_data.has_content():
                pages_scraped += 1
                yield page_data

            # Collect document links
            for doc_url in page_data.document_links:
                self.document_urls.add(doc_url)

            # Add new links to queue if not at max depth
            if depth < self.config.max_depth:
                for link in page_data.links:
                    if self.should_visit(link):
                        self.visited.add(self.normalize_url(link))
                        queue.append((link, depth + 1))

        logger.info(
            f"Spider finished: {pages_scraped} pages scraped, {len(self.document_urls)} documents found"
        )


# =============================================================================
# Document downloader
# =============================================================================


def download_documents(
    document_urls: set[str],
    output_dir: Path,
    rate_limiter: RateLimiter,
    max_documents: int = 100,
) -> int:
    """
    Download documents (PDFs, etc.) to output directory.

    Args:
        document_urls: Set of document URLs
        output_dir: Directory to save documents
        rate_limiter: Rate limiter instance
        max_documents: Maximum documents to download

    Returns:
        Number of documents successfully downloaded
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for url in list(document_urls)[:max_documents]:
        # Generate filename from URL
        parsed = urlparse(url)
        filename = Path(parsed.path).name

        # Ensure unique filename
        output_path = output_dir / filename
        if output_path.exists():
            # Add hash to make unique
            import hashlib

            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_dir / f"{stem}_{url_hash}{suffix}"

        if download_file(url, output_path, rate_limiter):
            downloaded += 1

    return downloaded


# =============================================================================
# Output utilities
# =============================================================================


def sanitize_filename(url: str) -> str:
    """Create a safe filename from URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = "index"
    path = re.sub(r"[^\w\-]", "_", path)
    path = re.sub(r"_+", "_", path)
    return path[:100] + ".json"


def save_page_data(page_data: PageData, output_dir: Path) -> bool:
    """Save page data to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(page_data.url)
    output_path = output_dir / filename

    # Handle duplicate filenames
    counter = 1
    while output_path.exists():
        stem = filename[:-5]  # Remove .json
        output_path = output_dir / f"{stem}_{counter}.json"
        counter += 1

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(page_data.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return False


# =============================================================================
# Main entry points
# =============================================================================


def scrape_with_spider(
    seeds: dict[str, list[str]],
    config: ScraperConfig,
    download_docs: bool = True,
) -> dict[str, int]:
    """
    Scrape using spider mode.

    Args:
        seeds: Dictionary of domain -> seed URLs
        config: Scraper configuration
        download_docs: Whether to download documents

    Returns:
        Statistics dictionary
    """
    stats = {"pages": 0, "documents": 0, "faqs": 0}

    for domain, seed_urls in seeds.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Spidering domain: {domain}")
        logger.info(f"{'='*60}")

        # Configure spider for this domain
        config.allowed_domains = {domain}
        spider = Spider(config)

        # Determine category from domain
        category = domain.split(".")[0]  # e.g., "cpf" from "cpf.gov.sg"
        output_dir = CORPUS_DIR / category

        # Crawl and save pages
        for page_data in spider.crawl(seed_urls, category):
            if save_page_data(page_data, output_dir):
                stats["pages"] += 1
                stats["faqs"] += len(page_data.faqs)

        # Download documents
        if download_docs and spider.document_urls:
            logger.info(f"Downloading {len(spider.document_urls)} documents...")
            doc_output = PDF_DIR / category
            downloaded = download_documents(
                spider.document_urls,
                doc_output,
                spider.rate_limiter,
            )
            stats["documents"] += downloaded

    return stats


def scrape_from_sitemap(
    domain: str,
    config: ScraperConfig,
    max_pages: int = 200,
) -> dict[str, int]:
    """
    Scrape pages discovered from sitemap.

    Args:
        domain: Domain to scrape (e.g., "cpf.gov.sg")
        config: Scraper configuration
        max_pages: Maximum pages to scrape

    Returns:
        Statistics dictionary
    """
    stats = {"pages": 0, "documents": 0, "faqs": 0}

    sitemap_url = f"https://www.{domain}/sitemap.xml"
    rate_limiter = RateLimiter(config.delay)

    logger.info(f"Parsing sitemap: {sitemap_url}")
    urls = parse_sitemap(sitemap_url, rate_limiter)

    if not urls:
        logger.warning(f"No URLs found in sitemap for {domain}")
        return stats

    # Filter URLs
    urls = filter_sitemap_urls(urls, config)
    logger.info(f"Found {len(urls)} URLs after filtering")

    # Limit pages
    urls = urls[:max_pages]

    category = domain.split(".")[0]
    output_dir = CORPUS_DIR / category
    document_urls: set[str] = set()

    for url in urls:
        logger.info(f"Scraping: {url}")

        response = fetch_url(url, rate_limiter)
        if not response:
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type.lower():
            continue

        page_data = extract_page_data(response.text, url, category)

        if page_data.has_content():
            if save_page_data(page_data, output_dir):
                stats["pages"] += 1
                stats["faqs"] += len(page_data.faqs)

        # Collect document links
        document_urls.update(page_data.document_links)

    # Download documents
    if config.download_documents and document_urls:
        logger.info(f"Downloading {len(document_urls)} documents...")
        doc_output = PDF_DIR / category
        downloaded = download_documents(document_urls, doc_output, rate_limiter)
        stats["documents"] += downloaded

    return stats


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Web scraper for Singapore government policy websites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic phase-based scraping with spider
  python scrape_gov_sites.py --phase 1 --spider

  # Sitemap-based scraping for specific domain
  python scrape_gov_sites.py --sitemap --domain cpf.gov.sg

  # Spider with custom depth and page limit
  python scrape_gov_sites.py --spider --domain hdb.gov.sg --max-depth 3 --max-pages 200
        """,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--spider",
        action="store_true",
        help="Enable spider mode (follow links recursively)",
    )
    mode_group.add_argument(
        "--sitemap",
        action="store_true",
        help="Discover pages from sitemap.xml",
    )

    # Scope selection
    scope_group = parser.add_argument_group("Scope")
    scope_group.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Scrape predefined phase (1=CPF/HDB/IRAS, 2=MOH/MOM, 3=Enterprise/SkillsFuture)",
    )
    scope_group.add_argument(
        "--domain",
        type=str,
        help="Specific domain to scrape (e.g., cpf.gov.sg)",
    )

    # Spider options
    spider_group = parser.add_argument_group("Spider Options")
    spider_group.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum crawl depth (default: 2)",
    )
    spider_group.add_argument(
        "--max-pages",
        type=int,
        default=300,
        help="Maximum pages to scrape per domain (default: 300)",
    )

    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between requests in seconds (default: 1.5)",
    )
    general_group.add_argument(
        "--no-pdfs",
        action="store_true",
        help="Skip downloading PDF documents",
    )

    args = parser.parse_args()

    # Build configuration
    config = ScraperConfig(
        delay=args.delay,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        download_documents=not args.no_pdfs,
    )

    # Determine seeds based on arguments
    if args.domain:
        # Single domain mode
        domain = args.domain
        if not domain.startswith("www."):
            domain_with_www = f"www.{domain}"
        else:
            domain_with_www = domain
            domain = domain.replace("www.", "")

        seed_url = f"https://{domain_with_www}"
        seeds = {domain: [seed_url]}

    elif args.phase:
        # Phase-based mode
        if args.phase == 1:
            seeds = PHASE_1_SEEDS
        elif args.phase == 2:
            seeds = PHASE_2_SEEDS
        else:
            seeds = PHASE_3_SEEDS
    else:
        # Default to phase 1
        seeds = PHASE_1_SEEDS

    # Execute based on mode
    total_stats = {"pages": 0, "documents": 0, "faqs": 0}

    if args.sitemap:
        # Sitemap mode
        for domain in seeds.keys():
            stats = scrape_from_sitemap(domain, config, args.max_pages)
            for key in total_stats:
                total_stats[key] += stats[key]

    elif args.spider:
        # Spider mode
        stats = scrape_with_spider(seeds, config, not args.no_pdfs)
        total_stats = stats

    else:
        # Default: spider mode
        logger.info("No mode specified, using spider mode")
        stats = scrape_with_spider(seeds, config, not args.no_pdfs)
        total_stats = stats

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SCRAPING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Pages scraped: {total_stats['pages']}")
    logger.info(f"FAQs extracted: {total_stats['faqs']}")
    logger.info(f"Documents downloaded: {total_stats['documents']}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
