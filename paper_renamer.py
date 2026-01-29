#!/usr/bin/env python3
"""
Paper Renamer - Automatically rename scientific PDF files based on metadata.

Extracts title, authors, year, and journal from PDF metadata or text content,
then renames files using a configurable format string.

Usage:
    python paper_renamer.py [OPTIONS] FILE [FILE...]

Examples:
    paper_renamer.py paper.pdf
    paper_renamer.py --dry-run *.pdf
    paper_renamer.py -f "{year}_{first_author}_{short_title}" paper.pdf
    paper_renamer.py -o ./renamed/ -v *.pdf
"""

import argparse
import re
import sys
import urllib.request
import urllib.error
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Optional dependencies - check at runtime
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import yake
    HAS_YAKE = True
except ImportError:
    HAS_YAKE = False

try:
    from pdf2image import convert_from_path
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


# Minimum text length to consider extraction successful
MIN_TEXT_LENGTH = 200


def extract_text_via_ocr(pdf_path: Path, pages: int = 2, verbose: bool = False) -> str:
    """Extract text from PDF using OCR on the first few pages."""
    if not HAS_OCR:
        if verbose:
            print("  OCR not available (install pdf2image and pytesseract)")
        return ""

    if verbose:
        print(f"  Attempting OCR on first {pages} page(s)...")

    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, first_page=1, last_page=pages, dpi=150)

        text_parts = []
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text_parts.append(page_text)
            if verbose:
                print(f"  OCR page {i+1}: {len(page_text)} chars")

        return "\n".join(text_parts)

    except Exception as e:
        if verbose:
            print(f"  OCR failed: {e}")
        return ""


def extract_title_and_authors_by_font_size(pdf_path: Path, verbose: bool = False) -> tuple[Optional[str], list[str]]:
    """Extract title and authors by analyzing font sizes on the first page.

    Strategy:
    1. Find the largest-font text cluster (title)
    2. Look for text immediately below the title that looks like author names
       (smaller than title but larger than body, with name-like patterns)
    """
    if not HAS_PDFPLUMBER:
        return None, []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return None, []

            first_page = pdf.pages[0]
            words = first_page.extract_words(extra_attrs=['size'])

            if not words:
                return None, []

            page_height = first_page.height

            # Calculate median font size from entire page (body text baseline)
            all_sizes = [w.get('size', 12) for w in words]
            all_sizes.sort()
            median_size = all_sizes[len(all_sizes) // 2]

            # Find words significantly larger than median (1.5x or at least 17pt)
            large_threshold = max(median_size * 1.5, 17.0)
            large_words = [w for w in words if w.get('size', 12) >= large_threshold]

            if verbose:
                print(f"  Font-size analysis: median={median_size:.1f}pt, "
                      f"threshold={large_threshold:.1f}pt, {len(large_words)} large words")

            if not large_words:
                return None, []

            # Group words into clusters by vertical proximity
            # First, group into lines (words within 5pt vertically)
            lines: list[dict] = []
            for word in sorted(large_words, key=lambda w: (w.get('top', 0), w.get('x0', 0))):
                word_top = word.get('top', 0)
                word_size = word.get('size', 12)

                found_line = False
                for line in lines:
                    if abs(line['top'] - word_top) < 5:
                        line['words'].append(word)
                        line['size'] = max(line['size'], word_size)
                        found_line = True
                        break

                if not found_line:
                    lines.append({
                        'top': word_top,
                        'words': [word],
                        'size': word_size
                    })

            if not lines:
                return None, []

            # Group lines into clusters (lines within 50pt of each other)
            lines.sort(key=lambda l: l['top'])
            clusters: list[list[dict]] = []
            current_cluster: list[dict] = []

            for line in lines:
                if not current_cluster:
                    current_cluster = [line]
                elif line['top'] - current_cluster[-1]['top'] < 50:
                    current_cluster.append(line)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [line]

            if current_cluster:
                clusters.append(current_cluster)

            if not clusters:
                return None, []

            # Score clusters: prefer larger fonts, not at bottom of page
            def cluster_score(cluster: list[dict]) -> tuple[float, float]:
                max_size = max(l['size'] for l in cluster)
                avg_top = sum(l['top'] for l in cluster) / len(cluster)
                position_penalty = 0 if avg_top < page_height * 0.8 else 100
                return (-max_size, position_penalty + avg_top / 1000)

            clusters.sort(key=cluster_score)
            best_cluster = clusters[0]

            # Build title from the best cluster
            max_cluster_size = max(l['size'] for l in best_cluster)
            title_lines = [l for l in best_cluster if l['size'] >= max_cluster_size * 0.85]
            title_lines.sort(key=lambda l: l['top'])

            title_words = []
            for line in title_lines:
                line_words = sorted(line['words'], key=lambda w: w.get('x0', 0))
                for w in line_words:
                    title_words.append(w.get('text', ''))

            title = ' '.join(title_words)
            title = ' '.join(title.split())
            title = re.sub(r'[.,:;]+$', '', title)
            title = re.sub(r'[\*†‡§]+', '', title)

            if verbose:
                print(f"  Font-size title: {title[:60]}{'...' if len(title) > 60 else ''} ({max_cluster_size:.1f}pt)")

            title = title if len(title) > 10 else None

            # --- Author extraction ---
            # Look for text below the title that could be author names
            # Authors are typically:
            # - Immediately below title (within ~80pt)
            # - Smaller than title but larger than body text
            # - Contain name-like patterns (capitalized, commas, initials)

            title_bottom = max(l['top'] for l in title_lines) + 20
            author_search_top = title_bottom
            author_search_bottom = title_bottom + 80

            # Author font threshold: between body text and title
            # Allow up to 95% of title size to catch author names that are almost as large
            author_min_size = max(median_size * 1.2, 14.0)
            author_max_size = max_cluster_size * 0.95  # Just below title size

            potential_author_words = [
                w for w in words
                if author_min_size <= w.get('size', 12) <= author_max_size
                and author_search_top < w.get('top', 0) < author_search_bottom
            ]

            if verbose:
                print(f"  Author search: y={author_search_top:.0f}-{author_search_bottom:.0f}, "
                      f"size={author_min_size:.1f}-{author_max_size:.1f}pt, "
                      f"{len(potential_author_words)} candidates")

            authors = []
            if potential_author_words:
                # Group into lines
                author_lines: list[list[dict]] = []
                for word in sorted(potential_author_words, key=lambda w: (w.get('top', 0), w.get('x0', 0))):
                    word_top = word.get('top', 0)
                    found_line = False
                    for line in author_lines:
                        if abs(line[0].get('top', 0) - word_top) < 8:
                            line.append(word)
                            found_line = True
                            break
                    if not found_line:
                        author_lines.append([word])

                # Extract text from each line and parse for names
                for line in author_lines:
                    line_words = sorted(line, key=lambda w: w.get('x0', 0))
                    line_text = ' '.join(w.get('text', '') for w in line_words)

                    # Clean up: remove affiliation numbers, asterisks, etc.
                    line_text = re.sub(r'[\d\*†‡§]+', '', line_text)
                    line_text = ' '.join(line_text.split())

                    if not line_text:
                        continue

                    # Check if this looks like an author line
                    # Heuristics: contains commas, has capitalized words, has initials
                    has_comma = ',' in line_text
                    has_capitals = bool(re.search(r'[A-Z][a-z]', line_text))
                    has_initials = bool(re.search(r'\b[A-Z]\.', line_text))

                    if has_capitals and (has_comma or has_initials):
                        # Parse individual names
                        # Split by common delimiters
                        if ' and ' in line_text.lower():
                            parts = re.split(r'\s+and\s+', line_text, flags=re.IGNORECASE)
                        else:
                            parts = [p.strip() for p in line_text.split(',') if p.strip()]

                        for part in parts:
                            part = part.strip(' ,')
                            # Skip if looks like affiliation
                            if re.search(r'(?:University|Institute|Department|Center|Centre|Hospital)', part, re.IGNORECASE):
                                continue
                            # Skip if too short or too long
                            if len(part) < 2 or len(part) > 40:
                                continue
                            # Should have at least one capital
                            if not re.search(r'[A-Z]', part):
                                continue
                            # Filter out common non-name words that might leak from adjacent content
                            noise_words = {'tip', 'the', 'and', 'for', 'with', 'from', 'that', 'this',
                                          'separation', 'analysis', 'method', 'results', 'figure',
                                          'table', 'data', 'sample', 'study', 'research', 'light',
                                          'high', 'low', 'new', 'first', 'using', 'based'}
                            # Remove any leading noise words
                            words_list = part.split()
                            while words_list and words_list[0].lower().rstrip('.,;:') in noise_words:
                                words_list = words_list[1:]
                            if not words_list:
                                continue
                            part = ' '.join(words_list)
                            # Also remove trailing noise
                            while words_list and words_list[-1].lower().rstrip('.,;:') in noise_words:
                                words_list = words_list[:-1]
                            if not words_list:
                                continue
                            part = ' '.join(words_list)
                            # Validate: should look like a name (has 2+ words or has initial)
                            words = part.split()
                            looks_like_name = (
                                len(words) >= 2 or  # First Last
                                re.search(r'\b[A-Z]\.\s*[A-Z]', part) or  # F. Last or F. M. Last
                                (len(words) == 1 and len(part) > 3)  # Single longer name
                            )
                            if looks_like_name:
                                authors.append(part)

            if verbose and authors:
                print(f"  Font-size authors: {authors[0]}{'...' if len(authors) > 1 else ''}")

            return title, authors

    except Exception as e:
        if verbose:
            print(f"  Font-size extraction failed: {e}")
        return None, []


def extract_title_by_font_size(pdf_path: Path, verbose: bool = False) -> Optional[str]:
    """Extract title by font size (wrapper for backwards compatibility)."""
    title, _ = extract_title_and_authors_by_font_size(pdf_path, verbose)
    return title


def extract_year_by_position(pdf_path: Path, verbose: bool = False) -> Optional[str]:
    """Extract publication year preferring header/footer regions.

    Years in headers/footers are more likely to be publication years,
    while years in the middle of the page might be from article content
    or adjacent articles in multi-column layouts.

    Also excludes years near "downloaded", "retrieved", "accessed" which
    indicate when the PDF was obtained, not when it was published.
    """
    if not HAS_PDFPLUMBER:
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return None

            first_page = pdf.pages[0]
            page_height = first_page.height
            page_width = first_page.width
            words = first_page.extract_words()

            if not words:
                return None

            # Build a map of words by approximate line (y position)
            # to detect "downloaded on YEAR" patterns
            download_keywords = {'downloaded', 'retrieved', 'accessed', 'printed', 'saved'}
            download_lines: set[int] = set()  # y positions (bucketed) with download keywords

            for word in words:
                text = word.get('text', '').lower()
                if any(kw in text for kw in download_keywords):
                    # Bucket y position to ~20pt lines
                    y_bucket = int(word.get('top', 0) / 20)
                    download_lines.add(y_bucket)

            # Find all 4-digit years in the text
            year_candidates: list[tuple[str, float, float, bool]] = []  # (year, y_pos, x_pos, near_download)

            for word in words:
                text = word.get('text', '')
                match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
                if match:
                    year = match.group(1)
                    y_pos = word.get('top', 0)
                    x_pos = word.get('x0', 0)
                    y_bucket = int(y_pos / 20)
                    # Check if this year is on or near a "download" line
                    near_download = any(abs(y_bucket - dl) <= 1 for dl in download_lines)
                    year_candidates.append((year, y_pos, x_pos, near_download))

            if not year_candidates:
                return None

            # Detect reference clusters: many different years in a small vertical area
            # (typical of reference sections from adjacent articles)
            year_buckets: dict[int, set[str]] = {}  # y_bucket -> set of years
            for year, y, x, near_dl in year_candidates:
                y_bucket = int(y / 50)  # ~50pt buckets
                if y_bucket not in year_buckets:
                    year_buckets[y_bucket] = set()
                year_buckets[y_bucket].add(year)

            # Buckets with 3+ different years are likely reference sections
            reference_buckets = {b for b, years in year_buckets.items() if len(years) >= 3}

            # Score candidates: lower score = better
            def year_score(candidate: tuple[str, float, float, bool]) -> float:
                year, y, x, near_download = candidate
                score = 0.0

                # Heavily penalize years near download/retrieved keywords
                if near_download:
                    score += 100  # Almost certainly not publication year

                # Penalize years in reference-like clusters (many different years nearby)
                y_bucket = int(y / 50)
                if y_bucket in reference_buckets:
                    score += 50  # Likely from a reference section

                # Prefer footer (bottom 10%) for publication dates
                # Many journals put the publication year in the footer
                y_ratio = y / page_height
                if y_ratio > 0.90:
                    score += 0  # Footer - best for publication year
                elif y_ratio < 0.12:
                    score += 5  # Header - sometimes has pub year, but often other content
                else:
                    score += 10  # Middle of page - less reliable

                # Prefer horizontally centered content (likely the main article)
                x_ratio = x / page_width
                if 0.25 < x_ratio < 0.75:
                    score += 0  # Centered
                else:
                    score += 3  # At edges - might be adjacent article

                # Prefer years that look like publication years (not too old, not future)
                year_int = int(year)
                if 1950 <= year_int <= 2030:
                    score += 0
                elif 1900 <= year_int < 1950:
                    score += 2  # Older papers
                else:
                    score += 20  # Suspicious

                return score

            # Sort by score and return best
            year_candidates.sort(key=year_score)

            if verbose:
                best = year_candidates[0]
                y_pct = best[1] / page_height * 100
                dl_note = " (near download keyword)" if best[3] else ""
                print(f"  Position-based year: {best[0]} (y={y_pct:.0f}% from top){dl_note}")

            return year_candidates[0][0]

    except Exception as e:
        if verbose:
            print(f"  Position-based year extraction failed: {e}")
        return None


def is_technical_term(word: str) -> bool:
    """Check if a word looks like a technical term based on formatting."""
    # Strip punctuation for checking
    clean = word.strip(',:;()[]')
    if len(clean) < 2:
        return False

    # ALL CAPS (like FACETS, DNA, RNA, CRISPR)
    if clean.isupper() and len(clean) >= 2:
        return True

    # CamelCase or mixed case (like cfDNA, mRNA, HeLa)
    if any(c.isupper() for c in clean[1:]) and any(c.islower() for c in clean):
        return True

    # Contains numbers (like p53, IL-6, COVID-19)
    if any(c.isdigit() for c in clean):
        return True

    # Hyphenated technical terms (like allele-specific, high-throughput)
    if '-' in clean and len(clean) > 5:
        return True

    # Ends with common technical suffixes
    technical_suffixes = ('omics', 'ome', 'ase', 'osis', 'itis', 'emia', 'seq')
    if any(clean.lower().endswith(s) for s in technical_suffixes):
        return True

    return False


def extract_technical_terms(text: str) -> list[str]:
    """Extract technical terms from text based on formatting heuristics."""
    words = text.split()
    technical = []
    for word in words:
        clean = word.strip(',:;()[]')
        if is_technical_term(clean) and clean not in technical:
            technical.append(clean)
    return technical


def extract_key_phrases(text: str, max_words: int = 5) -> Optional[str]:
    """Extract key phrases from text using YAKE + technical term heuristics.

    Strategy:
    1. Score each word based on technical term detection and YAKE importance
    2. Select the top N words by score
    3. Return them in their original order from the title
    """
    if not text:
        return None

    # Parse words with their positions
    raw_words = text.split()
    words_with_pos = []
    for i, word in enumerate(raw_words):
        clean = word.strip(',:;()[]')
        if clean:
            words_with_pos.append((i, clean, word))

    if not words_with_pos:
        return None

    # Score each word - lower score = more important (to match YAKE convention)
    word_scores: dict[str, float] = {}

    # Technical term bonus: give very low scores (high priority)
    for i, clean, _ in words_with_pos:
        if is_technical_term(clean):
            # Score based on position (earlier = slightly better) and technical importance
            word_scores[clean.lower()] = 0.01 + (i * 0.001)

    # Get YAKE scores if available
    if HAS_YAKE:
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # max ngram size
            dedupLim=0.7,  # deduplication threshold
            top=20,  # get more keywords for better coverage
            features=None,
        )
        keywords = kw_extractor.extract_keywords(text)

        # YAKE scores are typically 0-1, lower = more important
        for keyword, score in keywords:
            for kw_word in keyword.split():
                kw_lower = kw_word.lower()
                if kw_lower in word_scores:
                    # Combine: keep the better (lower) score
                    word_scores[kw_lower] = min(word_scores[kw_lower], score)
                else:
                    word_scores[kw_lower] = score

    # For words without scores, assign a high score (low priority)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to',
                  'for', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
                  'before', 'after', 'above', 'below', 'between', 'under', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}

    for i, clean, _ in words_with_pos:
        if clean.lower() not in word_scores:
            if clean.lower() in stop_words:
                word_scores[clean.lower()] = 100.0  # Very low priority
            else:
                word_scores[clean.lower()] = 10.0 + (i * 0.01)  # Medium priority

    # Select top words by score, avoiding duplicates
    seen_lower = set()
    scored_words = []
    for i, clean, original in words_with_pos:
        lower = clean.lower()
        if lower not in seen_lower and lower not in stop_words:
            seen_lower.add(lower)
            scored_words.append((i, clean, word_scores.get(lower, 10.0)))

    # Sort by score (lower = better), take top N
    scored_words.sort(key=lambda x: x[2])
    top_words = scored_words[:max_words]

    # Re-sort by original position to maintain title order
    top_words.sort(key=lambda x: x[0])

    result = [w[1] for w in top_words]
    return '_'.join(result) if result else None


@dataclass
class PaperMetadata:
    """Container for extracted paper metadata."""
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None

    @property
    def first_author(self) -> Optional[str]:
        """Get first author's last name."""
        if not self.authors:
            return None
        return self._extract_last_name(self.authors[0])

    @property
    def last_author(self) -> Optional[str]:
        """Get last author's last name."""
        if not self.authors:
            return None
        return self._extract_last_name(self.authors[-1])

    @property
    def first_author_full(self) -> Optional[str]:
        """Get first author's full name."""
        return self.authors[0] if self.authors else None

    def _extract_last_name(self, name: str) -> str:
        """Extract last name from full name, handling various formats."""
        name = name.strip()

        # Handle "Last, First" format
        if ',' in name:
            return name.split(',')[0].strip()

        # Handle multi-part surnames (van der, de la, etc.)
        parts = name.split()
        if len(parts) >= 2:
            # Check for common surname prefixes
            prefixes = {'van', 'von', 'de', 'del', 'della', 'di', 'da', 'dos', 'das', 'la', 'le', 'du'}
            # Find where the surname starts
            for i in range(len(parts) - 1, 0, -1):
                if parts[i-1].lower() in prefixes:
                    return ' '.join(parts[i-1:])
            return parts[-1]

        return name

    def short_title(self, max_words: int = 5, use_keywords: bool = True) -> Optional[str]:
        """Get shortened title using keyword extraction or simple truncation.

        Args:
            max_words: Maximum number of words in the short title
            use_keywords: If True, use YAKE keyword extraction; else use first N words
        """
        if not self.title:
            return None

        # Try YAKE keyword extraction first
        if use_keywords:
            key_phrase = extract_key_phrases(self.title, max_words)
            if key_phrase:
                return key_phrase

        # Fallback: first N words, avoiding stop word endings
        stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to',
                      'for', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
                      'before', 'after', 'above', 'below', 'between', 'under', 'is',
                      'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}

        all_words = self.title.split()
        words = all_words[:max_words]

        # If we end on a stop word, try to fix it
        while words and words[-1].lower().rstrip(',:;') in stop_words:
            if len(words) < len(all_words):
                words = all_words[:len(words) + 1]
            else:
                words = words[:-1]

        return '_'.join(words) if words else None

    def is_complete(self) -> bool:
        """Check if we have minimum required metadata (all 3 fields)."""
        return bool(self.title and self.authors and self.year)

    def is_minimally_viable(self) -> bool:
        """Check if we have at least 2 of 3 core fields (title, authors, year).

        This allows renaming with incomplete data when --force is used.
        """
        fields_present = sum([
            bool(self.title),
            bool(self.authors),
            bool(self.year)
        ])
        return fields_present >= 2


def extract_doi(text: str) -> Optional[str]:
    """Extract DOI from text using regex."""
    # DOI pattern: 10.XXXX/... (various formats)
    patterns = [
        r'doi[:\s]*10\.\d{4,}/[^\s]+',
        r'https?://doi\.org/10\.\d{4,}/[^\s]+',
        r'10\.\d{4,}/[^\s]+',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group()
            # Clean up the DOI
            doi = re.sub(r'^(doi[:\s]*|https?://doi\.org/)', '', doi, flags=re.IGNORECASE)
            # Remove trailing punctuation
            doi = re.sub(r'[.,;)\]]+$', '', doi)
            return doi

    return None


def query_crossref(doi: str, verbose: bool = False) -> Optional[PaperMetadata]:
    """Query CrossRef API for paper metadata using DOI."""
    if verbose:
        print(f"  Querying CrossRef for DOI: {doi}")

    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    headers = {
        'User-Agent': 'PaperRenamer/1.0 (mailto:user@example.com)',
        'Accept': 'application/json'
    }

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        if verbose:
            print(f"  CrossRef lookup failed: {e}")
        return None

    try:
        work = data['message']
        metadata = PaperMetadata(doi=doi)

        # Extract title
        if 'title' in work and work['title']:
            metadata.title = work['title'][0]

        # Extract authors
        if 'author' in work:
            for author in work['author']:
                if 'family' in author:
                    if 'given' in author:
                        metadata.authors.append(f"{author['given']} {author['family']}")
                    else:
                        metadata.authors.append(author['family'])

        # Extract year
        if 'published-print' in work and 'date-parts' in work['published-print']:
            metadata.year = str(work['published-print']['date-parts'][0][0])
        elif 'published-online' in work and 'date-parts' in work['published-online']:
            metadata.year = str(work['published-online']['date-parts'][0][0])
        elif 'created' in work and 'date-parts' in work['created']:
            metadata.year = str(work['created']['date-parts'][0][0])

        # Extract journal
        if 'container-title' in work and work['container-title']:
            metadata.journal = work['container-title'][0]

        if verbose:
            print(f"  CrossRef returned: {metadata.title[:50] if metadata.title else 'No title'}...")

        return metadata

    except (KeyError, IndexError) as e:
        if verbose:
            print(f"  Error parsing CrossRef response: {e}")
        return None


def extract_metadata_from_pdf(pdf_path: Path, verbose: bool = False) -> PaperMetadata:
    """Extract metadata from PDF properties."""
    if not HAS_PYPDF:
        if verbose:
            print("  pypdf not available, skipping PDF metadata extraction")
        return PaperMetadata()

    metadata = PaperMetadata()

    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            info = reader.metadata

            if info:
                # Title
                if info.title:
                    metadata.title = info.title.strip()
                    if verbose:
                        print(f"  PDF metadata title: {metadata.title[:50]}...")

                # Author
                if info.author:
                    # Author field may contain multiple authors separated by various delimiters
                    author_str = info.author.strip()
                    # Try common separators
                    if ';' in author_str:
                        metadata.authors = [a.strip() for a in author_str.split(';') if a.strip()]
                    elif ' and ' in author_str.lower():
                        metadata.authors = [a.strip() for a in re.split(r'\s+and\s+', author_str, flags=re.IGNORECASE) if a.strip()]
                    elif ',' in author_str and author_str.count(',') > 1:
                        # Multiple commas likely means comma-separated list
                        metadata.authors = [a.strip() for a in author_str.split(',') if a.strip()]
                    else:
                        metadata.authors = [author_str]

                    if verbose and metadata.authors:
                        print(f"  PDF metadata authors: {metadata.authors[0]}...")

                # Subject often contains journal/DOI info
                if info.subject:
                    subject = info.subject
                    # Try to extract DOI from subject
                    doi = extract_doi(subject)
                    if doi:
                        metadata.doi = doi
                        if verbose:
                            print(f"  PDF metadata DOI: {doi}")

                    # Try to extract journal from subject
                    # Common pattern: "Journal Name, doi:..."
                    if ',' in subject:
                        potential_journal = subject.split(',')[0].strip()
                        if not potential_journal.lower().startswith('doi'):
                            metadata.journal = potential_journal
                            if verbose:
                                print(f"  PDF metadata journal: {metadata.journal}")

                # Try to get year from creation date
                if info.creation_date:
                    date_str = str(info.creation_date)
                    year_match = re.search(r'(19|20)\d{2}', date_str)
                    if year_match:
                        metadata.year = year_match.group()
                        if verbose:
                            print(f"  PDF metadata year: {metadata.year}")

    except Exception as e:
        if verbose:
            print(f"  Error reading PDF metadata: {e}")

    return metadata


def extract_metadata_from_text(pdf_path: Path, verbose: bool = False) -> PaperMetadata:
    """Extract metadata from PDF text content using heuristics.

    Uses a tiered approach:
    1. Try pdfplumber text extraction
    2. If text is too short, fall back to OCR
    """
    metadata = PaperMetadata()
    text = ""

    # Tier 1: Try pdfplumber text extraction
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if pdf.pages:
                    first_page = pdf.pages[0]
                    text = first_page.extract_text() or ""
                    if verbose:
                        print(f"  Extracted {len(text)} characters from first page")
        except Exception as e:
            if verbose:
                print(f"  pdfplumber extraction failed: {e}")
    elif verbose:
        print("  pdfplumber not available")

    # Tier 2: If text extraction yielded little content, try OCR
    if len(text) < MIN_TEXT_LENGTH:
        if verbose:
            print(f"  Text too short ({len(text)} < {MIN_TEXT_LENGTH}), trying OCR...")
        ocr_text = extract_text_via_ocr(pdf_path, pages=2, verbose=verbose)
        if len(ocr_text) > len(text):
            text = ocr_text
            if verbose:
                print(f"  Using OCR text ({len(text)} chars)")

    if not text:
        return metadata

    # Extract DOI
    doi = extract_doi(text)
    if doi:
        metadata.doi = doi
        if verbose:
            print(f"  Found DOI in text: {doi}")

    # Extract year - prefer position-based (filters download dates, prefers header/footer)
    year = extract_year_by_position(pdf_path, verbose)
    if not year:
        # Fallback to text-based extraction
        year = extract_year_from_text(text)
        if year and verbose:
            print(f"  Found year in text: {year}")
    if year:
        metadata.year = year

    # Extract journal
    journal = extract_journal_from_text(text)
    if journal:
        metadata.journal = journal
        if verbose:
            print(f"  Found journal in text: {journal}")

    # Extract title and authors (more complex heuristics)
    title, authors = extract_title_and_authors(text, verbose)
    if title:
        metadata.title = title
    if authors:
        metadata.authors = authors

    # Fallback: try font-size-based extraction for multi-column layouts
    if not metadata.title or not metadata.authors:
        if verbose:
            print("  Text heuristics incomplete, trying font-size analysis...")
        font_title, font_authors = extract_title_and_authors_by_font_size(pdf_path, verbose)
        if font_title and not metadata.title:
            metadata.title = font_title
        if font_authors and not metadata.authors:
            metadata.authors = font_authors

    return metadata


def extract_year_from_text(text: str) -> Optional[str]:
    """Extract publication year from text."""
    # Look for year patterns near common keywords
    patterns = [
        r'(?:published|received|accepted|online)[^\d]*(\d{4})',
        r'(\d{4})[^\d]*(?:published|received|accepted)',
        r'Vol\.?\s*\d+[^\d]*(\d{4})',
        r'(\d{4}),?\s*Vol',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            year = match.group(1)
            # Sanity check: year should be reasonable
            if 1900 <= int(year) <= 2100:
                return year

    # Fallback: find any 4-digit year in reasonable range
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    if years:
        # Return the most common year, or first if tie
        from collections import Counter
        return Counter(years).most_common(1)[0][0]

    return None


def extract_journal_from_text(text: str) -> Optional[str]:
    """Extract journal name from text."""
    # Common journal name patterns
    known_journals = [
        'Nature', 'Science', 'Cell', 'PNAS', 'JAMA', 'Lancet', 'BMJ',
        'Nature Reviews', 'Nature Genetics', 'Nature Medicine', 'Nature Methods',
        'Nature Communications', 'Nature Cancer',
        'Nucleic Acids Research', 'Genome Research', 'Genome Biology',
        'PLoS ONE', 'PLoS Biology', 'PLoS Genetics',
        'eLife', 'EMBO', 'Molecular Cell', 'Cancer Cell',
        'Journal of', 'Annals of', 'Proceedings of',
    ]

    lines = text.split('\n')

    # Check first few lines for journal names
    for line in lines[:10]:
        line = line.strip()
        for journal in known_journals:
            if journal.lower() in line.lower():
                # Try to extract the full journal name
                # Look for the journal name phrase
                pattern = rf'({re.escape(journal)}[^,\n]*?)(?:,|\n|doi|\d{{4}}|$)'
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                return journal

    return None


def extract_title_and_authors(text: str, verbose: bool = False) -> tuple[Optional[str], list[str]]:
    """Extract title and authors from text using heuristics."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    title = None
    authors = []

    # First, look for ALL CAPS titles (common in older journals like Nature, Science)
    # These are highly distinctive and can appear anywhere in the text
    for i, line in enumerate(lines[:50]):  # Search more lines for ALL CAPS
        # Check if line is mostly uppercase (title-like)
        words = line.split()
        if len(words) >= 2 and len(line) >= 10:
            # Count uppercase words (excluding small words)
            upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            if upper_words >= 2 and upper_words / len(words) > 0.5:
                # This looks like an ALL CAPS title
                # Check it's not a reference or header
                if not re.match(r'^\d+\.', line) and 'REFERENCES' not in line:
                    title = line
                    # Check if next line is also ALL CAPS (multi-line title)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        next_words = next_line.split()
                        if next_words:
                            next_upper = sum(1 for w in next_words if w.isupper() and len(w) > 1)
                            if next_upper / len(next_words) > 0.5:
                                title = f"{line} {next_line}"
                    if verbose:
                        print(f"  Found ALL CAPS title: {title[:50]}...")
                    # Look for authors in the next few lines
                    start = i + 2 if i + 1 < len(lines) and lines[i+1].isupper() else i + 1
                    for j in range(start, min(start + 5, len(lines))):
                        if is_author_line(lines[j]):
                            authors = parse_author_line(lines[j])
                            break
                    if title:
                        break

    # If no ALL CAPS title found, use standard heuristics
    if not title:
        # Heuristic: Title is usually one of the first substantial lines
        # that's not a journal name, date, or URL
        skip_patterns = [
            r'^https?://',
            r'^\d{4}$',
            r'^vol\.?\s*\d+',
            r'^doi:',
            r'^published',
            r'^received',
            r'^\d+\s*$',  # Just a number
        ]

        # Known journals to skip
        journal_keywords = ['nature', 'science', 'cell', 'journal', 'review', 'proceedings']

        title_candidates = []

        for i, line in enumerate(lines[:20]):  # Look at first 20 lines
            # Skip short lines or lines matching skip patterns
            if len(line) < 10:
                continue

            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break

            if skip:
                continue

            # Skip lines that look like journal names
            line_lower = line.lower()
            if any(j in line_lower for j in journal_keywords):
                continue

            # Check if this looks like an author line
            # Author lines often have: names with initials, commas, "and", affiliations with numbers
            if is_author_line(line):
                # The title is likely the previous substantial line
                if title_candidates:
                    title = title_candidates[-1]
                authors = parse_author_line(line)
                break

            # This could be a title
            title_candidates.append(line)

        # If we didn't find authors but have title candidates, use the longest one
        if not title and title_candidates:
            # Filter out very long lines (likely abstracts)
            title_candidates = [t for t in title_candidates if len(t) < 300]
            if title_candidates:
                # Prefer lines that look like titles (capitalized, not too short)
                for candidate in title_candidates:
                    if len(candidate) > 20 and not candidate.endswith('.'):
                        title = candidate
                        break
                if not title:
                    title = title_candidates[0]

    # Clean up title
    if title:
        # Remove line breaks and extra whitespace
        title = ' '.join(title.split())
        # Remove trailing punctuation except ?
        title = re.sub(r'[.,:;]+$', '', title)

    if verbose:
        if title:
            print(f"  Extracted title: {title[:60]}...")
        if authors:
            print(f"  Extracted authors: {authors[0]}{'...' if len(authors) > 1 else ''}")

    return title, authors


def is_author_line(line: str) -> bool:
    """Check if a line looks like it contains author names."""
    # Author lines often have:
    # - Multiple names separated by commas or "and"
    # - Initials (capital letter followed by period)
    # - Superscript numbers (for affiliations)
    # - Email symbols or asterisks

    indicators = 0

    # Check for initials pattern
    if re.search(r'\b[A-Z]\.\s*[A-Z]', line):
        indicators += 1

    # Check for multiple commas (multiple authors)
    if line.count(',') >= 2:
        indicators += 1

    # Check for "and" between names
    if re.search(r'\s+and\s+', line, re.IGNORECASE):
        indicators += 1

    # Check for superscript-like numbers
    if re.search(r'\d+,?\s*(?:and|\s|$)', line):
        indicators += 1

    # Check for asterisks (corresponding author)
    if '*' in line:
        indicators += 1

    # Check for affiliation indicators
    if re.search(r'(?:University|Institute|Department|Center|Centre|Hospital)', line, re.IGNORECASE):
        indicators += 1

    return indicators >= 2


def parse_author_line(line: str) -> list[str]:
    """Parse author names from an author line."""
    # Remove affiliation numbers/superscripts
    line = re.sub(r'[\d\*†‡§]+', '', line)

    # Remove email addresses
    line = re.sub(r'\S+@\S+', '', line)

    # Split by common delimiters
    if ' and ' in line.lower():
        parts = re.split(r'\s+and\s+', line, flags=re.IGNORECASE)
    else:
        parts = line.split(',')

    authors = []
    for part in parts:
        part = part.strip()
        # Skip if it looks like an affiliation
        if re.search(r'(?:University|Institute|Department|Center|Centre|Hospital)', part, re.IGNORECASE):
            continue
        # Skip if too short or too long
        if len(part) < 3 or len(part) > 50:
            continue
        # Should have at least one capital letter
        if not re.search(r'[A-Z]', part):
            continue
        authors.append(part)

    return authors


def is_garbage_title(title: str) -> bool:
    """Check if a title looks like garbage (filename, placeholder, etc.)."""
    if not title:
        return True
    title_lower = title.lower()
    # Looks like a filename
    if re.search(r'\.(pdf|indd|doc|docx|tex|indb)$', title_lower):
        return True
    # Contains "untitled" or similar
    if 'untitled' in title_lower or 'unknown' in title_lower:
        return True
    # All digits or mostly non-alphabetic
    alpha_count = sum(1 for c in title if c.isalpha())
    if alpha_count < len(title) * 0.3:
        return True
    # Very short
    if len(title.split()) < 2:
        return True
    return False


def merge_metadata(primary: PaperMetadata, secondary: PaperMetadata) -> PaperMetadata:
    """Merge two metadata objects, preferring primary values.

    Special handling:
    - Year: if primary year is 10+ years newer than secondary, prefer secondary
      (handles scanned documents where PDF metadata has the scan date)
    - Title: if primary title looks like garbage (filename, etc.), prefer secondary
    """
    # Smart year selection
    year = primary.year or secondary.year
    if primary.year and secondary.year:
        try:
            primary_int = int(primary.year)
            secondary_int = int(secondary.year)
            # If primary is 10+ years newer, it's probably a scan date
            if primary_int - secondary_int >= 10:
                year = secondary.year
        except ValueError:
            pass  # Keep default behavior if years aren't valid integers

    # Smart title selection - prefer non-garbage title
    title = primary.title or secondary.title
    if primary.title and secondary.title:
        if is_garbage_title(primary.title) and not is_garbage_title(secondary.title):
            title = secondary.title

    return PaperMetadata(
        title=title,
        authors=primary.authors or secondary.authors,
        year=year,
        journal=primary.journal or secondary.journal,
        doi=primary.doi or secondary.doi,
    )


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Sanitize string for use as filename."""
    # Replace illegal characters
    illegal = r'[<>:"/\\|?*\x00-\x1f]'
    name = re.sub(illegal, '_', name)

    # Replace multiple underscores/spaces
    name = re.sub(r'[_\s]+', '_', name)

    # Remove leading/trailing underscores and periods
    name = name.strip('_. ')

    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length].rsplit('_', 1)[0]  # Try to break at word boundary

    return name


def format_filename(metadata: PaperMetadata, format_str: str, short_title_words: int = 5, use_keywords: bool = True) -> str:
    """Format filename using template and metadata."""
    # Build substitution dict
    subs = {
        'first_author': sanitize_filename(metadata.first_author or 'Unknown'),
        'last_author': sanitize_filename(metadata.last_author or 'Unknown'),
        'first_author_full': sanitize_filename(metadata.first_author_full or 'Unknown'),
        'title': sanitize_filename(metadata.title or 'Untitled'),
        'short_title': sanitize_filename(metadata.short_title(short_title_words, use_keywords=use_keywords) or 'Untitled'),
        'year': metadata.year or 'NoYear',
        'journal': sanitize_filename(metadata.journal or 'Unknown'),
        'journal_abbrev': sanitize_filename(metadata.journal[:20] if metadata.journal else 'Unknown'),
    }

    # Perform substitution
    result = format_str
    for key, value in subs.items():
        result = result.replace(f'{{{key}}}', value)

    # Final sanitization
    result = sanitize_filename(result)

    return result


def extract_all_metadata(pdf_path: Path, use_crossref: bool = True, verbose: bool = False) -> PaperMetadata:
    """Extract metadata using all available methods."""
    if verbose:
        print(f"\nExtracting metadata from: {pdf_path.name}")

    # Start with PDF metadata
    metadata = extract_metadata_from_pdf(pdf_path, verbose)

    # Extract from text
    text_metadata = extract_metadata_from_text(pdf_path, verbose)
    metadata = merge_metadata(metadata, text_metadata)

    # If we have a DOI and CrossRef is enabled, try to get authoritative data
    if use_crossref and metadata.doi:
        crossref_metadata = query_crossref(metadata.doi, verbose)
        if crossref_metadata and crossref_metadata.is_complete():
            # CrossRef data is authoritative, prefer it
            metadata = merge_metadata(crossref_metadata, metadata)

    return metadata


def rename_pdf(
    pdf_path: Path,
    format_str: str,
    output_dir: Optional[Path] = None,
    short_title_words: int = 5,
    use_crossref: bool = True,
    use_keywords: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    force: bool = False,
) -> tuple[Path, Optional[Path], Optional[str]]:
    """
    Rename a PDF file based on extracted metadata.

    Args:
        force: If True, allow renaming with at least 2 of 3 core fields
               (title, authors, year) instead of requiring all 3.

    Returns: (original_path, new_path, error_message)
    """
    if not pdf_path.exists():
        return pdf_path, None, f"File not found: {pdf_path}"

    if not pdf_path.suffix.lower() == '.pdf':
        return pdf_path, None, f"Not a PDF file: {pdf_path}"

    # Extract metadata
    metadata = extract_all_metadata(pdf_path, use_crossref, verbose)

    if verbose:
        print(f"  Final metadata:")
        print(f"    Title: {metadata.title}")
        print(f"    Authors: {metadata.authors}")
        print(f"    Year: {metadata.year}")
        print(f"    Journal: {metadata.journal}")
        print(f"    DOI: {metadata.doi}")

    # Check completeness - force mode allows 2 of 3 fields
    missing = []
    if not metadata.title:
        missing.append('title')
    if not metadata.authors:
        missing.append('authors')
    if not metadata.year:
        missing.append('year')

    if force:
        if not metadata.is_minimally_viable():
            return pdf_path, None, f"Insufficient metadata (missing: {', '.join(missing)}; need at least 2 of 3)"
    else:
        if not metadata.is_complete():
            return pdf_path, None, f"Incomplete metadata (missing: {', '.join(missing)})"

    # Generate new filename
    new_name = format_filename(metadata, format_str, short_title_words, use_keywords)
    new_name = f"{new_name}.pdf"

    # Determine output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / new_name
    else:
        new_path = pdf_path.parent / new_name

    # Handle duplicate names
    if new_path.exists() and new_path != pdf_path:
        base = new_path.stem
        suffix = 1
        while new_path.exists():
            new_path = new_path.parent / f"{base}_{suffix}.pdf"
            suffix += 1

    # Perform rename
    if not dry_run and new_path != pdf_path:
        try:
            if output_dir:
                # Copy to new location
                import shutil
                shutil.copy2(pdf_path, new_path)
            else:
                # Rename in place
                pdf_path.rename(new_path)
        except OSError as e:
            return pdf_path, None, f"Failed to rename: {e}"

    return pdf_path, new_path, None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rename scientific PDF files based on extracted metadata.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Template variables:
  {first_author}      - First author's last name
  {last_author}       - Last author's last name
  {first_author_full} - First author's full name
  {title}             - Full title
  {short_title}       - First N words of title (see --short-title-words)
  {year}              - Publication year
  {journal}           - Journal name
  {journal_abbrev}    - Abbreviated journal name

Examples:
  %(prog)s paper.pdf
  %(prog)s --dry-run *.pdf
  %(prog)s -f "{year}_{first_author}_{short_title}" paper.pdf
  %(prog)s -o ./renamed/ -v *.pdf
"""
    )

    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='PDF files to rename'
    )
    parser.add_argument(
        '-f', '--format',
        default='{first_author}_{year}_{short_title}',
        help='Output filename format (default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        help='Output directory (default: same as source, copies files)'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Show what would be renamed without making changes'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed extraction information'
    )
    parser.add_argument(
        '--short-title-words',
        type=int,
        default=5,
        help='Number of words for {short_title} (default: %(default)s)'
    )
    parser.add_argument(
        '--no-crossref',
        action='store_true',
        help='Disable CrossRef API lookups'
    )
    parser.add_argument(
        '--no-keywords',
        action='store_true',
        help='Use simple first-N-words for title instead of keyword extraction'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rename files with incomplete metadata (requires at least 2 of: title, authors, year)'
    )

    args = parser.parse_args()

    # Check dependencies
    if not HAS_PYPDF and not HAS_PDFPLUMBER:
        print("Error: Neither pypdf nor pdfplumber is installed.", file=sys.stderr)
        print("Install with: pip install pypdf pdfplumber", file=sys.stderr)
        sys.exit(1)

    if not HAS_PYPDF:
        print("Warning: pypdf not installed, PDF metadata extraction disabled.", file=sys.stderr)

    if not HAS_PDFPLUMBER:
        print("Warning: pdfplumber not installed, text extraction disabled.", file=sys.stderr)

    # Process files
    success_count = 0
    error_count = 0

    for pdf_path in args.files:
        original, new_path, error = rename_pdf(
            pdf_path,
            format_str=args.format,
            output_dir=args.output_dir,
            short_title_words=args.short_title_words,
            use_crossref=not args.no_crossref,
            use_keywords=not args.no_keywords,
            dry_run=args.dry_run,
            verbose=args.verbose,
            force=args.force,
        )

        if error:
            print(f"ERROR: {original.name}: {error}", file=sys.stderr)
            error_count += 1
        elif new_path:
            action = "Would rename" if args.dry_run else "Renamed"
            if args.output_dir:
                action = "Would copy" if args.dry_run else "Copied"
            print(f"{action}: {original.name} -> {new_path.name}")
            success_count += 1
        else:
            print(f"Skipped: {original.name} (no changes needed)")

    # Summary
    if len(args.files) > 1:
        print(f"\nProcessed {len(args.files)} files: {success_count} renamed, {error_count} errors")

    sys.exit(1 if error_count > 0 else 0)


if __name__ == '__main__':
    main()
