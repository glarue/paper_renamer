# Paper Renamer

Automatically rename scientific PDF files based on extracted metadata (title, authors, year, journal) using a configurable format string.

## Installation

Requires [pixi](https://pixi.sh/):

```bash
git clone https://github.com/glarue/paper_renamer.git
cd paper_renamer
pixi install -e default  # Base install (~380MB)
```

For scanned/older PDFs, install with OCR support:

```bash
pixi install -e ocr  # With OCR (~750MB)
```

## Usage

```bash
# Basic usage - rename files in place
pixi run paper_renamer paper.pdf

# Preview changes without renaming
pixi run paper_renamer --dry-run *.pdf

# Custom format
pixi run paper_renamer -f "{year}_{first_author}_{journal}" paper.pdf

# Copy to new directory instead of renaming
pixi run paper_renamer -o ./renamed/ *.pdf

# Verbose output showing extraction details
pixi run paper_renamer -v paper.pdf

# Use simple first-N-words instead of keyword extraction
pixi run paper_renamer --no-keywords paper.pdf
```

Or activate the environment first:

```bash
pixi shell
paper_renamer --dry-run *.pdf
```

## Format Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{first_author}` | First author's last name | `Shen` |
| `{last_author}` | Last author's last name | `Seshan` |
| `{first_author_full}` | First author's full name | `Ronglai_Shen` |
| `{year}` | Publication year | `2016` |
| `{title}` | Full title | `FACETS_allele-specific_copy_number...` |
| `{short_title}` | Key phrases from title | `FACETS_allele-specific_high-throughput_DNA_sequencing` |
| `{journal}` | Journal name | `Nucleic_Acids_Research` |
| `{journal_abbrev}` | Abbreviated journal (20 chars) | `Nucleic_Acids_Resear` |

**Default format:** `{first_author}_{year}_{short_title}`

## Metadata Extraction

The tool extracts metadata using multiple strategies (in priority order):

1. **PDF metadata** - Embedded title, author, subject fields
2. **CrossRef API** - If a DOI is found, queries CrossRef for authoritative data
3. **Text heuristics** - Parses first page text to identify title, authors, journal, year
4. **Font-size analysis** - Falls back to finding the largest-font text cluster (handles multi-column layouts)
5. **OCR fallback** - For scanned PDFs with no text layer (optional, see below)

## Keyword Extraction

The `{short_title}` variable uses intelligent keyword extraction:

- **Technical term detection** - Prioritizes ALL CAPS (FACETS, DNA), CamelCase (cfDNA), hyphenated terms (allele-specific), and words with numbers (p53)
- **YAKE keywords** - Identifies statistically important terms
- **Original order preserved** - Selected keywords maintain their order from the title

Use `--no-keywords` to fall back to simple first-N-words truncation.

## Options

| Option | Description |
|--------|-------------|
| `-f, --format` | Output filename format (default: `{first_author}_{year}_{short_title}`) |
| `-o, --output-dir` | Directory for renamed files (copies instead of renaming) |
| `-n, --dry-run` | Show what would be renamed without making changes |
| `-v, --verbose` | Show detailed extraction information |
| `--short-title-words` | Number of words for `{short_title}` (default: 5) |
| `--no-crossref` | Disable CrossRef API lookups (offline mode) |
| `--no-keywords` | Use simple first-N-words instead of keyword extraction |
| `--force` | Rename with incomplete metadata (requires at least 2 of: title, authors, year) |

## Examples

Example PDFs are included in `example_pdf/` for testing.

```bash
# Preview renames on example files
$ pixi run paper_renamer --dry-run example_pdf/*.pdf
Would rename: Widman et al. 2024 - Nat Med.pdf -> Widman_2024_Ultrasensitive_plasma-based_monitoring_machine-learning-guided_signal.pdf
Would rename: chalfie1994.pdf -> Chalfie_1994_Green_Fluorescent_Protein_Marker_Gene.pdf
Would rename: gkw520.pdf -> Shen_2016_FACETS_allele-specific_high-throughput_DNA_sequencing.pdf
Would rename: s41467-024-55513-2.pdf -> Buyan_2025_calling_allelic_high-throughput_sequencing_data.pdf
ERROR: watson1953.pdf: Incomplete metadata (missing: authors)

# Use --force for files with incomplete metadata (e.g., very old scanned papers)
$ pixi run -e ocr paper_renamer --dry-run --force example_pdf/watson1953.pdf
Would rename: watson1953.pdf -> Unknown_1953_MOLECULAR_STRUCTURE_NUCLEIC_ACIDS.pdf
```

## Dependencies

**Core (installed by default):**
- `pypdf` - PDF metadata extraction
- `pdfplumber` - PDF text extraction
- `yake` - Keyword extraction

**Optional OCR (for scanned PDFs):**
- `tesseract` - OCR engine
- `poppler` - PDF to image conversion
- `pdf2image`, `pytesseract` - Python bindings

## OCR Support for Scanned PDFs

Older papers (pre-2000s) are often scanned images with no embedded text layer. The base installation cannot extract text from these. The included `watson1953.pdf` (Watson & Crick's DNA structure paper) demonstrates this:

```bash
# Without OCR: only garbage PDF metadata available
$ pixi run paper_renamer --dry-run --force -v example_pdf/watson1953.pdf
  Extracted 33 characters from first page
  Text too short (33 < 200), trying OCR...
  OCR not available (install pdf2image and pytesseract)
  Final metadata:
    Title: 171-4356indd.indd    # Garbage from PDF metadata
    Year: 2005                   # Scan date, not publication
Would rename: watson1953.pdf -> Unknown_2005_171-4356indd.indd.pdf

# With OCR: extracts actual title and year from scanned image
$ pixi run -e ocr paper_renamer --dry-run --force -v example_pdf/watson1953.pdf
  Attempting OCR on first 2 page(s)...
  OCR page 1: 6266 chars
  Using OCR text (11178 chars)
  Found year in text: 1953
  Found ALL CAPS title: MOLECULAR STRUCTURE OF NUCLEIC ACIDS...
  Final metadata:
    Title: MOLECULAR STRUCTURE OF NUCLEIC ACIDS
    Year: 1953
Would rename: watson1953.pdf -> Unknown_1953_MOLECULAR_STRUCTURE_NUCLEIC_ACIDS.pdf
```

To install OCR support:

```bash
pixi install -e ocr  # Adds ~370MB (tesseract + language data)
```

**When is OCR needed?**
- Papers from before ~1995 (often scanned from print)
- PDFs where text cannot be selected/copied
- Files where `pdftotext file.pdf -` produces little or no output

**How it works:**
1. If text extraction yields < 200 characters, OCR is attempted automatically
2. First 2 pages are converted to images and processed with Tesseract
3. Extracted text is then parsed using the same heuristics as normal PDFs

**Note:** OCR adds ~400MB to the environment (mostly Tesseract language data):
- Base environment: ~380MB
- OCR environment: ~750MB

## Known Limitations

- **Multi-article pages** - Older journals (Science, Nature pre-2000) sometimes have multiple articles per page. The font-size heuristic usually finds the correct title, but author extraction may fail if authors are in a complex multi-column format.
- **Very old scanned papers** - OCR helps but papers from before ~1960 may have unusual formatting that confuses the title/author heuristics.
- **Non-English papers** - Keyword extraction and heuristics are tuned for English text.
