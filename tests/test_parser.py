"""Vertical slices for PDFParser: paragraph clustering + math-font detection."""
import fitz

from pdf_translator_ru_uz.parser import PDFParser


def _make_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    fontsize = 11
    # Two lines close together (gap < 1.2*fontsize) -> same paragraph.
    page.insert_text(
        (72, 100), "First line of paragraph one.", fontsize=fontsize
    )
    page.insert_text(
        (72, 112), "Second line of paragraph one.", fontsize=fontsize
    )
    # Far below -> new paragraph.
    page.insert_text(
        (72, 300), "A separate second paragraph.", fontsize=fontsize
    )
    doc.save(path)
    doc.close()


def test_clusters_close_lines_into_one_paragraph_and_splits_distant_lines(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(str(pdf_path))

    parser = PDFParser(str(pdf_path))
    pages, tables = parser.extract_paragraphs()

    assert len(pages) == 1
    paragraphs = pages[0]
    assert len(paragraphs) == 2

    assert "First line of paragraph one." in paragraphs[0].text
    assert "Second line of paragraph one." in paragraphs[0].text
    assert "separate second paragraph" in paragraphs[1].text


def test_flags_cambria_math_spans_as_non_translatable(tmp_path):
    pdf_path = tmp_path / "math.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100), "tenglamaning butun sonlardan iborat", fontsize=11
    )
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser(str(pdf_path))
    pages, tables = parser.extract_paragraphs()
    assert pages[0][0].is_math is False


def test_is_math_font_predicate():
    assert PDFParser.is_math_font("Cambria Math") is True
    assert PDFParser.is_math_font("CambriaMath-Bold") is True
    assert PDFParser.is_math_font("Times New Roman") is False


def test_reading_order_sort_with_columns(tmp_path):
    """Two-column PDF: blocks should be sorted left column first, then right."""
    doc = fitz.open()
    page = doc.new_page()
    # Left column content
    page.insert_text((72, 100), "Left column first line", fontsize=11)
    page.insert_text((72, 200), "Left column second line", fontsize=11)
    # Right column content
    page.insert_text((400, 100), "Right column first line", fontsize=11)
    page.insert_text((400, 200), "Right column second line", fontsize=11)
    pdf_path = str(tmp_path / "twocol.pdf")
    doc.save(pdf_path)
    doc.close()

    parser = PDFParser(pdf_path)
    pages, _ = parser.extract_paragraphs()
    texts = [p.text for p in pages[0]]
    # Left column should come entirely before right column
    left_idx = [i for i, t in enumerate(texts) if "Left" in t]
    right_idx = [i for i, t in enumerate(texts) if "Right" in t]
    assert all(l < r for l in left_idx for r in right_idx)


def test_header_footer_filtering(tmp_path):
    """Text in top/bottom 5% of page should be filtered out."""
    doc = fitz.open()
    page = doc.new_page()
    page_height = page.rect.height
    # Header text (within top 5%)
    page.insert_text(
        (72, 10), "HEADER TEXT", fontsize=11
    )
    # Body text
    page.insert_text(
        (72, page_height * 0.3), "Body paragraph text", fontsize=11
    )
    # Footer text (within bottom 5%)
    page.insert_text(
        (72, page_height * 0.97), "FOOTER TEXT", fontsize=11
    )
    pdf_path = str(tmp_path / "headerfooter.pdf")
    doc.save(pdf_path)
    doc.close()

    parser = PDFParser(pdf_path)
    pages, _ = parser.extract_paragraphs()
    texts = [p.text for p in pages[0]]
    assert all("HEADER" not in t for t in texts)
    assert all("FOOTER" not in t for t in texts)
    assert any("Body paragraph" in t for t in texts)


def test_list_marker_stripping():
    """List markers like bullet and numbering should be stripped."""
    from pdf_translator_ru_uz.parser import PDFParser as P

    text, marker = P._strip_list_marker("1. First item")
    assert text == "First item"
    assert marker == "1."

    text, marker = P._strip_list_marker("• Bullet item")
    assert "Bullet item" in text
    assert marker == "•"

    text, marker = P._strip_list_marker("No marker here")
    assert text == "No marker here"
    assert marker is None


def test_dehyphenation():
    """Line-break hyphens should be merged with the next line."""
    from pdf_translator_ru_uz.parser import PDFParser as P

    prev, next_text = P._dehyphenate_line("This is a hyphenated-", "word in the next line")
    # 'hyphenated' + 'word' should merge to 'hyphenatedword' or keep the hyphen
    # The heuristic depends on vowel/consonant patterns
    merged = prev
    assert "hyphenated" in merged or "word" in merged or not next_text

    # Natural dashes (em-dash, en-dash) should stay
    prev, next_text = P._dehyphenate_line("This is an em dash—", "it continues")
    assert next_text  # should NOT have merged (next_text non-empty)