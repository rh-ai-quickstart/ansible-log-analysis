#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import uuid
from typing import List, Dict, Any
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)


class AnsibleErrorParser:
    """
    Parses Ansible error documentation PDFs into structured chunks.

    Fixes:
      - Robust paragraph reflow (no more one-word-per-line).
      - Conservative code detection (won't misclassify lightly-indented prose).
      - Optional 'Benefits …' section support.
      - Handles leading whitespace before numbered titles.
    """

    def __init__(self):
        # Allow optional leading whitespace before the number (e.g., "  1. Error: …")
        self.error_title_pattern = re.compile(r"^\s*\d+\.\s*.+$", re.MULTILINE)

        # Section boundaries (order matters for lookaheads)
        # We support 'Benefits:', 'Benefits of …:', etc.
        self.re_desc = re.compile(
            r"Description:\s*(.*?)(?=Symptoms:|Resolution:|Code:|Benefits|$)",
            re.IGNORECASE | re.DOTALL,
        )
        self.re_symp = re.compile(
            r"Symptoms:\s*(.*?)(?=Resolution:|Code:|Benefits|$)",
            re.IGNORECASE | re.DOTALL,
        )
        self.re_reso = re.compile(
            r"Resolution:\s*(.*?)(?=Code:|Benefits|$)", re.IGNORECASE | re.DOTALL
        )
        # Stop Code at Benefits, next numbered error, or EOF
        self.re_code = re.compile(
            r"Code:?\s*\n(.*?)(?=Benefits|^\s*\d+\.\s*Error\s+\d+|$)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        # Benefits headers vary: "Benefits:", "Benefits of …:", "Benefits of Following …:"
        self.re_bens = re.compile(
            r"Benefits(?:\s+of[^\n:]*)?:\s*(.*?)(?=^\s*(?:Description|Symptoms|Resolution|Code)\s*:|^\s*\d+\.\s*[A-Z]|$)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )

    # ------------------------------------------------------------------
    # Loading & page-level reflow
    # ------------------------------------------------------------------

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and return reflowed documents with page metadata."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Reflow each page's content to undo hard wraps while preserving bullets/code.
        for i, doc in enumerate(documents):
            documents[i] = Document(
                page_content=self._reflow_text(doc.page_content), metadata=doc.metadata
            )

        logger.debug("Loaded PDF: %s", pdf_path)
        logger.debug("  Total pages: %d", len(documents))
        return documents

    def _reflow_text(self, text: str) -> str:
        """
        Convert hard-wrapped lines to paragraphs while preserving:
        - blank lines,
        - bullets/list items,
        - real code/config blocks,
        - section headers and error titles.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"\s+$", "", ln) for ln in text.split("\n")]
        return self._unwrap_paragraphs(lines)

    def _is_bullet(self, ln: str) -> bool:
        return bool(re.match(r"^\s*(?:[-*•●▪○]|[0-9]+[.)])\s+", ln))

    def _looks_like_header(self, ln: str) -> bool:
        return bool(
            re.match(
                r"^\s*(?:Description|Symptoms|Resolution|Code|Benefits)(?:\s+of[^\n:]*)?:?\s*$",
                ln,
                flags=re.I,
            )
        )

    def _looks_like_next_error(self, ln: str) -> bool:
        # "  12. Error: …"
        return bool(re.match(r"^\s*\d+\.\s+[A-Z]", ln))

    def _looks_like_code_line(self, ln: str, prev: str, in_code_block: bool) -> bool:
        """
        Conservative code detection to avoid breaking paragraphs:
        - Inside ``` blocks, everything counts as code until closing fence.
        - Indentation threshold: >= 6 spaces (not 2).
        - YAML-style keys ('key:') only count as code if there's >= 2-space indent.
        - Jinja/pipe lines need >= 2-space indent to be considered code.
        """
        s = ln.rstrip("\n")
        if s.strip().startswith("```"):
            return True
        if in_code_block:
            return True

        indent = len(s) - len(s.lstrip(" "))

        if indent >= 6:
            return True

        # bullets are NOT code
        if self._is_bullet(s):
            return False

        # YAML-ish key lines considered code only with some indent
        if indent >= 2 and re.search(r":\s*$", s):
            return True

        # Jinja or pipes considered code only with some indent
        if indent >= 2 and re.search(r"\{\{.*\}\}|\|\s*\w+", s):
            return True

        return False

    def _unwrap_paragraphs(self, lines: List[str]) -> str:
        """
        Merge wrapped prose lines into paragraphs while preserving:
        - bullets, headers, next-error markers,
        - real code blocks (fenced or sufficiently indented),
        - hard paragraph breaks (>= 2 consecutive blank lines).
        Treat a single blank line as a soft break (do not flush).
        Also fixes hyphenated wraps: 'configu-\\nration' → 'configuration'
        """
        out: List[str] = []
        buf: List[str] = []
        prev = ""
        in_code_block = False
        pending_blank = False  # one blank line seen; decide later

        def flush():
            nonlocal buf
            if not buf:
                return
            paragraph = " ".join(buf)
            # de-hyphenate splits
            paragraph = re.sub(r"(\w)-\s+(\w)", r"\1\2", paragraph)
            # collapse excessive internal spacing
            paragraph = re.sub(r"[ \t]{2,}", " ", paragraph)
            out.append(paragraph.strip())
            buf = []

        for raw in lines:
            ln = raw.rstrip("\n")

            # fenced code toggles
            if ln.strip().startswith("```"):
                if pending_blank:
                    # a pending blank becomes real before code fences
                    flush()
                    out.append("")
                    pending_blank = False
                if in_code_block:
                    out.append(ln)  # closing fence
                    in_code_block = False
                else:
                    flush()
                    out.append(ln)  # opening fence
                    in_code_block = True
                prev = raw
                continue

            # blank line handling
            if ln.strip() == "":
                if pending_blank:
                    # two blanks → hard paragraph break
                    flush()
                    out.append("")
                    pending_blank = False
                else:
                    # one blank → soft break (hold; may be ignored if prose continues)
                    pending_blank = True
                prev = raw
                continue

            # headers / bullets / next-error markers: finalize any pending blank and flush
            if (
                self._is_bullet(raw)
                or self._looks_like_header(raw)
                or self._looks_like_next_error(raw)
            ):
                if pending_blank:
                    flush()
                    out.append("")
                    pending_blank = False
                flush()
                out.append(raw)
                prev = raw
                continue

            if self._looks_like_code_line(raw, prev, in_code_block):
                if pending_blank:
                    flush()
                    out.append("")
                    pending_blank = False
                flush()
                out.append(raw)
                prev = raw
                continue

            # normal prose
            if pending_blank:
                # soft break: just continue merging (do not emit a real blank)
                pending_blank = False

            if buf:
                # if previous chunk ends a sentence, just append; else join tightly
                if re.search(r"[.!?;:)\]]\s*$", buf[-1]):
                    buf.append(ln.strip())
                else:
                    buf[-1] = buf[-1].rstrip("-") + " " + ln.strip()
            else:
                buf.append(ln.strip())

            prev = raw

        # end of loop
        flush()
        if pending_blank:
            out.append("")  # file ended after a single blank; keep one

        text = "\n".join(out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Error extraction
    # ------------------------------------------------------------------

    def extract_errors_from_documents(
        self, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extract individual error entries from the (reflowed) PDF documents."""
        full_text = "\n".join([doc.page_content for doc in documents])

        error_matches = list(self.error_title_pattern.finditer(full_text))
        logger.debug("Found %d error title matches", len(error_matches))
        for i, match in enumerate(error_matches[:5]):
            logger.debug("  Match %d: %s", i + 1, match.group(0).strip()[:60])

        errors = []
        for i, match in enumerate(error_matches):
            error_start = match.start()
            error_end = (
                error_matches[i + 1].start()
                if i + 1 < len(error_matches)
                else len(full_text)
            )

            error_text = full_text[error_start:error_end]
            error_title = match.group(0).strip()

            page_num = self._find_page_number(documents, error_start)

            parsed_error = self._parse_error_sections(
                error_text=error_text,
                error_title=error_title,
                page=page_num,
                source_file=documents[0].metadata.get("source", "unknown"),
            )
            errors.append(parsed_error)

        logger.debug("Extracted %d error entries", len(errors))
        return errors

    def _find_page_number(self, documents: List[Document], char_position: int) -> int:
        """Approximate page for a character position (post-reflow)."""
        current_pos = 0
        for i, doc in enumerate(documents):
            current_pos += (
                len(doc.page_content) + 1
            )  # +1 for the newline added when joining
            if char_position < current_pos:
                return i + 1
        return len(documents)

    def _reflow_prose_block(self, text: str) -> str:
        """
        Final pass for prose sections: join single newlines (and single blank lines) into spaces,
        keep bullets / code-ish lines, and only treat 2+ blank lines as paragraph breaks.
        """
        if not text:
            return text

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        out: List[str] = []
        buf: List[str] = []
        pending_blank = False

        def flush():
            nonlocal buf
            if not buf:
                return
            # collapse multiple spaces inside paragraphs
            para = re.sub(r"[ \t]{2,}", " ", " ".join(buf)).strip()
            out.append(para)
            buf = []

        for ln in lines:
            if ln.strip() == "":
                if pending_blank:
                    flush()
                    out.append("")
                    pending_blank = False
                else:
                    pending_blank = True
                continue

            if self._is_bullet(ln) or self._looks_like_code_line(ln, "", False):
                if pending_blank:
                    flush()
                    out.append("")
                    pending_blank = False
                flush()
                out.append(ln)
                continue

            if pending_blank:
                pending_blank = False

            if buf:
                if re.search(r"[.!?;:)\]]\s*$", buf[-1]):
                    buf.append(ln.strip())
                else:
                    buf[-1] = buf[-1].rstrip("-")
                    buf.append(ln.strip())
            else:
                buf.append(ln.strip())

        flush()
        if pending_blank:
            out.append("")

        text = "\n".join(out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<!\w)(?:None)(?!\w)", "", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _parse_error_sections(
        self, error_text: str, error_title: str, page: int, source_file: str
    ) -> Dict[str, Any]:
        """Parse an error entry into component sections, including optional Benefits."""
        sections = {
            "error_title": error_title,
            "description": "",
            "symptoms": "",
            "resolution": "",
            "code": "",
            "benefits": "",
        }

        # Extract sections with robust lookaheads
        m = self.re_desc.search(error_text)
        if m:
            sections["description"] = m.group(1).strip()

        m = self.re_symp.search(error_text)
        if m:
            sections["symptoms"] = m.group(1).strip()

        m = self.re_reso.search(error_text)
        if m:
            sections["resolution"] = m.group(1).strip()

        m = self.re_code.search(error_text)
        if m:
            sections["code"] = m.group(1).strip()

        m = self.re_bens.search(error_text)
        if m:
            sections["benefits"] = m.group(1).strip()

        # Cleanup + final prose reflow for non-code sections
        for key in ["description", "symptoms", "resolution", "benefits"]:
            txt = sections[key]
            if not txt:
                continue
            txt = re.sub(r"(?im)^\s*none\s*$", "", txt)
            # Reflow prose to merge single newlines → spaces
            txt = self._reflow_prose_block(txt)
            sections[key] = txt

        if sections["code"]:
            sections["code"] = re.sub(r"(?im)^\s*none\s*$", "", sections["code"])
            sections["code"] = re.sub(r"\n{3,}", "\n\n", sections["code"]).strip()

        return {"sections": sections, "page": page, "source_file": source_file}

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def create_chunks(self, errors: List[Dict[str, Any]]) -> List[Document]:
        """Create LangChain Document chunks from parsed errors."""
        chunks: List[Document] = []

        for error in errors:
            error_id = str(uuid.uuid4())
            error_title = error["sections"]["error_title"]

            section_types = [
                "description",
                "symptoms",
                "resolution",
                "code",
                "benefits",
            ]

            for section_type in section_types:
                content = error["sections"].get(section_type, "")
                if not content or not content.strip():
                    continue

                metadata = {
                    "error_id": error_id,
                    "error_title": error_title,
                    "section_type": section_type,
                    "source_file": error["source_file"],
                    "page": error["page"],
                }

                chunk_content = f"Error: {error_title}\n\nSection: {section_type.capitalize()}\n\n{content}"
                chunk = Document(page_content=chunk_content, metadata=metadata)
                chunks.append(chunk)

        logger.debug("Created %d chunks from %d errors", len(chunks), len(errors))
        return chunks

    def parse_pdf_to_chunks(self, pdf_path: str) -> List[Document]:
        """Main method: Parse a PDF file into structured chunks."""
        documents = self.load_pdf(pdf_path)
        errors = self.extract_errors_from_documents(documents)
        chunks = self.create_chunks(errors)
        return chunks


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def export_metadata_to_json(documents, output_path="metadata_export.json"):
    """
    Export all metadata to JSON for analysis.
    """
    metadata_list = []
    for doc in documents:
        metadata = dict(doc.metadata) if doc.metadata else {}
        content = doc.page_content or ""
        metadata["content_preview"] = content[:200]
        metadata["content_length"] = len(content)
        metadata["error_title"] = metadata.get("error_title")
        metadata["section_type"] = metadata.get("section_type")
        metadata["page"] = metadata.get("page")
        metadata["source_file"] = metadata.get("source_file")
        metadata_list.append(metadata)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    logger.debug("Metadata exported to: %s", output_path)


def main():
    """Test the parser end-to-end."""
    logger.info("=" * 60)
    logger.info(
        "ANSIBLE ERROR KNOWLEDGE BASE - STEP 1: PDF PARSING (REFLOW + BENEFITS)"
    )
    logger.info("=" * 60)

    parser = AnsibleErrorParser()

    # Update this path to whichever file you're testing
    pdf_path = "/home/mtalvi/ansible-log-analysis/data/knowledge_base/file_7.pdf"
    chunks = parser.parse_pdf_to_chunks(pdf_path)

    # Export metadata for quick inspection
    export_metadata_to_json(
        chunks, output_path="/home/mtalvi/ansible-log-analysis/metadata_export.json"
    )

    logger.info("=" * 60)
    logger.info("SAMPLE CHUNKS (first 3):")
    logger.info("=" * 60)
    for c in chunks[:3]:
        logger.info(
            "[%s] %s", c.metadata.get("section_type"), c.metadata.get("error_title")
        )
        logger.info("%s...", c.page_content[:500])
    logger.info("Total chunks created: %d", len(chunks))

    return chunks


if __name__ == "__main__":
    chunks = main()
