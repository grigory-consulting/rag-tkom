import tempfile
import unittest
import warnings
from pathlib import Path

from common.corpus import ACL_DEFAULT, _parse, load_corpus


class CorpusParsingTests(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.md"
            path.write_text(text, encoding="utf-8")
            return _parse(path)

    def test_parses_bom_before_frontmatter(self):
        doc = self.parse_text(
            "\ufeff---\ndoc_id: bom\nacl: vertraulich\n---\nGeheim."
        )

        self.assertEqual(doc.doc_id, "bom")
        self.assertEqual(doc.acl, "vertraulich")

    def test_parses_leading_whitespace_before_frontmatter(self):
        doc = self.parse_text(
            "\n  \n---\ndoc_id: whitespace\nacl: wartung\n---\nIntern."
        )

        self.assertEqual(doc.doc_id, "whitespace")
        self.assertEqual(doc.acl, "wartung")

    def test_removes_inline_comment_from_acl(self):
        doc = self.parse_text(
            "---\ndoc_id: comment\n"
            "acl: vertraulich   # all | wartung | vertraulich\n"
            "---\nGeheim."
        )

        self.assertEqual(doc.acl, "vertraulich")

    def test_missing_acl_uses_restrictive_default_and_warns(self):
        with self.assertWarnsRegex(RuntimeWarning, "keine ACL"):
            doc = self.parse_text("Dokument ohne Frontmatter.")

        self.assertEqual(doc.acl, ACL_DEFAULT)
        self.assertEqual(doc.acl, "vertraulich")

    def test_unknown_acl_raises_error(self):
        with self.assertRaisesRegex(ValueError, "ungültige ACL-Stufe"):
            self.parse_text(
                "---\ndoc_id: typo\nacl: vertaulich\n---\nGeheim."
            )

    def test_unclosed_frontmatter_raises_error(self):
        with self.assertRaisesRegex(ValueError, "nicht abgeschlossen"):
            self.parse_text("---\ndoc_id: broken\nacl: all\nText")

    def test_public_filter_does_not_leak_document_without_acl(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus_dir = Path(directory)
            (corpus_dir / "public.md").write_text(
                "---\ndoc_id: public\nacl: all\n---\nÖffentlich.",
                encoding="utf-8",
            )
            (corpus_dir / "missing.md").write_text(
                "Kein Frontmatter.", encoding="utf-8"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                docs = load_corpus({"all"}, corpus_dir=corpus_dir)

        self.assertEqual([doc.doc_id for doc in docs], ["public"])


if __name__ == "__main__":
    unittest.main()
