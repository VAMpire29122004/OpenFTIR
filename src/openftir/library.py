"""
Parses the reference CSV into PeakRule objects.

This is the ONLY place reference-CSV parsing happens. ftir_plot's old standalone
parse_range()/get_tier() are retired — plotting.py now goes through this class,
via FunctionalGroupClassifier, so there is one detection code path, not two.
"""

import pandas as pd
from .models import PeakRule


class ReferenceLibrary:
    """
    Peak Position can be:
      "1705-1725"  -> range
      "1760"       -> single wavenumber, expanded to +-5 cm-1
    Rows with unparseable positions (#ERROR! etc.) are silently skipped.

    Accepts a file path OR a file-like object (e.g. a Streamlit UploadedFile),
    since pandas.read_csv handles both transparently.
    """

    def __init__(self, path_or_buffer):
        self.rules: list[PeakRule] = []
        self._load(path_or_buffer)

    def _parse_position(self, pos):
        pos = str(pos).strip().replace("≈", "").replace("~", "")
        if "-" in pos:
            parts = pos.split("-")
            return float(parts[0]), float(parts[1])
        val = float(pos)
        return val - 5.0, val + 5.0

    def _load(self, path_or_buffer):
        df = pd.read_csv(path_or_buffer)
        # Column-name whitespace crept into this from the v2 app cleanup —
        # keeping it here means every caller gets it, not just the app.
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            try:
                lo, hi = self._parse_position(row["Peak Position"])
            except Exception:
                continue
            group = str(row.get("Group", "")).strip()
            cls   = str(row.get("Class", "")).strip()
            det   = str(row.get("Peak Details", "")).strip()
            if group and cls:
                self.rules.append(PeakRule(lo, hi, group, cls, det))

    def __len__(self):
        return len(self.rules)
