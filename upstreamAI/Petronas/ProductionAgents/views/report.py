import io
import zipfile

import streamlit as st

from prodintel.report import build_zip
from prodintel.ui import header, qc_badges, require_run

header("Report and download", "Engineering-ready ZIP: PDF, CSV tables, figures, interactive 3D view, run manifest")
ctx = require_run()
t = ctx.results["target"]

st.markdown("**Summary**")
st.write(ctx.narrative)
st.markdown("**Quality control**")
if ctx.qc:
    qc_badges(ctx)
else:
    st.success("No QC flags raised.")

if "zip_bytes" not in st.session_state:
    with st.spinner("Assembling the report package"):
        st.session_state["zip_bytes"] = build_zip(ctx)
data = st.session_state["zip_bytes"]
stem = (t["api"] or f"proposed_{t['lat']:.4f}_{t['lon']:.4f}").replace("-", "")
st.download_button("Download ZIP report", data, file_name=f"{stem}_production_intelligence.zip",
                   mime="application/zip", type="primary")
with zipfile.ZipFile(io.BytesIO(data)) as z:
    st.caption("Contents")
    st.code("\n".join(f"{i.filename}  ({i.file_size / 1024:,.0f} KB)" for i in z.infolist()), language=None)
