"""Runs every page headless: python smoke_test.py"""
import time
from streamlit.testing.v1 import AppTest
PAGES = ["Case overview", "1 · Geomechanics along the lateral", "2 · Stress profile and fracture height",
         "3 · What controls fracture effectiveness", "4 · Model validation", "5 · Business integration",
         "6 · Real-time drilling", "Findings and Day 1 wrap-up"]
for p in PAGES:
    t = time.time(); at = AppTest.from_file("app.py", default_timeout=600); at.run()
    at.sidebar.radio[0].set_value(p).run()
    print(f"{p:45s} {time.time()-t:5.1f}s exceptions={[str(e.value)[:300] for e in at.exception]}")
