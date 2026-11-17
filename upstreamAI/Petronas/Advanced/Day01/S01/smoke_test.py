"""Headless check: open every page of app.py and report exceptions.
Page names are read from the app's own sidebar, so renaming a page never breaks this test."""
import time
from streamlit.testing.v1 import AppTest

first = AppTest.from_file("app.py", default_timeout=600)
first.run()
pages = list(first.sidebar.radio[0].options)
for p in pages:
    t = time.time()
    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value(p).run()
    errs = [e.value for e in at.exception]
    print(f"{p:35s} {time.time()-t:6.1f}s  exceptions={len(errs)}")
    for e in errs:
        print("   ", str(e)[:1500])
    if p.startswith("Findings") and not errs:
        for m in at.metric:
            print(f"    {m.label}: {m.value}")
        print(at.dataframe[0].value.to_string())
