import time, sys
from streamlit.testing.v1 import AppTest
pages = ["Case overview", "1 · QC and alignment", "2 · XRD mineralogy", "3 · Rock-Eval and fluids",
         "Rock typing", "4 · Permeability models", "5 · Well-test check", "Findings for slide 22"]
for p in pages:
    t=time.time()
    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value(p).run()
    errs = [e.value for e in at.exception]
    print(f"{p:35s} {time.time()-t:6.1f}s  exceptions={len(errs)}")
    for e in errs: print("   ", str(e)[:1500])
    if p.startswith("Findings") and not errs:
        print(at.dataframe[0].value.to_string())
