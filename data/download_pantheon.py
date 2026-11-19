"""Download the Pantheon+ data used in Stage 3.

The two files come from the official Pantheon+SH0ES data release,
github.com/PantheonPlusSH0ES/DataRelease:

    Pantheon+SH0ES.dat             magnitudes and redshifts of 1701 light curves
    Pantheon+SH0ES_STAT+SYS.cov    their 1701 x 1701 covariance matrix (33 MB)

Run:  python data/download_pantheon.py
"""

import urllib.parse
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
        "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
FILES = ["Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"]

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name in FILES:
        target = here / name
        if target.exists():
            print(f"{name}: already here")
            continue
        print(f"{name}: downloading ...")
        urllib.request.urlretrieve(BASE + urllib.parse.quote(name), target)
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB")
