r"""
Dispatcher for the image sources. Keeps argument order irrelevant, which batch
files are bad at.

  fetch_images.bat <out> --source pd12m --count 20000
  fetch_images.bat --source pd12m --count 20000 --output <out>
  fetch_images.bat <out> --count 20000                 (defaults to 4klsdb)

Sources:
  4klsdb  native-4K, very diverse, but aesthetic-filtered (Q-Align top 80%),
          which systematically favours shallow depth of field. CC-BY-4.0.
  pd12m   public domain (Europeana / Wikimedia Commons / Smithsonian) re-hosted
          on S3, so no Wikimedia rate limits, and overwhelmingly deep focus.
          CDLA-Permissive-2.0.
"""
from __future__ import annotations

import sys
from os import path

_TOOLS = path.dirname(path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

SOURCES = {"4klsdb": "fetch_4klsdb", "pd12m": "fetch_pd12m"}


def main():
    argv = sys.argv[1:]

    source = "4klsdb"
    if "--source" in argv:
        i = argv.index("--source")
        if i + 1 >= len(argv):
            print("--source needs a value: " + " | ".join(sorted(SOURCES)), file=sys.stderr)
            return 2
        source = argv[i + 1].lower()
        del argv[i:i + 2]
    if source not in SOURCES:
        print(f"unknown --source {source!r}; choose from {' | '.join(sorted(SOURCES))}",
              file=sys.stderr)
        return 2

    # allow the output folder as a bare first argument
    if argv and not argv[0].startswith("-") and "--output" not in argv:
        argv = ["--output", argv[0]] + argv[1:]

    mod = __import__(SOURCES[source])
    sys.argv = [SOURCES[source]] + argv
    print(f"source : {source}")
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
