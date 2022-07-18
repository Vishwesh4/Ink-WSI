import sys
from pathlib import Path
sys.path.append("/home/ramanav/Projects/Ink-WSI")

from modules.patch_extraction import SedeenAnnotationParser

# slide_path = Path("/home/ramanav/Downloads/121484.svs")
slide_path = Path("/home/ramanav/Downloads/103732.svs")
xml_path = slide_path.parent / Path(slide_path.stem + ".session.xml")

# label_set = {"Clean":"#00ff00ff", "Ink":"#ff0000ff"}
label_set = {"tils":"#ffff00ff","tumor":"#00ff00ff","calcification":"#000000ff","cell":"#00ffffff"}

annotations = SedeenAnnotationParser(renamed_label=label_set)
ext_annotation = annotations.parse(xml_path)

print(ext_annotation[0].geometry.area)
