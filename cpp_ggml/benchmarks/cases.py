# Benchmark cases (VLX-Seek style): official demo photos, all released
# task families. The first six keep the original sibling harness set; the
# rest extend coverage to ocr / point / extra detection-counting combos on
# the remaining official example photos (demo/examples/images).
# (image, task, query, lang) — query is a list for the box-set tasks.
CASES = [
    ("livingroom.jpg", "detection", ["sofa", "pillow", "lamp"], "en"),
    ("seabreeze.jpg", "matting", "person", "en"),
    ("dogs.jpg", "referring", "the dog closest to the camera", "en"),
    ("boys.jpg", "count_box", "people", "en"),
    ("livingroom.jpg", "referring", "the lamp on the table", "en"),
    ("seabreeze.jpg", "detection", ["person", "surfboard"], "en"),
    ("coffee_sign.jpg", "ocr", "text", "en"),
    ("parasol.jpg", "point", "umbrella", "en"),
    ("chess.jpg", "referring", "the white king", "en"),
    ("parasol.jpg", "detection", ["umbrella"], "en"),
    ("dogs.jpg", "count_box", "dogs", "en"),
    ("chess.jpg", "count_box", "chess pieces", "en"),
]
