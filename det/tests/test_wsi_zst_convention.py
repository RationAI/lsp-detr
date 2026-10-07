"""CPU tests for det/wsi_infer.py's .zst container (no model, no GPU, no slide).

Guards the two things a viewer sees but a plain encode/decode round trip cannot check:
  * the value actually STORED in the tile is (cx, height - cy) - a round trip through
    mapbox_vector_tile is self-consistent under either y convention, so the raw protobuf
    has to be inspected;
  * the output base name is "<slide file name incl. extension>_<tag>".
"""
import importlib.util
import os

import numpy as np
import pytest

zstd = pytest.importorskip("zstandard")
mvt = pytest.importorskip("mapbox_vector_tile")
from mapbox_vector_tile.Mapbox import vector_tile_pb2 as pb  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lsp_wsi_infer", os.path.join(_HERE, "..", "wsi_infer.py"))
wsi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsi)

H, W = 45440, 67456


class _Slide:
    name, height, width, mpp = "129S.tif", H, W, 0.4567


def _unzigzag(v):
    return (v >> 1) ^ (-(v & 1))


def _raw_points(path):
    """Raw stored (x, y) per feature; the MVT cursor resets at the start of every feature."""
    tile = pb.tile()
    tile.ParseFromString(zstd.ZstdDecompressor().decompress(open(path, "rb").read()))
    layer = tile.layers[0]
    pts = [(_unzigzag(f.geometry[1]), _unzigzag(f.geometry[2])) for f in layer.features]
    return layer.extent, pts


def _dets(centres, labels):
    det = np.empty(len(centres), dtype=wsi.DET_DTYPE)
    for i, (cx, cy) in enumerate(centres):
        det["x1"][i], det["x2"][i] = cx - 8, cx + 8
        det["y1"][i], det["y2"][i] = cy - 6, cy + 6
    det["score"] = 0.9
    det["label"] = labels
    return det


CENTRES = [(110.0, 60.0), (5010.0, 20010.0), (60010.0, 45010.0)]

TNT = wsi.ZST_CONVENTIONS["tnt"]                 # [(cat, term, nt, name)] indexed by model label
TNT_LYMPH = wsi.ZST_CONVENTIONS["tnt-lymph"]


def test_stored_y_is_height_minus_centre(tmp_path):
    """What the viewer reads must be (cx, H - cy) - the platform convention, cf.
    lsp-detr/wsi/results/SSMH_BRS_HE_051.i2syntax_LSPDETR_512.zst."""
    out = str(tmp_path / "t.zst")
    wsi.write_zst(_dets(CENTRES, [0, 1, 1]), _Slide(), out, TNT)
    extent, pts = _raw_points(out)
    assert extent == H
    for (cx, cy), (sx, sy) in zip(CENTRES, pts):
        assert abs(sx - cx) <= 1
        assert abs(sy - (H - cy)) <= 1          # NOT cy: that renders the slide upside down


def test_decodes_back_to_image_space(tmp_path):
    """mapbox_vector_tile.decode() undoes the flip, so a finished file reads back in image space."""
    out = str(tmp_path / "t.zst")
    wsi.write_zst(_dets(CENTRES, [0, 1, 1]), _Slide(), out, TNT)
    feats = mvt.decode(zstd.ZstdDecompressor().decompress(open(out, "rb").read()))["default"]["features"]
    for (cx, cy), f in zip(CENTRES, feats):
        gx, gy = f["geometry"]["coordinates"]
        assert abs(gx - cx) <= 1 and abs(gy - cy) <= 1


def test_encoder_y_semantics_are_what_write_zst_assumes():
    """Pins the library behaviour write_zst relies on (it silently changed nothing so far, but a
    version that flipped on decode instead would turn every output upside down)."""
    def stored(y_in, y_down):
        blob = mvt.encode({"name": "l", "features": [{"geometry": {"type": "Point", "coordinates": [100.0, y_in]},
                                                      "properties": {}, "id": 1}]},
                          default_options={"quantize_bounds": None, "y_coord_down": y_down, "extents": 1000})
        tile = pb.tile(); tile.ParseFromString(blob)
        return _unzigzag(tile.layers[0].features[0].geometry[2])

    assert stored(200.0, True) == 200            # y_coord_down=True stores the input as is
    assert stored(200.0, False) == 800           # y_coord_down=False stores extent - input


def _props(path):
    return [f["properties"] for f in
            mvt.decode(zstd.ZstdDecompressor().decompress(open(path, "rb").read()))["default"]["features"]]


def test_tumour_class_convention(tmp_path):
    """2-class arm: non-tumour 5 / tumour 1."""
    out = str(tmp_path / "t.zst")
    wsi.write_zst(_dets(CENTRES, [0, 1, 0]), _Slide(), out, TNT)
    props = _props(out)
    assert [p["categoryId"] for p in props] == ["5", "1", "5"]
    assert [p["nt"] for p in props] == [True, False, True]
    assert [p["termId"] for p in props] == [TNT[0][1], TNT[1][1], TNT[0][1]]


def test_three_class_convention(tmp_path):
    """3-class arm: the platform's tnt-lymph encoding, i.e. what the KongNet reference file
    SSMH_BRS_HE_106.i2syntax_KongNet.zst carries (non_tumor 2 / tumor 1 / lymphocyte 3, and a
    lymphocyte is still nt=True so the viewer's T/NT mode reads it as non-tumour)."""
    out = str(tmp_path / "t3.zst")
    wsi.write_zst(_dets(CENTRES, [0, 1, 2]), _Slide(), out, TNT_LYMPH)
    props = _props(out)
    assert [p["categoryId"] for p in props] == ["2", "1", "3"]
    assert [p["termId"] for p in props] == ["66d54d2a89181badfeac2d75", "66d54cd789181badfeac2d69",
                                            "66d54cee89181badfeac2d6d"]
    assert [p["nt"] for p in props] == [True, False, True]
    assert all(p["imageId"] == 0 and p["positivity_rank"] == 0 for p in props)


def test_resolve_convention():
    assert wsi.resolve_convention("auto", 2)[0] == "tnt"
    assert wsi.resolve_convention("auto", 3)[0] == "tnt-lymph"
    # a convention whose class count disagrees with the model must not silently mis-map labels
    with pytest.raises(SystemExit):
        wsi.resolve_convention("tnt", 3)
    with pytest.raises(SystemExit):
        wsi.resolve_convention("auto", 4)
    _, table = wsi.resolve_convention("tnt-lymph", 3, ["a", "b", "c"])
    assert [t[3] for t in table] == ["a", "b", "c"]          # names override, ids do not
    assert [t[0] for t in table] == ["2", "1", "3"]


def test_write_zst_rejects_out_of_range_labels(tmp_path):
    with pytest.raises(RuntimeError, match="outside the"):
        wsi.write_zst(_dets(CENTRES, [0, 1, 2]), _Slide(), str(tmp_path / "bad.zst"), TNT)


def test_verify_zst_catches_a_wrong_class(tmp_path):
    """A file whose properties do not match its labels must fail the self-check."""
    out = str(tmp_path / "t.zst")
    wsi.write_zst(_dets(CENTRES, [0, 1, 2]), _Slide(), out, TNT_LYMPH)
    cx = np.array([c[0] for c in CENTRES]); cy = np.array([c[1] for c in CENTRES])
    with pytest.raises(RuntimeError, match="wrong"):
        wsi.verify_zst(out, cx, cy, np.array([2, 1, 0]), TNT_LYMPH)     # labels permuted


def test_verify_zst_catches_a_flip(tmp_path):
    """The self-check in write_zst must actually fail on a flipped file."""
    out = str(tmp_path / "flipped.zst")
    cx = np.array([c[0] for c in CENTRES])
    cy = np.array([c[1] for c in CENTRES])
    blob = mvt.encode({"name": "default",
                       "features": [{"geometry": {"type": "Point", "coordinates": [x, H - y]},
                                     "properties": {}, "id": i} for i, (x, y) in enumerate(CENTRES)]},
                      default_options={"quantize_bounds": None, "y_coord_down": False, "extents": H})
    open(out, "wb").write(zstd.ZstdCompressor().compress(blob))     # the old, wrong call
    with pytest.raises(RuntimeError, match="flipped"):
        wsi.verify_zst(out, cx, cy, np.zeros(len(CENTRES), dtype=np.int64), TNT)


def test_output_base_name_keeps_the_slide_extension():
    """<slide file name incl. extension>_<tag>, and the tag holds the only underscore."""
    name = f"{_Slide.name}_{'LSPDETR-TNT'}"
    assert name == "129S.tif_LSPDETR-TNT"
    assert name.rsplit("_", 1)[0] == _Slide.name
