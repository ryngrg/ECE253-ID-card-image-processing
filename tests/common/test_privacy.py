"""Real-card text must never reach a tracked table; scores and specimen / MIDV text must stay."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import privacy  # noqa: E402


def boxes():
    return pd.DataFrame(dict(image=["B04_S_G_0_rectified.png", "F01_S_G_0_rectified.png", "alb_id_60.png"],
                             box=[0, 0, 0], target=["realname", "sample", "midv"], hyp=["realnane", "sample", "midv"],
                             tlen=[8, 6, 4], ed=[1, 0, 0]))


def test_only_real_cards_are_private():
    assert [c for c in ("B01", "B03", "B04", "B08", "F05", "F06", "F14", "alb_id_60") if privacy.is_private(c)] == \
        ["B04", "B08", "F06", "F14"]


def test_redact_blanks_text_and_keeps_numbers():
    r = privacy.redact(boxes())
    assert list(r.target) == ["", "sample", "midv"] and list(r.hyp) == ["", "sample", "midv"]
    assert list(r.ed) == [1, 0, 0] and list(r.tlen) == [8, 6, 4]


def test_save_and_read_keep_the_full_table_private(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    privacy.save_csv(boxes(), "t_boxes.csv")
    public = pd.read_csv(tmp_path / "t_boxes.csv", keep_default_na=False)
    assert "realname" not in public.to_csv() and list(public.ed) == [1, 0, 0]
    assert privacy.read_csv("t_boxes.csv").target[0] == "realname"
    layout = pd.DataFrame(dict(card_id=["F06", "B02"], ref_text=["Real Name", "Sample"], target=["realname", "sample"]))
    privacy.save_csv(layout, "lay.csv")
    assert list(pd.read_csv("lay.csv", keep_default_na=False).ref_text) == ["", "Sample"]


def test_no_private_copy_without_real_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    privacy.save_csv(boxes().iloc[1:], "s_boxes.csv")
    assert not (tmp_path / "private").exists()
