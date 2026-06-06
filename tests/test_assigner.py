"""
tests/test_assigner.py
unit tests for the Assigner (team assignment) module.
run with:  pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from assigner.assigner import Assigner



def make_frame_with_colors(colors: list, bbox_size: int = 60) -> np.ndarray:
    """
    create a synthetic BGR frame containing colored rectangles.
    each color in the list gets its own player-sized patch.
    """
    n = len(colors)
    frame = np.zeros((200, n * 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        x1 = i * 100 + 20
        y1 = 20
        x2 = x1 + bbox_size
        y2 = y1 + bbox_size
        frame[y1:y2, x1:x2] = color
    return frame


def make_bboxes(n: int, bbox_size: int = 60) -> list:
    """return n bounding boxes matching make_frame_with_colors layout."""
    bboxes = []
    for i in range(n):
        x1 = i * 100 + 20
        y1 = 20
        x2 = x1 + bbox_size
        y2 = y1 + bbox_size
        bboxes.append([x1, y1, x2, y2])
    return bboxes


def test_assigner_init():
    a = Assigner()
    assert not a.fitted
    assert len(a.team1_ids) == 0
    assert len(a.team2_ids) == 0
    assert len(a.memory) == 0



def test_get_team_unknown_returns_0():
    a = Assigner()
    assert a.get_team(999) == 0


def test_get_team_never_returns_none():
    a = Assigner()
    for pid in range(100):
        result = a.get_team(pid)
        assert result is not None
        assert result in (0, 1, 2)



def test_init_teams_splits_two_colors():
    """two clearly distinct jersey colors should end up in different teams."""
    # team A: bright red jerseys
    # team B: bright blue jerseys
    red  = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, red, red, blue, blue, blue]
    frame  = make_frame_with_colors(colors)
    bboxes = make_bboxes(len(colors))

    players = {i: {"bounding_box": bboxes[i]} for i in range(len(colors))}

    a = Assigner()
    a.init_teams(frame, players)

    assert a.fitted
    assert len(a.team1_ids) > 0
    assert len(a.team2_ids) > 0
    assert len(a.team1_ids) + len(a.team2_ids) == len(colors)


def test_init_teams_too_few_players():
    """with only 1 player, init_teams should not crash and fitted stays False."""
    frame   = make_frame_with_colors([(0, 0, 200)])
    players = {0: {"bounding_box": make_bboxes(1)[0]}}

    a = Assigner()
    a.init_teams(frame, players)

    assert not a.fitted   # can't cluster with 1 player


def test_init_teams_uses_real_track_ids():
    """init_teams must use the actual dict keys as IDs, not sequential ints."""
    red  = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, blue]
    frame  = make_frame_with_colors(colors)
    bboxes = make_bboxes(2)

    # use non-sequential track IDs like ByteTrack would give
    players = {
        42: {"bounding_box": bboxes[0]},
        87: {"bounding_box": bboxes[1]},
    }

    a = Assigner()
    a.init_teams(frame, players)

    if a.fitted:
        all_ids = a.team1_ids | a.team2_ids
        assert 42 in all_ids or 87 in all_ids
        # Sequential IDs 0,1 should NOT be in the sets
        assert 0 not in all_ids
        assert 1 not in all_ids




def test_get_or_create_returns_track_id():
    red   = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox  = make_bboxes(1)[0]

    a   = Assigner()
    pid = a.get_or_create_id(frame, bbox, track_id=5)
    assert pid == 5


def test_get_or_create_empty_crop_returns_none():
    """a bounding box with zero area should return None."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox  = [50, 50, 50, 50]   # zero size

    a      = Assigner()
    result = a.get_or_create_id(frame, bbox, track_id=1)
    assert result is None


def test_get_or_create_stores_in_memory():
    red   = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox  = make_bboxes(1)[0]

    a = Assigner()
    a.get_or_create_id(frame, bbox, track_id=7)
    assert 7 in a.memory


def test_get_or_create_same_id_twice():
    """calling twice with the same track_id should not raise."""
    red   = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox  = make_bboxes(1)[0]

    a = Assigner()
    pid1 = a.get_or_create_id(frame, bbox, track_id=3)
    pid2 = a.get_or_create_id(frame, bbox, track_id=3)
    assert pid1 == pid2 == 3




def test_team_assignment_consistent():
    """once assigned, a player's team should not change on re-query."""
    red   = (0, 0, 200)
    frame = make_frame_with_colors([red])
    bbox  = make_bboxes(1)[0]

    a = Assigner()
    a.get_or_create_id(frame, bbox, track_id=10)
    team1 = a.get_team(10)
    team2 = a.get_team(10)
    assert team1 == team2


def test_no_player_in_both_teams():
    """no track_id should appear in both team1_ids and team2_ids."""
    red  = (0, 0, 200)
    blue = (200, 0, 0)
    colors = [red, red, blue, blue]
    frame  = make_frame_with_colors(colors)
    bboxes = make_bboxes(4)

    players = {i: {"bounding_box": bboxes[i]} for i in range(4)}

    a = Assigner()
    a.init_teams(frame, players)

    overlap = a.team1_ids & a.team2_ids
    assert len(overlap) == 0



def test_read_video_returns_list():
    """read_video should return a list (even if empty for bad path)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils import read_video

    # non-existent file should return empty list, not crash
    frames = read_video("non_existent_file.mp4")
    assert isinstance(frames, list)


def test_save_video_creates_file(tmp_path):
    """save_video should create a file at the given path."""
    from utils import save_video

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    out_path = str(tmp_path / "test_output.avi")
    save_video(frames, out_path)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0