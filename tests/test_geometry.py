"""
tests/test_geometry.py
unit tests for bounding box geometry helper functions.
run with:  pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from tracking.tracker import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)



def test_center_basic():
    bbox = [100, 200, 300, 400]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 200
    assert cy == 300

def test_center_square():
    bbox = [0, 0, 100, 100]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 50
    assert cy == 50

def test_center_non_square():
    bbox = [10, 20, 110, 60]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 60
    assert cy == 40

def test_center_returns_integers():
    bbox = [0, 0, 101, 101]
    cx, cy = get_center_of_bbox(bbox)
    assert isinstance(cx, int)
    assert isinstance(cy, int)

def test_center_float_bbox():
    bbox = [10.5, 20.5, 50.5, 80.5]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 30
    assert cy == 50



def test_width_basic():
    bbox = [100, 0, 300, 0]
    assert get_bbox_width(bbox) == 200

def test_width_zero():
    bbox = [100, 0, 100, 0]
    assert get_bbox_width(bbox) == 0

def test_width_float():
    bbox = [10.5, 0, 50.5, 0]
    assert get_bbox_width(bbox) == pytest.approx(40.0)




def test_foot_is_bottom_center():
    bbox = [100, 200, 300, 400]
    fx, fy = get_foot_position(bbox)
    assert fx == 200   # horizontal centre
    assert fy == 400   # bottom edge

def test_foot_y_equals_y2():
    bbox = [0, 50, 200, 350]
    _, fy = get_foot_position(bbox)
    assert fy == 350

def test_foot_returns_integers():
    bbox = [0.0, 0.0, 101.0, 201.0]
    fx, fy = get_foot_position(bbox)
    assert isinstance(fx, int)
    assert isinstance(fy, int)



def test_distance_zero():
    assert measure_distance((0, 0), (0, 0)) == pytest.approx(0.0)

def test_distance_horizontal():
    assert measure_distance((0, 0), (3, 0)) == pytest.approx(3.0)

def test_distance_vertical():
    assert measure_distance((0, 0), (0, 4)) == pytest.approx(4.0)

def test_distance_diagonal():
    # 3-4-5 right triangle
    assert measure_distance((0, 0), (3, 4)) == pytest.approx(5.0)

def test_distance_symmetric():
    p1, p2 = (10, 20), (30, 50)
    assert measure_distance(p1, p2) == pytest.approx(measure_distance(p2, p1))


def test_xy_distance_basic():
    dx, dy = measure_xy_distance((10, 20), (3, 8))
    assert dx == 7
    assert dy == 12

def test_xy_distance_negative():
    dx, dy = measure_xy_distance((0, 0), (5, 5))
    assert dx == -5
    assert dy == -5