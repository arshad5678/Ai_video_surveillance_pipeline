"""Unit tests for ZoneManager and load_zones_config — pure shapely/yaml, no I/O
beyond temp files, no GPU. Track objects are plain dataclass instances built by
make_track(), the simplest reliable "mock" for an immutable domain model."""

from pathlib import Path

import pytest

from src.surveillance.models.domain.bounding_box import BoundingBox
from src.surveillance.models.domain.track import Track
from src.surveillance.pipelines.zones import (
    InvalidPolygonError,
    ZoneConfigurationError,
    ZoneManager,
    load_zones_config,
)

SQUARE_ZONE_YAML = """
zones:
  - id: zone_a
    name: Zone A
    type: intrusion
    enabled: true
    polygon:
      - [0, 0]
      - [100, 0]
      - [100, 100]
      - [0, 100]
"""

TWO_ZONE_YAML = """
zones:
  - id: zone_a
    name: Zone A
    type: intrusion
    enabled: true
    polygon:
      - [0, 0]
      - [100, 0]
      - [100, 100]
      - [0, 100]
  - id: zone_b
    name: Zone B
    type: monitoring
    enabled: true
    polygon:
      - [200, 200]
      - [300, 200]
      - [300, 300]
      - [200, 300]
"""

DISABLED_ZONE_YAML = """
zones:
  - id: zone_a
    name: Zone A
    type: intrusion
    enabled: false
    polygon:
      - [0, 0]
      - [100, 0]
      - [100, 100]
      - [0, 100]
"""


def write_zones(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "zones.yaml"
    path.write_text(content)
    return path


def make_track(
    track_id: int = 1,
    center_x: float = 50.0,
    center_y: float = 50.0,
    frame_index: int = 3,
    timestamp: float = 42.0,
    source_id: str = "cam-1",
) -> Track:
    half_w, half_h = 10.0, 20.0
    return Track(
        track_id=track_id,
        bounding_box=BoundingBox(
            x1=center_x - half_w, y1=center_y - half_h, x2=center_x + half_w, y2=center_y + half_h
        ),
        confidence=0.9,
        class_name="person",
        class_id=0,
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
        is_confirmed=True,
        age=5,
        hits=5,
        time_since_update=0,
        history=((center_x, center_y),),
    )


# --- config loading -------------------------------------------------------


def test_loading_valid_zones_config(tmp_path: Path) -> None:
    zones = load_zones_config(write_zones(tmp_path, SQUARE_ZONE_YAML))

    assert len(zones) == 1
    zone = zones[0]
    assert zone.zone_id == "zone_a"
    assert zone.zone_name == "Zone A"
    assert zone.zone_type == "intrusion"
    assert zone.enabled is True
    assert len(zone.polygon) == 4


def test_missing_zones_file_raises_zone_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ZoneConfigurationError):
        load_zones_config(tmp_path / "does_not_exist.yaml")


def test_polygon_too_few_points_raises_invalid_polygon_error(tmp_path: Path) -> None:
    content = """
zones:
  - id: bad_zone
    polygon:
      - [0, 0]
      - [10, 10]
"""
    with pytest.raises(InvalidPolygonError):
        load_zones_config(write_zones(tmp_path, content))


def test_polygon_malformed_point_raises_invalid_polygon_error(tmp_path: Path) -> None:
    content = """
zones:
  - id: bad_zone
    polygon:
      - [0, 0]
      - [10]
      - [10, 10]
"""
    with pytest.raises(InvalidPolygonError):
        load_zones_config(write_zones(tmp_path, content))


def test_self_intersecting_polygon_raises_invalid_polygon_error(tmp_path: Path) -> None:
    # Bowtie: edges cross, making this an invalid (self-intersecting) ring.
    content = """
zones:
  - id: bowtie
    polygon:
      - [0, 0]
      - [100, 100]
      - [100, 0]
      - [0, 100]
"""
    with pytest.raises(InvalidPolygonError):
        load_zones_config(write_zones(tmp_path, content))


def test_missing_required_field_raises_zone_configuration_error(tmp_path: Path) -> None:
    content = """
zones:
  - name: Missing ID And Polygon
"""
    with pytest.raises(ZoneConfigurationError):
        load_zones_config(write_zones(tmp_path, content))


def test_duplicate_zone_id_raises_zone_configuration_error(tmp_path: Path) -> None:
    content = """
zones:
  - id: dup
    polygon: [[0,0],[10,0],[10,10],[0,10]]
  - id: dup
    polygon: [[20,20],[30,20],[30,30],[20,30]]
"""
    with pytest.raises(ZoneConfigurationError):
        load_zones_config(write_zones(tmp_path, content))


# --- ZoneManager evaluation ------------------------------------------------


def test_point_inside_polygon(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, SQUARE_ZONE_YAML))

    memberships = manager.evaluate([make_track(center_x=50.0, center_y=50.0)])

    assert len(memberships) == 1
    assert memberships[0].inside is True
    assert memberships[0].zone_id == "zone_a"
    assert memberships[0].track_id == 1


def test_point_outside_polygon(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, SQUARE_ZONE_YAML))

    memberships = manager.evaluate([make_track(center_x=500.0, center_y=500.0)])

    assert len(memberships) == 1
    assert memberships[0].inside is False


def test_multiple_zones_produce_one_membership_each(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, TWO_ZONE_YAML))

    memberships = manager.evaluate([make_track(center_x=50.0, center_y=50.0)])  # inside zone_a only

    assert len(memberships) == 2
    by_zone = {m.zone_id: m.inside for m in memberships}
    assert by_zone == {"zone_a": True, "zone_b": False}


def test_multiple_tracks_each_evaluated_against_each_zone(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, TWO_ZONE_YAML))

    tracks = [
        make_track(track_id=1, center_x=50.0, center_y=50.0),  # inside zone_a
        make_track(track_id=2, center_x=250.0, center_y=250.0),  # inside zone_b
    ]
    memberships = manager.evaluate(tracks)

    assert len(memberships) == 4  # 2 tracks x 2 zones
    assert {(m.track_id, m.zone_id): m.inside for m in memberships} == {
        (1, "zone_a"): True,
        (1, "zone_b"): False,
        (2, "zone_a"): False,
        (2, "zone_b"): True,
    }


def test_disabled_zone_produces_no_membership(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, DISABLED_ZONE_YAML))

    memberships = manager.evaluate([make_track(center_x=50.0, center_y=50.0)])

    assert memberships == []


def test_empty_tracks_returns_empty_list(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, SQUARE_ZONE_YAML))

    assert manager.evaluate([]) == []


def test_membership_preserves_track_frame_context(tmp_path: Path) -> None:
    manager = ZoneManager(write_zones(tmp_path, SQUARE_ZONE_YAML))

    track = make_track(center_x=50.0, center_y=50.0, frame_index=17, timestamp=123.5, source_id="cam-9")
    membership = manager.evaluate([track])[0]

    assert membership.frame_index == 17
    assert membership.timestamp == 123.5
    assert membership.source_id == "cam-9"


def test_reload_picks_up_changed_zones_file(tmp_path: Path) -> None:
    zones_path = write_zones(tmp_path, SQUARE_ZONE_YAML)
    manager = ZoneManager(zones_path)

    memberships_before = manager.evaluate([make_track(center_x=250.0, center_y=250.0)])
    assert memberships_before[0].inside is False  # zone_a only, point is outside it

    write_zones(tmp_path, TWO_ZONE_YAML)  # now adds zone_b, which contains (250, 250)
    manager.reload()

    memberships_after = manager.evaluate([make_track(center_x=250.0, center_y=250.0)])
    by_zone = {m.zone_id: m.inside for m in memberships_after}
    assert by_zone == {"zone_a": False, "zone_b": True}
