"""Attributing a visit to a pet by weight, on a box with no face recognition
(`ai/pets.py::match_by_weight` and `PetRegistry.attribute`).

No ESP32 visit has been captured yet, so the reports below use the documented
code-10 shape: `pet_weight` in whole grams inside `content`, the same field a
captured T5 visit carries (`events/codes.py` row "10", which covers every
litter box including t3/t4).
"""
import json
import urllib.parse
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from petkit_local.ai.pets import PetRegistry, match_by_weight
from petkit_local.devices.base import Device
from petkit_local.devices.registry import DeviceRegistry
from petkit_local.events.normalize import from_event_report, parse_event_report_form
from petkit_local.events.store import EventStore
from petkit_local.http.server import create_app

ON = {"pet_weight_match": True, "pet_weight_max_diff_g": 800}

HEAVY = {"id": 1, "name": "Biscuit", "weight": 4900}
LIGHT = {"id": 2, "name": "Pepper", "weight": 4500}


def _visit(weight=None, event_type="10", **extra) -> str:
    content = dict(extra)
    if weight is not None:
        content["pet_weight"] = weight
    return (f"eventType={event_type}&eventId=v{weight}&content="
            + urllib.parse.quote(json.dumps(content)))


def _row(device: Device, body: str) -> dict:
    row = from_event_report(device, parse_event_report_form(body))
    row.pop("_state", None)
    row.pop("_content", None)
    return row


def test_nearest_pet_wins():
    assert match_by_weight(4950, [HEAVY, LIGHT], 800)[0] == 1
    assert match_by_weight(4520, [HEAVY, LIGHT], 800)[0] == 2
    pet_id, reason = match_by_weight(4950, [HEAVY, LIGHT], 800)
    assert reason == "weight: 4950 g, nearest Biscuit (4900 g)"


def test_the_unsure_band_either_side_of_the_midpoint():
    # 400 g apart: the band is max(100, 25% of 400) = 100 g each side of 4700.
    for weight in (4610, 4650, 4700, 4750, 4790):
        pet_id, reason = match_by_weight(weight, [HEAVY, LIGHT], 800)
        assert pet_id is None, weight
        assert reason.startswith("unsure: ")
    assert match_by_weight(4590, [HEAVY, LIGHT], 800)[0] == 2
    assert match_by_weight(4810, [HEAVY, LIGHT], 800)[0] == 1


def test_the_band_grows_with_the_gap():
    # 2000 g apart: 25% is 500 g, so 4500..5500 either side of 5000.
    big = {"id": 3, "name": "Tank", "weight": 6000}
    small = {"id": 4, "name": "Pip", "weight": 4000}
    assert match_by_weight(5400, [big, small], 800)[0] is None
    assert match_by_weight(5600, [big, small], 800)[0] == 3


def test_two_pets_of_almost_the_same_weight_are_never_told_apart():
    twin = {"id": 5, "name": "Twin", "weight": 4520}
    # Lighter than both: no midpoint between them, still not a decision.
    assert match_by_weight(4300, [twin, LIGHT], 800)[0] is None


def test_too_far_from_every_pet_is_no_match():
    pet_id, reason = match_by_weight(6000, [HEAVY, LIGHT], 800)
    assert pet_id is None
    assert reason == "no match: 6000 g, nearest Biscuit (4900 g) is 1100 g away"
    assert match_by_weight(5700, [HEAVY, LIGHT], 800)[0] == 1
    assert match_by_weight(5701, [HEAVY, LIGHT], 800)[0] is None


def test_with_one_pet_only_the_distance_applies():
    assert match_by_weight(4700, [HEAVY], 800)[0] == 1
    assert match_by_weight(4100, [HEAVY], 800)[0] == 1
    assert match_by_weight(4099, [HEAVY], 800)[0] is None


def test_pets_without_a_weight_are_skipped_and_kg_is_read_as_kg():
    unweighed = {"id": 6, "name": "Ghost", "weight": None}
    in_kg = {"id": 7, "name": "Mochi", "weight": 4.9}
    assert match_by_weight(4950, [unweighed, in_kg, LIGHT], 800)[0] == 7
    pet_id, reason = match_by_weight(4950, [unweighed], 800)
    assert pet_id is None and "no pet on this device has a weight" in reason


async def _pets(pet_registry: PetRegistry, device_id: int = 100) -> tuple[int, int]:
    heavy = await pet_registry.create("Biscuit", device_ids=[device_id], weight=4900)
    light = await pet_registry.create("Pepper", device_ids=[device_id], weight=4500)
    return heavy["id"], light["id"]


async def test_attribute_matches_a_t3_visit_and_records_why(pet_registry: PetRegistry):
    heavy, light = await _pets(pet_registry)
    t3 = Device(device_type="t3", petkit_id=100)

    row = _row(t3, _visit(4950))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] == heavy
    assert row["pet_match"] == "weight: 4950 g, nearest Biscuit (4900 g)"

    row = _row(t3, _visit(4700))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] is None
    assert row["pet_match"].startswith("unsure: 4700 g")

    row = _row(t3, _visit(4520))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] == light

    row = _row(t3, _visit(6000))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] is None
    assert row["pet_match"].startswith("no match: 6000 g")


async def test_only_pets_linked_to_the_box_are_considered(pet_registry: PetRegistry):
    await pet_registry.create("Elsewhere", device_ids=[999], weight=4950)
    heavy, _ = await _pets(pet_registry)
    t3 = Device(device_type="t3", petkit_id=100)
    row = _row(t3, _visit(4950))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] == heavy


async def test_a_visit_without_a_weight_is_left_alone(pet_registry: PetRegistry):
    await _pets(pet_registry)
    t3 = Device(device_type="t3", petkit_id=100)
    for body in (_visit(None), _visit(0)):
        row = _row(t3, body)
        await pet_registry.attribute(row, t3, ON)
        assert row["pet_id"] is None
        assert "pet_match" not in row


async def test_a_reported_pet_id_wins_over_the_weight(pet_registry: PetRegistry):
    heavy, light = await _pets(pet_registry)
    t3 = Device(device_type="t3", petkit_id=100)
    row = _row(t3, _visit(4950, score_info=[{"id": light, "score": 900}]))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] == light
    assert "pet_match" not in row

    # An id we cannot resolve is still the device's claim: no weight guess.
    row = _row(t3, _visit(4950, score_info=[{"id": 101392625, "score": 900}]))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] is None
    assert "pet_match" not in row


async def test_option_off_changes_nothing(pet_registry: PetRegistry):
    await _pets(pet_registry)
    t3 = Device(device_type="t3", petkit_id=100)
    for config in ({}, {"pet_weight_match": False}):
        row = _row(t3, _visit(4950))
        await pet_registry.attribute(row, t3, config)
        assert row["pet_id"] is None
        assert "pet_match" not in row


async def test_a_box_with_face_recognition_is_left_to_it(pet_registry: PetRegistry):
    await _pets(pet_registry)
    t5 = Device(device_type="t5", petkit_id=100)
    row = _row(t5, _visit(4950))
    await pet_registry.attribute(row, t5, ON)
    assert row["pet_id"] is None
    assert "pet_match" not in row


async def test_only_the_visit_summary_is_matched(pet_registry: PetRegistry):
    await _pets(pet_registry)
    t3 = Device(device_type="t3", petkit_id=100)
    row = _row(t3, _visit(4950, event_type="6"))
    await pet_registry.attribute(row, t3, ON)
    assert row["pet_id"] is None
    assert "pet_match" not in row


async def test_dev_event_report_stores_the_match_and_feeds_the_pet_stats(tmp_path: Path):
    reg = DeviceRegistry()
    reg.get_or_create(petkit_id=100, device_type="t3", serial_number="SN100")
    store = EventStore(tmp_path / "petkit.db")
    pets = PetRegistry(store, str(tmp_path / "faces"))
    app = create_app(reg, {"api_url": "http://server/6/", "proxy_mode": False,
                           "capture": False, **ON})
    app["event_store"] = store
    app["pet_registry"] = pets
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        heavy, _ = await _pets(pets)
        r = await client.post("/6/t3/dev_event_report", data=_visit(4950),
                              headers={"X-Device": "id=100&sn=SN100"})
        assert r.status == 200
        rows = await store.all_events()
        assert rows[-1]["pet_id"] == heavy
        assert rows[-1]["pet_match"] == "weight: 4950 g, nearest Biscuit (4900 g)"
        stats = await store.pet_visit_stats(heavy)
        assert stats["last_visit_weight"] == 4950
        assert stats["visits_today"] == 1
    finally:
        await client.close()
        await store.close()
