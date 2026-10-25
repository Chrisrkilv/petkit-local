"""A Pura Max (T4) visit over HTTP is handled exactly like a Pura X (T3) one.

No T4 visit has been captured, so the reports use the documented litter-family
code 10 shape, the one a real T3 sends: `time_in`, `time_out`, `pet_weight` and
the rest inside `content`, with a code 9 weight check sharing its event_id.
"""
import json
import urllib.parse
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from petkit_local.ai.pets import PetRegistry
from petkit_local.devices.base import Device
from petkit_local.devices.registry import DeviceRegistry
from petkit_local.events.store import EventStore
from petkit_local.ha.categories import get_entities_for_device
from petkit_local.http.server import create_app

ON = {"pet_weight_match": True, "pet_weight_max_diff_g": 800}
CONFIG = {"api_url": "http://server/6/", "mqtt_port": 1883, "proxy_mode": False,
          "proxy_upstream": "", "capture": False}

SUMMARY = {"time_in": 1790000001, "time_out": 1790000105, "auto_clear": 1,
           "is_shit": 1, "interval": 30, "pet_weight": 4950, "shit_weight": 0}


def _report(code: str, content: dict) -> str:
    return (f"eventType={code}&eventId=500_1790000000&content="
            + urllib.parse.quote(json.dumps(content)))


class _Publisher:
    def __init__(self):
        self.events = []

    async def publish_event(self, device, entity_suffix, event_type, attrs=None):
        self.events.append((entity_suffix, event_type, attrs))

    async def publish_state(self, *a, **kw):
        pass

    async def publish_availability(self, *a, **kw):
        pass

    async def publish_pet_discovery(self, pet):
        pass

    async def publish_pet_state(self, pet, store):
        pass


async def _visit(tmp_path: Path, device_type: str):
    """Post one visit (weight check, then summary) from a box of this type.

    Returns what HA was fired, the stored rows, and the two pets' ids.
    """
    reg = DeviceRegistry()
    reg.get_or_create(petkit_id=500, device_type=device_type, serial_number="SN500")
    store = EventStore(tmp_path / f"{device_type}.db")
    pets = PetRegistry(store, str(tmp_path / f"faces-{device_type}"))
    heavy = await pets.create("Biscuit", device_ids=[500], weight=4900)
    light = await pets.create("Pepper", device_ids=[500], weight=4500)
    pub = _Publisher()

    app = create_app(reg, {**CONFIG, **ON})
    app["event_store"] = store
    app["pet_registry"] = pets
    app["ha_publisher"] = pub
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for code, content in (("9", {"pet_weight": 4012}), ("10", SUMMARY)):
            r = await client.post(f"/6/{device_type}/dev_event_report",
                                  headers={"X-Device": "id=500&sn=SN500"},
                                  data=_report(code, content))
            assert r.status == 200
        rows = await store.all_events()
        stats = await store.pet_visit_stats(heavy["id"], now=1790000200.0)
    finally:
        await client.close()
        await store.close()
    return pub.events, rows, stats, heavy["id"], light["id"]


async def test_a_t4_visit_summary_fires_pet_out_on_the_toilet_event(tmp_path):
    events, *_ = await _visit(tmp_path, "t4")
    assert ("toilet_event", "pet_out", SUMMARY) in events


def test_the_t4_toilet_event_declares_pet_out():
    t4 = Device(device_type="t4", petkit_id=500)
    [toilet] = [e for e in get_entities_for_device(t4) if e.key == "toilet_event"]
    assert "pet_out" in toilet.options


async def test_a_t4_visit_is_named_by_weight_like_a_t3(tmp_path):
    t4_events, t4_rows, t4_stats, heavy, _ = await _visit(tmp_path, "t4")
    t3_events, t3_rows, t3_stats, _, _ = await _visit(tmp_path, "t3")

    [summary] = [r for r in t4_rows if r["event_type"] == "10"]
    assert summary["pet_ref"] is None
    assert summary["pet_id"] == heavy
    assert summary["pet_match"] == "weight: 4950 g, nearest Biscuit (4900 g)"
    assert t4_stats["last_visit_weight"] == 4950

    # Field for field what the same visit on a T3 gives.
    def view(rows):
        return [(r["event_type"], r["event_kind"], r["pet_id"], r["pet_match"])
                for r in rows]
    assert view(t4_rows) == view(t3_rows)
    assert t4_events == t3_events
    # `last_visit_ts` is the arrival time, which is the wall clock here.
    t4_stats.pop("last_visit_ts")
    t3_stats.pop("last_visit_ts")
    assert t4_stats == t3_stats


@pytest.mark.parametrize("device_type", ["t3", "t4"])
def test_neither_esp32_box_is_left_to_face_recognition(device_type):
    assert Device(device_type=device_type, petkit_id=500).supports_ai is False
