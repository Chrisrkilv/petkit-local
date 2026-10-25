"""A T4 keeps PetKit's real Aliyun `productKey` (issue #44).

Handed the key we mint, a T4 on firmware 1.652 loops through dev_ota_check,
dev_signup and dev_iot_device_info every ~80s and never reports state. With the
real key it runs, deviceName and deviceSecret still ours. So for the models in
`KEEP_UPSTREAM_PRODUCT_KEY_TYPES` a proxied reply keeps the real key, the key is
adopted onto the device, and our own answers hand it out from then on.
"""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from amqtt.contexts import BaseContext

from petkit_local.devices import payloads
from petkit_local.devices.base import Device
from petkit_local.devices.registry import DeviceRegistry
from petkit_local.http.proxy import close_proxy_session
from petkit_local.http.redact import RedactionPolicy, redact_body
from petkit_local.http.server import create_app
from petkit_local.mqtt.auth import AliyunAuthPlugin, compute_aliyun_sign
from petkit_local.web.hub import EventHub

MQTT_HOST = "192.0.2.199"
REAL_PK = "a9XyZ1Ab2Cd"  # 11 characters, the shape of a real Aliyun key
REAL_INSTANCE = "iot-0a1b2c3d"


def _upstream(**extra) -> dict:
    """PetKit's flat dev_iot_device_info reply, values made up."""
    result = {
        "id": 200, "deviceName": "realdn", "deviceSecret": "realsecret",
        "iotPlatform": "ALI", "iotInstanceId": REAL_INSTANCE, "productKey": REAL_PK,
        "mqttHost": "iot-0a1b2c3d.mqtt.iothub.aliyuncs.com", "regionId": "eu-central-1",
    }
    result.update(extra)
    return {"result": result}


def _redact(device: Device, payload: dict, **policy) -> dict:
    result = redact_body(json.dumps(payload).encode(),
                         endpoint=f"/6/{device.device_type}/dev_iot_device_info",
                         policy=RedactionPolicy(device=device, mqtt_host=MQTT_HOST,
                                                **policy))
    return json.loads(result.body)


# --- redaction --------------------------------------------------------------

@pytest.mark.parametrize("wrap", [
    lambda r: r,
    lambda r: {"result": {"ali": r["result"]}},
])
def test_a_t4_keeps_the_real_product_key_and_nothing_else(wrap):
    device = Device(device_type="t4", petkit_id=200, serial_number="SN200")
    body = _redact(device, wrap(_upstream()))
    block = body["result"].get("ali", body["result"])

    assert block["productKey"] == REAL_PK
    # What #44 proved: the instance id then equals that same real key.
    assert block["iotInstanceId"] == REAL_PK
    assert block["mqttHost"] == MQTT_HOST
    assert block["deviceName"] == device.mqtt_device_name
    assert block["deviceSecret"] == device.mqtt_device_secret


@pytest.mark.parametrize("device_type", ["t3", "t5"])
def test_other_models_still_get_ours(device_type):
    device = Device(device_type=device_type, petkit_id=200, serial_number="SN200")
    block = _redact(device, _upstream())["result"]
    assert block["productKey"] == device.mqtt_product_key
    assert block["iotInstanceId"] == device.mqtt_product_key


def test_option_off_gives_a_t4_ours_too():
    device = Device(device_type="t4", petkit_id=200, serial_number="SN200")
    block = _redact(device, _upstream(), keep_upstream_product_key=False)["result"]
    assert block["productKey"] == device.mqtt_product_key
    assert block["iotInstanceId"] == device.mqtt_product_key


def test_the_capture_is_unchanged():
    device = Device(device_type="t4", petkit_id=200, serial_number="SN200")
    result = redact_body(json.dumps(_upstream()).encode(),
                         endpoint="/6/t4/dev_iot_device_info",
                         policy=RedactionPolicy(device=device, mqtt_host=MQTT_HOST))
    assert result.captured["mqtt"]["product_key"] == REAL_PK
    assert result.captured["mqtt"]["iot_instance_id"] == REAL_INSTANCE
    assert result.captured["mqtt"]["device_secret"] == "realsecret"


# --- adoption, through the real middleware ----------------------------------

T4_HDR = {"X-Device": "id=200&sn=SN200&type=T4"}
T3_HDR = {"X-Device": "id=300&sn=SN300&type=T3"}


def _config(**kw) -> dict:
    base = {
        "api_url": "http://192.0.2.199:8080/6/",
        "mqtt_port": 1883,
        "bucket_endpoint": "https://192.0.2.199:9000",
        "data_dir": "/tmp",
        "capture": False,
        "capture_dir": "/tmp/capture",
        "proxy_mode": False,
        "proxy_upstream": "",
    }
    base.update(kw)
    return base


class _Registry(DeviceRegistry):
    def __init__(self):
        super().__init__()
        self.dirty = 0

    def mark_dirty(self):
        self.dirty += 1
        super().mark_dirty()


async def _clients(**config):
    async def cloud(request):
        return web.json_response(_upstream())

    cloud_app = web.Application()
    cloud_app.router.add_route("*", "/{path:.*}", cloud)
    up = TestClient(TestServer(cloud_app))
    await up.start_server()

    registry = _Registry()
    app = create_app(registry, _config(proxy_upstream=str(up.make_url("")).rstrip("/"),
                                       **config))
    app["event_hub"] = EventHub()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, up, registry


async def _close(client, up):
    await close_proxy_session(client.app)
    await client.close()
    await up.close()


async def test_a_t4_adopts_the_real_key_and_serves_it_locally(caplog):
    client, up, registry = await _clients()
    try:
        r = await client.post("/6/t4/dev_signup", headers=T4_HDR)
        assert r.status == 200
        device = registry.get(200)
        minted = device.mqtt_product_key
        assert minted != REAL_PK
        client.app["config"]["proxy_mode"] = True
        registry.dirty = 0

        with caplog.at_level("INFO"):
            r = await client.post("/6/t4/dev_iot_device_info", headers=T4_HDR)
        # Proxied, the device gets the cloud's own (flat) shape, redacted.
        block = (await r.json())["result"]
        assert block["productKey"] == REAL_PK
        assert block["iotInstanceId"] == REAL_PK
        assert block["mqttHost"] == MQTT_HOST
        assert block["deviceName"] == device.mqtt_device_name
        assert block["deviceSecret"] == device.mqtt_device_secret

        assert device.mqtt_product_key == REAL_PK
        assert registry.dirty >= 1
        adopted = [m for m in caplog.messages if "product key" in m]
        assert adopted == ["Adopted the real PetKit product key for device 200"]
        assert not any(device.mqtt_device_secret in m or "realsecret" in m
                       for m in caplog.messages)

        # A second proxied reply changes nothing and logs nothing more.
        caplog.clear()
        with caplog.at_level("INFO"):
            await client.post("/6/t4/dev_iot_device_info", headers=T4_HDR)
        assert not [m for m in caplog.messages if "product key" in m]

        # Proxy off: our own answer now carries the adopted key.
        client.app["config"]["proxy_mode"] = False
        r = await client.post("/6/t4/dev_iot_device_info", headers=T4_HDR)
        block = (await r.json())["result"]["ali"]
        assert block["productKey"] == REAL_PK
        assert block["iotInstanceId"] == REAL_PK
    finally:
        await _close(client, up)


async def test_a_t3_is_not_adopted():
    client, up, registry = await _clients()
    try:
        await client.post("/6/t3/dev_signup", headers=T3_HDR)
        device = registry.get(300)
        minted = device.mqtt_product_key
        client.app["config"]["proxy_mode"] = True

        r = await client.post("/6/t3/dev_iot_device_info", headers=T3_HDR)
        assert (await r.json())["result"]["productKey"] == minted
        assert device.mqtt_product_key == minted
    finally:
        await _close(client, up)


async def test_option_off_restores_todays_behaviour_for_a_t4():
    client, up, registry = await _clients(keep_upstream_product_key=False)
    try:
        await client.post("/6/t4/dev_signup", headers=T4_HDR)
        device = registry.get(200)
        minted = device.mqtt_product_key
        client.app["config"]["proxy_mode"] = True

        r = await client.post("/6/t4/dev_iot_device_info", headers=T4_HDR)
        block = (await r.json())["result"]
        assert block["productKey"] == minted
        assert block["iotInstanceId"] == minted
        assert device.mqtt_product_key == minted
    finally:
        await _close(client, up)


# --- what the adopted key has to keep working -------------------------------

def test_the_adopted_key_survives_a_restart():
    device = Device(device_type="t4", petkit_id=200, serial_number="SN200")
    device.mqtt_product_key = REAL_PK
    restored = Device.from_dict(json.loads(json.dumps(device.to_dict())))
    assert restored.mqtt_product_key == REAL_PK
    assert restored.mqtt_device_name == device.mqtt_device_name
    assert restored.mqtt_device_secret == device.mqtt_device_secret


@pytest.mark.parametrize("build", [payloads.to_iot_device_info_flat,
                                   lambda d, h: {"result": payloads.to_iot_device_info(d, h)
                                                 ["result"]["ali"]}])
def test_both_local_payloads_hand_out_the_adopted_key(build):
    device = Device(device_type="t4", petkit_id=200, serial_number="SN200")
    device.mqtt_product_key = REAL_PK
    block = build(device, MQTT_HOST)["result"]
    assert block["productKey"] == REAL_PK
    assert block["iotInstanceId"] == REAL_PK
    assert block["mqttHost"] == MQTT_HOST


async def test_mqtt_auth_finds_the_device_by_the_adopted_key():
    registry = DeviceRegistry()
    device = registry.get_or_create(200, "t4", serial_number="SN200")
    minted = device.mqtt_product_key
    device.mqtt_product_key = REAL_PK
    dn = device.mqtt_device_name

    assert registry.by_mqtt_name(REAL_PK, dn) is device
    assert registry.by_mqtt_name(minted, dn) is None

    plugin = AliyunAuthPlugin(BaseContext())
    plugin.set_registry(registry)
    plugin.set_strict(True)

    class _Session:
        pass

    s = _Session()
    s.client_id = f"{REAL_PK}.{dn}|timestamp=1,signmethod=hmacsha256,securemode=2|"
    s.username = f"{dn}&{REAL_PK}"
    s.password = compute_aliyun_sign(device.mqtt_device_secret, f"{REAL_PK}.{dn}",
                                     dn, REAL_PK, "1")
    assert await plugin.authenticate(session=s) is True
    assert device.mqtt_connected is True
