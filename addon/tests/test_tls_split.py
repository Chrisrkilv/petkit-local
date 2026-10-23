"""The device TLS port: HTTPS API calls and MQTT on one port (`http/tls_split.py`).

Real TLS over loopback against a real broker and a real aiohttp app, because the
whole point is what each side reads off a connection it did not accept itself.
"""
import asyncio
import json
import socket
import ssl

import pytest
from aiohttp import web

from petkit_local.http import tls_split
from petkit_local.http.tls_split import classify, start_tls_split
from petkit_local.mqtt.broker import ensure_self_signed, start_broker

#: A minimal MQTT 3.1.1 CONNECT: clean session, keep-alive 60, client id "x".
CONNECT = bytes([0x10, 13, 0, 4]) + b"MQTT" + bytes([4, 0x02, 0, 60, 0, 1]) + b"x"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client_ctx() -> ssl.SSLContext:
    # The boxes do not validate the certificate either.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _whoami(request: web.Request) -> web.Response:
    sockname = request.transport.get_extra_info("sockname")
    return web.json_response({"path": request.path, "remote": request.remote,
                              "local_port": sockname[1], "body": (await request.text())})


@pytest.fixture
async def split(tmp_path):
    cert, key = str(tmp_path / "broker.crt"), str(tmp_path / "broker.key")
    assert ensure_self_signed(cert, key)
    broker = await start_broker(_free_port(), tls=True, tls_port=0,
                                certfile=cert, keyfile=key, tls_split=True)
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", _whoami)
    runner = web.AppRunner(app)
    await runner.setup()
    s = await start_tls_split(broker, app, 0, cert, key)
    assert s is not None
    s.port = s._server.sockets[0].getsockname()[1]
    yield s
    await s.stop()
    await runner.cleanup()
    await broker.shutdown()


async def _open(port: int):
    return await asyncio.open_connection("127.0.0.1", port, ssl=_client_ctx())


async def _settled(collection) -> bool:
    """Whether `collection` empties within a second; the client sees EOF first."""
    for _ in range(50):
        if not collection:
            return True
        await asyncio.sleep(0.02)
    return False


def test_classify():
    assert classify(b"POST /6/t3/dev_state_report HTTP/1.1\r\n") == "http"
    assert classify(b"GET / HTTP/1.1\r\n") == "http"
    assert classify(CONNECT) == "mqtt"
    # Not yet a verdict: still a prefix of a method.
    assert classify(b"") is None
    assert classify(b"PO") is None
    assert classify(b"OPTIONS") is None
    # A method name without its space is not a request line.
    assert classify(b"PUTX") == "mqtt"


async def test_an_https_api_call_reaches_the_app_as_the_device(split):
    reader, writer = await _open(split.port)
    body = b"id=1&state=1"
    writer.write(b"POST /6/t3/dev_state_report HTTP/1.1\r\nHost: api.petkt.com\r\n"
                 b"Content-Type: application/x-www-form-urlencoded\r\n"
                 b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
    await writer.drain()
    reply = await asyncio.wait_for(reader.read(), 5)
    writer.close()

    head, _, payload = reply.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200")
    seen = json.loads(payload)
    assert seen["path"] == "/6/t3/dev_state_report"
    assert seen["body"] == body.decode()
    # The connection the app saw is the device's own, not a loopback pipe's.
    assert seen["remote"] == "127.0.0.1"
    assert seen["local_port"] == split.port


async def test_a_method_split_across_records_is_still_http(split):
    reader, writer = await _open(split.port)
    writer.write(b"PO")
    await writer.drain()
    await asyncio.sleep(0.05)
    writer.write(b"ST /6/poll/t3/heartbeat HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Length: 0\r\nConnection: close\r\n\r\n")
    await writer.drain()
    reply = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    assert reply.startswith(b"HTTP/1.1 200")
    assert b"/6/poll/t3/heartbeat" in reply


async def test_an_mqtt_connect_reaches_the_broker(split):
    reader, writer = await _open(split.port)
    writer.write(CONNECT)
    await writer.drain()
    connack = await asyncio.wait_for(reader.readexactly(4), 5)
    # CONNACK, remaining length 2, return code 0 (no registry here, so the auth
    # plugin lets everything on).
    assert connack == bytes([0x20, 0x02, 0x00, 0x00])
    assert len(split._sessions) == 1
    writer.write(bytes([0xE0, 0x00]))  # DISCONNECT
    await writer.drain()
    assert await asyncio.wait_for(reader.read(), 5) == b""
    writer.close()
    assert await _settled(split._sessions)


async def test_a_silent_connection_is_closed(split, monkeypatch):
    monkeypatch.setattr(tls_split, "SNIFF_TIMEOUT", 0.2)
    reader, writer = await _open(split.port)
    assert await asyncio.wait_for(reader.read(), 5) == b""
    writer.close()
    assert await _settled(split._sniffing)


async def test_a_connection_closed_before_saying_anything_is_forgotten(split):
    reader, writer = await _open(split.port)
    writer.close()
    await writer.wait_closed()
    assert await _settled(split._sniffing)


async def test_stop_closes_connections_still_open(split):
    _, idle_writer = await _open(split.port)
    mqtt_reader, mqtt_writer = await _open(split.port)
    mqtt_writer.write(CONNECT)
    await mqtt_writer.drain()
    await asyncio.wait_for(mqtt_reader.readexactly(4), 5)

    await split.stop()

    assert not split._sniffing
    assert not split._sessions
    assert await asyncio.wait_for(mqtt_reader.read(), 5) == b""
    idle_writer.close()
    mqtt_writer.close()


async def test_a_port_that_cannot_be_bound_is_reported_not_raised(tmp_path, caplog):
    cert, key = str(tmp_path / "broker.crt"), str(tmp_path / "broker.key")
    assert ensure_self_signed(cert, key)
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", 0))
        taken.listen()
        port = taken.getsockname()[1]
        assert await start_tls_split(None, web.Application(), port, cert, key) is None
    assert "Could not serve the TLS port" in caplog.text


async def test_the_broker_binds_nothing_on_the_tls_port_when_split(tmp_path):
    cert, key = str(tmp_path / "broker.crt"), str(tmp_path / "broker.key")
    tls_port = _free_port()
    broker = await start_broker(_free_port(), tls=True, tls_port=tls_port,
                                certfile=cert, keyfile=key, tls_split=True)
    try:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", tls_port))  # would raise if amqtt held it
    finally:
        await broker.shutdown()
