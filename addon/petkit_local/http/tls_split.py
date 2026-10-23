"""The device TLS port, shared by MQTT and the HTTPS device API.

Ingenic models only ever send MQTT on it. Some ESP32 litter boxes send their API
calls there too, as HTTPS: a Pura X (T3, firmware 1.491) makes every call that
way, and a Pura Max (T4, firmware 1.652) tries 443 first and falls back to plain
HTTP on 80 after 10s. Neither validates the certificate. With amqtt listening
there, the request is read as a malformed CONNECT and dropped with "No more
data", so the T3 never reaches the API at all and the T4 pays 10s per bootstrap.

So the port is terminated here instead, with the broker's certificate and its
device-friendly context (`mqtt/broker.py::device_ssl_context`), and the first
decrypted bytes decide where the connection goes. An HTTP method token goes to
the device-facing app; anything else goes to the broker. MQTT's CONNECT starts
with 0x10, which no method does, so a device that only ever sent MQTT here still
reaches the broker as before, credentials and all.

Both sides are handed the TLS transport itself rather than a pipe to their own
port, so what they read off the connection stays true. `request.remote` is the
device, and it becomes `state["ip"]`, which go2rtc and the patchers dial; through
a pipe it would be 127.0.0.1. `proxy.py::_local_socket` is the port the device
dialled, which `http/dns.py::loops_back` compares upstream against. The broker
logs the device's own address on its `tls` listener, as it did when it owned the
port. The broker side is amqtt's `external` listener type, which exists for
exactly this; the HTTP side is a second protocol factory for the same app.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from amqtt.adapters import StreamReaderAdapter, StreamWriterAdapter

from petkit_local.mqtt.broker import device_ssl_context

if TYPE_CHECKING:
    from aiohttp import web
    from amqtt.broker import Broker

log = logging.getLogger(__name__)

#: The broker listener MQTT from this port is handed to. Same name the TLS
#: listener had when amqtt bound the port itself, so its log lines read the same.
LISTENER = "tls"

#: What an HTTP request can start with. The trailing space matters: without it a
#: binary MQTT payload that happened to start with "PUT" would be taken for one.
HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"HEAD ", b"DELETE ", b"OPTIONS ", b"PATCH ")

#: How long a connection may stay silent after the handshake. HTTP and MQTT both
#: have the client speak first, so one that says nothing is neither, and is
#: closed rather than held open indefinitely.
SNIFF_TIMEOUT = 5.0

#: How long `stop` lets an HTTP request still being answered finish.
SHUTDOWN_TIMEOUT = 5.0


def classify(head: bytes) -> str | None:
    """`"http"`, `"mqtt"`, or None while `head` could still become a method.

    None is not a verdict: TLS can deliver `PO` in one record and `ST ` in the
    next, so the caller reads on until this decides.
    """
    if head.startswith(HTTP_METHODS):
        return "http"
    if any(method.startswith(head) for method in HTTP_METHODS):
        return None
    return "mqtt"


def _peer(transport: asyncio.BaseTransport) -> str:
    """`address:port` of the far end, for log lines only."""
    peer = transport.get_extra_info("peername")
    return f"{peer[0]}:{peer[1]}" if peer and len(peer) >= 2 else "?"


class _Sniffer(asyncio.Protocol):
    """Holds a fresh TLS connection until its first bytes say what it is."""

    def __init__(self, split: TlsSplit) -> None:
        self._split = split
        self._transport: asyncio.Transport | None = None
        self._head = b""
        self._timer: asyncio.TimerHandle | None = None
        self._timed_out = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]
        self._split._sniffing.add(transport)
        self._timer = asyncio.get_running_loop().call_later(SNIFF_TIMEOUT, self._give_up)

    def data_received(self, data: bytes) -> None:
        self._head += data
        kind = classify(self._head)
        if kind is None:
            return
        self._settle()
        self._split._hand_over(kind, self._transport, self._head)

    def connection_lost(self, exc: Exception | None) -> None:
        # Only reached for a connection that was never handed over: once it is,
        # the transport reports to its new protocol instead.
        self._settle()
        if not self._timed_out:
            log.debug("TLS connection from %s closed before saying what it was",
                      _peer(self._transport))

    def _give_up(self) -> None:
        log.debug("TLS connection from %s sent nothing in %.0fs, closing",
                  _peer(self._transport), SNIFF_TIMEOUT)
        self._timed_out = True
        self._transport.close()

    def _settle(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._split._sniffing.discard(self._transport)


class TlsSplit:
    """The TLS port's listener, and every connection it has handed on."""

    def __init__(self, broker: Broker, app: web.Application) -> None:
        self._broker = broker
        self._app = app
        # Built on first use, not here: this runs during the app's startup, and
        # `_make_handler` freezes the app, after which `start_background` could
        # no longer store what it starts.
        self._api: web.Server | None = None
        self._server: asyncio.Server | None = None
        self._sniffing: set[asyncio.BaseTransport] = set()
        self._sessions: set[asyncio.Task[Any]] = set()

    async def start(self, port: int, certfile: str, keyfile: str) -> None:
        """Bind the TLS port.

        Raises:
            OSError: the port could not be bound.
            Exception: the certificate could not be loaded (amqtt's
                `BrokerError`, or `ssl.SSLError`).
        """
        ctx = device_ssl_context({"certfile": certfile, "keyfile": keyfile})
        self._server = await asyncio.get_running_loop().create_server(
            lambda: _Sniffer(self), "0.0.0.0", port, ssl=ctx, reuse_address=True)

    def _hand_over(self, kind: str, transport: asyncio.Transport, head: bytes) -> None:
        """Give `transport` to the API or the broker, `head` first."""
        loop = asyncio.get_running_loop()
        if kind == "http":
            log.debug("TLS connection from %s is HTTP, to the device API", _peer(transport))
            if self._api is None:
                # aiohttp's runner builds its own server the same way; this is a
                # second one for the same app, so every middleware and route
                # applies exactly as on the plain port.
                self._api = self._app._make_handler(loop=loop)
            protocol: asyncio.Protocol = self._api()
            transport.set_protocol(protocol)
            protocol.connection_made(transport)
            protocol.data_received(head)
            return

        log.debug("TLS connection from %s is MQTT, to the broker", _peer(transport))
        reader = asyncio.StreamReader(loop=loop)
        protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
        transport.set_protocol(protocol)
        protocol.connection_made(transport)
        protocol.data_received(head)
        writer = asyncio.StreamWriter(transport, protocol, reader, loop)
        task = loop.create_task(self._mqtt(reader, writer))
        self._sessions.add(task)
        task.add_done_callback(self._sessions.discard)

    async def _mqtt(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Run one MQTT session through the broker, and close it after."""
        try:
            await self._broker.external_connected(
                StreamReaderAdapter(reader), StreamWriterAdapter(writer), LISTENER)
        except Exception:
            log.exception("MQTT session on the TLS port failed")
        finally:
            # amqtt closes a session it ends itself, but not every one it gives
            # up on before CONNECT; closing twice is harmless.
            writer.close()

    async def stop(self) -> None:
        """Stop listening and close every connection still open."""
        if self._server is not None:
            self._server.close()
        for transport in list(self._sniffing):
            transport.close()
        self._sniffing.clear()
        for task in list(self._sessions):
            task.cancel()
        await asyncio.gather(*self._sessions, return_exceptions=True)
        if self._api is not None:
            await self._api.shutdown(SHUTDOWN_TIMEOUT)
        if self._server is not None:
            await self._server.wait_closed()


async def start_tls_split(broker: Broker, app: web.Application, port: int,
                          certfile: str, keyfile: str) -> TlsSplit | None:
    """Serve the TLS port, or say why not and return None.

    Never raises. A port that cannot be bound or a cert that cannot be loaded
    costs the devices that use this port, while the plain-HTTP API on 80 keeps
    working, so it must not stop the add-on.
    """
    split = TlsSplit(broker, app)
    try:
        await split.start(port, certfile, keyfile)
    except Exception as e:
        log.error("Could not serve the TLS port %d (%s); devices that use it will not "
                  "connect", port, e)
        return None
    return split
