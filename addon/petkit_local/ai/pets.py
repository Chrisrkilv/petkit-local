"""PetRegistry — pet CRUD backed by EventStore's `pets`/`pet_faces` tables,
face-photo hosting for on-device facial recognition, and the dev_discern_pic
payload builder.

The panel calls these photos **mugshots**; this package and the schema call
them faces (`pet_faces`, `add_face`, `/faces/{filename}`). Same thing — the
short user-facing word and the identifier that predates it.

The device's NPU runs recognition locally; we host the reference photos
(`dev_discern_pic`) and read the result back out of an event's
`content.score_info[].id`. That id is the one we handed out — the firmware
copies it verbatim from the outer list entry (`get_update_face_score_info`) —
so no id-mapping protocol exists for a device we configured. `resolve_pet_ref`
covers the one case where it does not hold: a box still matching against faces
cached from PetKit's cloud reports that cloud's id until it re-syncs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from petkit_local.events import codes
from petkit_local.events.normalize import _as_dict
from petkit_local.media.transcode import normalize_face_photo
from petkit_local.utils.coerce import to_float, to_int
from petkit_local.utils.paths import UnsafePathError, safe_join, sanitize_filename

if TYPE_CHECKING:
    from petkit_local.devices.base import Device
    from petkit_local.events.store import EventStore

log = logging.getLogger(__name__)

#: The smallest gap, in grams, either side of the halfway point between the two
#: nearest pets inside which a visit is left unattributed (`match_by_weight`).
#: A cat's weight moves by more than the scale's resolution from one visit to
#: the next, so two cats a few grams apart can never be told apart by weight.
WEIGHT_UNSURE_MIN_G = 100

#: The same band as a share of the gap between the two nearest pets, used once
#: it is wider than `WEIGHT_UNSURE_MIN_G`: the further apart two cats are, the
#: more their own visit-to-visit drift there is room for.
WEIGHT_UNSURE_SHARE = 0.25


def _grams(weight: object) -> float | None:
    """A pet's stored weight in grams, or None if it has none.

    The panel stores whatever it is given. Grams is what the scale reports and
    what this compares against, but a weight typed in kilograms (4.9) would
    otherwise be a 5 g cat, so anything under 100 is read as kg.
    """
    value = to_float(weight, None)
    if value is None or value <= 0:
        return None
    return value * 1000 if value < 100 else value


def match_by_weight(weight_g: float, pets: list[dict],
                    max_diff_g: float) -> tuple[int | None, str]:
    """Pick the pet a visit of `weight_g` belongs to, by weight alone.

    For a box with no face recognition, weight is the only thing a visit says
    about who made it. The nearest pet wins, unless:

    * it is more than `max_diff_g` away, which is a visitor, a cat that has
      changed a lot, or a weight that is not a cat at all; or
    * the second-nearest is nearly as close. "Nearly" is the difference
      between the two distances being under twice the unsure band, the band
      being `WEIGHT_UNSURE_SHARE` of the gap between those two pets but never
      less than `WEIGHT_UNSURE_MIN_G`. Between two pets that is the band
      either side of their midpoint; it also covers a visit lighter or heavier
      than both of two pets that weigh almost the same.

    A wrong name is worse than none: the per-pet entities would show one cat's
    visit as the other's, with nothing to say it was a guess.

    Args:
        pets: Rows as `PetRegistry.get` returns them. Those without a weight
            are skipped.

    Returns:
        `(pet_id, reason)`; `pet_id` is None when no pet was picked, and
        `reason` says why either way, in words.
    """
    weighed = []
    for pet in pets:
        grams = _grams(pet.get("weight"))
        if grams is None:
            continue
        if grams != to_float(pet.get("weight"), None):
            log.info("Pet %s weight %s read as kg (%.0f g)", pet.get("name"),
                     pet.get("weight"), grams)
        weighed.append((abs(weight_g - grams), grams, pet))
    if not weighed:
        return None, f"no match: {weight_g:.0f} g, no pet on this device has a weight"
    weighed.sort(key=lambda w: w[0])

    distance, grams, nearest = weighed[0]
    if distance > max_diff_g:
        return None, (f"no match: {weight_g:.0f} g, nearest {nearest['name']} "
                      f"({grams:.0f} g) is {distance:.0f} g away")
    if len(weighed) > 1:
        distance2, grams2, second = weighed[1]
        band = max(WEIGHT_UNSURE_MIN_G, WEIGHT_UNSURE_SHARE * abs(grams - grams2))
        if distance2 - distance < 2 * band:
            return None, (f"unsure: {weight_g:.0f} g is between {nearest['name']} "
                          f"({grams:.0f} g) and {second['name']} ({grams2:.0f} g)")
    return nearest["id"], f"weight: {weight_g:.0f} g, nearest {nearest['name']} ({grams:.0f} g)"

# The generated name is substituted into the URL the device is handed
# (`http://{host}/faces/{name}` in discern_pic_payload), which is built WITHOUT
# percent-encoding. sanitize_filename's blacklist is right for media paths but
# lets spaces, quotes and `!` through, and those would break that URL — so a
# whitelist runs first and sanitize_filename then does what only it does:
# strip control characters, collapse `..` runs and cap the length.
_NON_URL_SAFE = re.compile(r'[^A-Za-z0-9_.-]')

# Capped per component, not on the composed name: `pet_{id}_face_` + 64 + "."
# + 8 stays well inside utils.paths.MAX_FILENAME_LENGTH, so a long face_id can
# never squeeze out the extension.
_MAX_FACE_ID_LENGTH = 64
_MAX_EXT_LENGTH = 8


def _safe_photo_filename(pet_id: int, face_id: str | int, ext: str = "jpg") -> str:
    """Build the stored/served filename for a pet's reference face photo.

    `face_id` is our own `pet_faces.id` today, but it ends up both as a path
    under the faces directory and as a URL path segment, so it is still reduced
    to a single component that can neither traverse (no separators, no `..`)
    nor create a hidden file (no leading dot) — the sanitising is what makes it
    safe to widen the source of that value later.

    It also ends up inside a SHELL COMMAND on the device. Observed on a T5
    (fw 943): `get_face_picture_by_url` fetches each photo with

        sh -c "wget -T 10 -O /tmp/pet_face_pic.jpg -c <our url>"

    so the whitelist below — alphanumerics, `_`, `.`, `-` only — is what stands
    between this filename and command injection on the device, not merely a
    tidy URL. Do not relax it.
    """
    safe_face_id = sanitize_filename(
        _NON_URL_SAFE.sub("-", str(face_id)),
        fallback="0",
        max_length=_MAX_FACE_ID_LENGTH,
    )
    safe_ext = sanitize_filename(
        _NON_URL_SAFE.sub("", str(ext).lstrip(".")),
        fallback="jpg",
        max_length=_MAX_EXT_LENGTH,
    )
    return f"pet_{pet_id}_face_{safe_face_id}.{safe_ext}"


# A face photo is hundreds of KB, so this is real blocking I/O on a loop that is
# also answering devices — the caller hands it to `asyncio.to_thread`.
def _write_photo(path: str, data: bytes) -> None:
    """Write a whole face photo (blocking — call via a thread)."""
    with open(path, "wb") as f:
        f.write(data)


#: How many reference photos one imported pet may bring across. The same cap
#: `add_face` enforces, applied before the download rather than after, so a
#: reply claiming a hundred faces costs a hundred fetches to discover that.
MAX_CLOUD_FACES = 10


def cloud_pets(payload: Any, device_id: int) -> list[dict[str, Any]]:
    """The pets a proxied `dev_discern_pic` reply describes.

    That reply is the account's recognition set for one device: which animals
    it should match against, and where their reference photos are. It already
    passes through us untouched — the cloud's pets reaching the device is what
    proxy mode MEANS — and we read it for the same reason the device does.

    Args:
        payload: The decoded upstream JSON body.
        device_id: The device whose request this answered.

    Returns:
        `[{"pet_ref", "link_with", "faces": [{"id", "url"}]}]`. `pet_ref` is
        PetKit's OWN pet id, deliberately not called `pet_id`: it is a foreign
        identity until somebody binds it (`resolve_pet_ref`), and a box still
        matching against cloud-cached faces is already reporting it.

        There is no name here. The device-facing payload carries ids and URLs
        and nothing else — a pet's name lives in the account API, which no
        device ever asks for — so an imported pet arrives unnamed rather than
        with one invented for it.
    """
    if not isinstance(payload, dict):
        return []
    entries = (payload.get("result") or {}).get("list") \
        if isinstance(payload.get("result"), dict) else None
    if not isinstance(entries, list):
        return []

    found = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pet_ref = to_int(entry.get("id"), 0) or 0
        if pet_ref <= 0:
            continue
        faces = []
        for face in (entry.get("discern") or [])[:MAX_CLOUD_FACES]:
            if not isinstance(face, dict):
                continue
            url = str(face.get("url", ""))
            # http/https only. The bytes end up on disk as a JPEG and are never
            # echoed back, so the exposure is narrow — but `file:` and friends
            # have no business being followed from a network-supplied field.
            if urlparse(url).scheme not in ("http", "https"):
                continue
            faces.append({"id": to_int(face.get("id"), 0) or 0, "url": url})
        if faces:
            found.append({"pet_ref": pet_ref, "link_with": device_id, "faces": faces})
    return found


class PetRegistry:
    """The pets known to this add-on, and their reference face photos.

    A thin, typed facade over `EventStore`'s `pets`/`pet_faces` tables plus the
    one piece of state SQL cannot hold: the JPEGs on disk under `faces_dir`.
    The two are kept consistent here — `add_face` writes the file and the row
    together, `delete_face` and `delete` remove both.

    Invariant a reader must know: a pet row's `id` is the SAME id the device
    reports back in `content.score_info[].id`, because it is the id we hand it
    in `discern_pic_payload`. `resolve_pet_ref` is the only mapping, and it
    exists purely for identities we did NOT hand out.
    """

    def __init__(self, store: EventStore, faces_dir: str) -> None:
        """Bind to the store and make `faces_dir` exist.

        Created eagerly because `add_face` writes straight into it —
        it resolves a filename with `safe_join` and opens it, with no
        `makedirs` of its own, so on a fresh install the very first upload
        would otherwise fail.
        """
        self._store = store
        self._faces_dir = faces_dir
        os.makedirs(faces_dir, exist_ok=True)

    # --- CRUD ---------------------------------------------------------

    async def create(self, name: str, device_ids: list[int] | None = None,
                     weight: float | None = None) -> dict:
        """Insert a pet and return the stored row (including its new `id`)."""
        pid = await self._store.upsert_pet({
            "name": name, "device_ids_json": device_ids or [], "weight": weight,
        })
        return await self._store.get_pet(pid)

    async def update(self, pet_id: int, **fields: Any) -> dict | None:
        """Patch the named columns, returning the updated row or None if
        `pet_id` doesn't exist. Unknown keys are dropped by the store."""
        if await self._store.get_pet(pet_id) is None:
            return None
        if fields:
            await self._store.upsert_pet(fields, pet_id=pet_id)
        return await self._store.get_pet(pet_id)

    async def delete(self, pet_id: int) -> bool:
        """Delete a pet and every face photo it owns. False if it didn't exist.

        The files are unlinked first and best-effort: one that is already gone
        (or on a read-only mount) must not leave the rows behind, since the
        rows are what the panel and `discern_pic_payload` actually read.
        """
        pet = await self._store.get_pet(pet_id)
        if pet is None:
            return False
        for face in await self._store.pet_faces(pet_id):
            if face.get("photo_path"):
                try:
                    os.remove(face["photo_path"])
                except OSError:
                    pass
        await self._store.delete_pet(pet_id)
        return True

    async def get(self, pet_id: int) -> dict | None:
        """One pet row, or None if `pet_id` is unknown.

        Row shape (the `pets` table verbatim, see events/models.py::Pet)::

            {id, name, device_ids_json, face_id, photo_path, weight, created_at}

        `device_ids_json` comes back as the raw JSON **string** the column
        stores, not the list that `create`/`update` accept — the store only
        encodes on the way in.
        """
        return await self._store.get_pet(pet_id)

    async def all(self) -> list[dict]:
        """Every pet, ordered by `id`. Rows have the shape `get` documents."""
        return await self._store.all_pets()

    async def for_device(self, device_id: int) -> list[dict]:
        """The pets assigned to one device (its `device_ids_json` membership)."""
        return await self._store.pets_for_device(device_id)

    # --- face photos ----------------------------------------------------

    async def faces(self, pet_id: int) -> list[dict]:
        """One pet's reference photos: `[{id, pet_id, photo_path, created_at}]`.

        `id` is the value the device is handed as `discern[].id` and echoes
        back in the state report's `discernPic` array.
        """
        return await self._store.pet_faces(pet_id)

    async def face(self, face_id: int) -> dict | None:
        """One face row by id, or None."""
        return await self._store.get_pet_face(face_id)

    async def add_face(self, pet_id: int, data: bytes) -> dict | None:
        """Validate via JPEG magic bytes, store under `{faces_dir}/`, register.

        Returns the new face row, or None if `pet_id` doesn't exist, `data`
        isn't a real JPEG (`dev_discern_pic` only ever serves JPEGs), the pet
        already holds `MAX_FACES_PER_PET`, or the path check refuses.

        The image is normalised to the size the NPU is fed — every reference
        photo PetKit's cloud served was a 224x224 face crop, never a plain
        snapshot — so an ordinary phone photo becomes the same kind of input
        the recogniser was tuned against. One already at that size is stored
        byte-for-byte.

        The id is allocated by the store BEFORE the file is written, because
        it is part of the filename — see `_safe_photo_filename`.
        """
        if await self._store.get_pet(pet_id) is None:
            return None
        if data[:3] != b"\xff\xd8\xff":
            return None
        data = await normalize_face_photo(data)

        face_id = await self._store.add_pet_face(pet_id, "")
        if face_id is None:
            return None

        filename = _safe_photo_filename(pet_id, face_id)
        # _safe_photo_filename already yields a single component; re-checking
        # the resolved target against the root is what actually proves the
        # write lands inside the faces directory, symlinks included.
        try:
            path = safe_join(self._faces_dir, filename)
        except UnsafePathError:
            log.warning("Refusing to write a face photo outside %s (face_id=%r)",
                        self._faces_dir, face_id)
            await self._store.delete_pet_face(face_id)
            return None
        try:
            await asyncio.to_thread(_write_photo, path, data)
        except OSError:
            # Never leave a row pointing at a file that isn't there: the device
            # would be told to fetch a URL that 404s and would retry forever.
            # Swallowed rather than raised because this method documents "or
            # None", and a full disk should reach the panel as an error message
            # rather than a 500 with a traceback.
            log.warning("Could not write a face photo for pet %s", pet_id, exc_info=True)
            await self._store.delete_pet_face(face_id)
            return None
        await self._store.update_pet_face_path(face_id, path)
        return await self._store.get_pet_face(face_id)

    async def delete_face(self, face_id: int) -> bool:
        """Drop one reference photo and its file. False if it didn't exist.

        The device keeps the extracted features in `/system/feature.bin` until
        its next `dev_discern_pic` poll, so recognition does not stop the
        instant this returns.
        """
        face = await self._store.get_pet_face(face_id)
        if face is None:
            return False
        if face.get("photo_path"):
            try:
                os.remove(face["photo_path"])
            except OSError:
                pass
        await self._store.delete_pet_face(face_id)
        return True

    # --- device-facing payloads ------------------------------------------

    async def discern_pic_payload(self, device_id: int, api_host: str) -> dict:
        """`dev_discern_pic` response: the face photos this device should
        recognize against, one entry per pet assigned to it.

        Shape taken from the real PetKit cloud (captured 2026-07-27), NOT from
        the firmware error strings this once guessed at::

            {"result": {"list": [{"id": <pet>, "discern": [{"id": <face>, "url": …}]}]}}

        Both ids are INTEGERS — the firmware reads the outer one with `%d` and
        the inner with `%lld`. The cloud sends no `area` key inside a list
        entry even though the parser has a "not find area" log string for one;
        we follow the cloud, since a configuration that demonstrably works
        outranks a log line. If a device ever ignores this payload, that
        omission is the first thing to try adding back.
        """
        entries = []
        for pet in await self.for_device(device_id):
            discern = [
                {"id": f["id"], "url": f"http://{api_host}/faces/{os.path.basename(f['photo_path'])}"}
                for f in await self.faces(pet["id"]) if f.get("photo_path")
            ]
            if discern:
                entries.append({"id": pet["id"], "discern": discern})
        return {"result": {"list": entries}}

    # --- ingestion hooks --------------------------------------------------

    async def resolve_pet_ref(self, pet_ref: int | None) -> int | None:
        """Map an identity the device reported to one of our pet rows.

        Normally the identity IS our `pets.id`, because that is what
        `discern_pic_payload` handed the device. The alias path exists for the
        other case: a box still carrying faces cached from PetKit's cloud keeps
        reporting that cloud's id (`101392625` on this household's device), and
        the user binds it explicitly via `device_pet_ids_json`.

        Returns None for an unknown id, which is what keeps `events.pet_id`
        honest — writing an unresolvable id there would fabricate an HA pet
        device for a row that does not exist.
        """
        if pet_ref is None:
            return None
        for pet in await self.all():
            if pet["id"] == pet_ref:
                return pet["id"]
            try:
                aliases = json.loads(pet.get("device_pet_ids_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                aliases = []
            if pet_ref in aliases:
                return pet["id"]
        return None

    async def attribute(self, row: dict, device: Device, config: dict) -> None:
        """Set `row["pet_id"]` for an event about to be stored.

        The id the device reported decides, through `resolve_pet_ref`. Only
        when it reported none, has no face recognition of its own, and the
        `pet_weight_match` option is on, is a visit summary matched to one of
        the device's pets by weight (`match_by_weight`). `row["pet_match"]`
        then records what was decided and why, including when nothing was.

        Args:
            config: The live config dict (`pet_weight_match`,
                `pet_weight_max_diff_g`).
        """
        row["pet_id"] = await self.resolve_pet_ref(row.get("pet_ref"))
        if row.get("pet_ref") is not None or not config.get("pet_weight_match"):
            return
        if device.supports_ai:
            return
        code = codes.lookup(str(row.get("event_type") or ""), device.device_type)
        if code is None or code.role != codes.ROLE_VISIT_SUMMARY:
            return
        content = _as_dict(row.get("content_json"))
        weight = to_float(content.get("pet_weight", content.get("petWeight")), None)
        if not weight or weight <= 0:
            return

        max_diff = to_float(config.get("pet_weight_max_diff_g"), None)
        if max_diff is None:
            max_diff = 800
        pet_id, reason = match_by_weight(weight, await self.for_device(device.petkit_id),
                                         max_diff)
        row["pet_id"] = pet_id
        row["pet_match"] = reason
        log.info("Visit on %s %d: %s", device.device_type, device.petkit_id, reason)

    async def pet_for_related_event(self, related_event: str) -> dict | None:
        """Best-effort pet lookup for a visit session — used by
        media/pipeline.py to append `· {Pet}` to friendly filenames. Reads
        the most recent event sharing this related_event and its attributed
        pet_id, if any (see events/ingest.py's `pet_id` extraction)."""
        event = await self._store.event_by_related(related_event)
        if not event or event.get("pet_id") is None:
            return None
        return await self.get(event["pet_id"])
