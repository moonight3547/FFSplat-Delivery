"""Safely unpack a manually downloaded organizer ZIP and record its exact bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from .contracts import sha256


def members(archive):
    result, seen = [], set()
    for info in archive.infolist():
        name = info.filename.rstrip("/")
        parts = name.split("/")
        if not name or "\\" in info.orig_filename or "\x00" in info.orig_filename or ":" in name or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"Unsafe archive member: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Archive symlink is not supported: {name}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate/case-colliding archive member: {name}")
        seen.add(name.casefold())
        result.append((info, PurePosixPath(name)))
    if not any(not info.is_dir() for info, _ in result):
        raise ValueError("Empty dataset archive")
    return result


def unpack(source, destination, record, expected_record=None):
    source, destination, record = map(Path, (source, destination, record))
    if destination.exists() or record.exists():
        raise FileExistsError("Preserve previous extraction/record; choose fresh paths")
    if record.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Store the archive record outside the extracted source tree")
    digest = sha256(source)
    expected = json.loads(Path(expected_record).read_text(encoding="utf-8")) if expected_record else None
    if expected and (expected["archive"]["sha256"] != digest or expected["archive"]["bytes"] != source.stat().st_size):
        raise ValueError("Archive differs from the pinned download snapshot")
    with zipfile.ZipFile(source) as archive:
        selected = members(archive)  # Validate every name before writing any member.
        total = sum(info.file_size for info, _ in selected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < total:
            raise OSError("Insufficient free space to unpack the archive")
        destination.mkdir()
        base, files = destination.resolve(), {}
        for info, relative in selected:
            path = destination.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(base):
                raise ValueError("Archive member resolves outside extraction directory")
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with archive.open(info) as stream, path.open("xb") as output:
                for block in iter(lambda: stream.read(4*1024*1024), b""):
                    h.update(block); output.write(block)
            # ZipFile checks each entry's CRC while streaming to EOF.
            files[relative.as_posix()] = dict(bytes=info.file_size, sha256=h.hexdigest())
    if sha256(source) != digest:
        raise ValueError("Archive changed during extraction; preserve the partial output for inspection")
    report = dict(schema_version=1, archive=dict(name=source.name, bytes=source.stat().st_size,
                  sha256=digest, crc="all extracted entries verified", sha256_provenance="observed user-supplied download"),
                  extracted_bytes=total, files=files)
    if expected and files != expected["files"]:
        raise ValueError("Extracted bytes differ from the pinned source inventory")
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--expected-record", type=Path, help="Compare with the committed archive inventory on a new host")
    args = p.parse_args()
    report = unpack(args.archive, args.destination, args.record, args.expected_record)
    print(json.dumps(dict(archive=report["archive"], files=len(report["files"]), extracted_bytes=report["extracted_bytes"]), indent=2))


if __name__ == "__main__":
    main()
