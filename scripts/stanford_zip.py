"""Read the original Stanford ZIPs, including the large ZIP32 bound edition."""

from contextlib import contextmanager
from struct import Struct
from zipfile import BadZipFile, ZipFile


ZIP32_LIMIT = 1 << 32
LOCAL_HEADER = Struct("<4s5H3I2H")
DATA_DESCRIPTOR = Struct("<4s3I")


def _restore_bound_offsets(archive, stream):
    # This archive's central directory stores positions modulo 2**32. Derive
    # physical positions from local records, verifying each against the index.
    position = 0
    for item in archive.infolist():
        def invalid():
            return BadZipFile(f"Inconsistent ZIP32 local record for {item.filename!r}")

        if position % ZIP32_LIMIT != item.header_offset % ZIP32_LIMIT:
            raise invalid()
        stream.seek(position)
        header = stream.read(LOCAL_HEADER.size)
        if len(header) != LOCAL_HEADER.size:
            raise invalid()
        signature, _, flags, method, _, _, crc, size, raw_size, name_len, extra_len = LOCAL_HEADER.unpack(header)
        if (signature, flags, method) != (b"PK\x03\x04", item.flag_bits, item.compress_type):
            raise invalid()
        if stream.read(name_len) != item.filename.encode("ascii"):
            raise invalid()
        end = stream.tell() + extra_len + item.compress_size
        expected = (item.CRC, item.compress_size, item.file_size)
        if flags & 8:
            stream.seek(end)
            data = stream.read(DATA_DESCRIPTOR.size)
            if len(data) != DATA_DESCRIPTOR.size or DATA_DESCRIPTOR.unpack(data) != (b"PK\x07\x08", *expected):
                raise invalid()
            end += DATA_DESCRIPTOR.size
        elif (crc, size, raw_size) != expected:
            raise invalid()

        item.header_offset = position
        # ZipFile.open uses this boundary for its overlapping-entry check.
        item._end_offset = end
        position = end

    if position != archive.start_dir:
        raise BadZipFile("ZIP32 local records do not end at the central directory")


@contextmanager
def open_stanford_zip(path):
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if (ZIP32_LIMIT < archive.start_dir < ZIP32_LIMIT * 2 and entries
                and entries[0].filename == "hein-bound/" and entries[0].header_offset == ZIP32_LIMIT):
            with open(path, "rb") as stream:
                _restore_bound_offsets(archive, stream)
        yield archive
