# Copyright (c) 2026 KU Leuven.
"""Exercise real DMA helpers with deferred copies and strict L1/L2 bounds."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("cc"), "Host C compiler is required")
class TestDmaTransfers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        folder = Path(cls.directory.name)
        ops = Path(__file__).resolve().parents[1] / "ops"
        (folder / "pmsis.h").write_text(
            r"""
#pragma once
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#define PI_CL_DMA_DIR_EXT2LOC 1
#define PI_CL_DMA_DIR_LOC2EXT 0
typedef struct {
 uintptr_t external, local;
 size_t bytes, stride, length;
 int direction;
} pi_cl_dma_cmd_t;
extern uintptr_t host_external, host_local;
extern size_t host_external_bytes, host_local_bytes;
extern unsigned host_commands, host_waits, host_pending;
static inline void pi_cl_dma_cmd_2d(uintptr_t ext,uintptr_t loc,size_t size,
 size_t stride,size_t length,int direction,pi_cl_dma_cmd_t* cmd) {
 assert(!host_pending && size && size<=65535 && stride && stride<=65535 && length && length<=65535);
 size_t span=((size-1)/length)*stride+(size-1)%length+1;
 assert(ext>=host_external && span<=host_external_bytes && ext-host_external<=host_external_bytes-span);
 assert(loc>=host_local && size<=host_local_bytes && loc-host_local<=host_local_bytes-size);
 assert(direction==PI_CL_DMA_DIR_EXT2LOC || direction==PI_CL_DMA_DIR_LOC2EXT);
 *cmd=(pi_cl_dma_cmd_t){ext,loc,size,stride,length,direction};
 ++host_commands;host_pending=1;
}
static inline void pi_cl_dma_cmd(uintptr_t ext,uintptr_t loc,size_t size,int direction,pi_cl_dma_cmd_t* cmd) {
 assert(size<=65535);
 pi_cl_dma_cmd_2d(ext,loc,size,size,size,direction,cmd);
}
static inline void pi_cl_dma_cmd_wait(pi_cl_dma_cmd_t* cmd) {
 assert(host_pending);++host_waits;host_pending=0;
 while(cmd->bytes) {
  size_t size=cmd->bytes<cmd->length?cmd->bytes:cmd->length;
  if(cmd->direction==PI_CL_DMA_DIR_EXT2LOC)memcpy((void*)cmd->local,(void*)cmd->external,size);
  else memcpy((void*)cmd->external,(void*)cmd->local,size);
  cmd->external+=cmd->stride;cmd->local+=size;cmd->bytes-=size;
 }
}
"""
        )
        (folder / "test.c").write_text(
            r"""
#include "pulp_dma.h"
uintptr_t host_external,host_local;
size_t host_external_bytes,host_local_bytes;
unsigned host_commands,host_waits,host_pending;
static void setup(void* ext,void* loc,size_t ext_bytes,size_t loc_bytes) {
 host_external=(uintptr_t)ext;host_local=(uintptr_t)loc;
 host_external_bytes=ext_bytes;host_local_bytes=loc_bytes;
 host_commands=host_waits=host_pending=0;
}
void copy_in(void* dest,void* source,size_t size) {
 setup(source,dest,size,size);pulp_dma_copy_in(dest,source,size);
}
void copy_out(void* dest,void* source,size_t size) {
 setup(dest,source,size,size);pulp_dma_copy_out(dest,source,size);
}
void gather(void* dest,void* source,size_t pixels,size_t channels) {
 setup(source,dest,pixels?(pixels-1)*channels+1:0,pixels);
 pulp_dma_gather_channel(dest,source,pixels,channels);
}
"""
        )
        library = folder / "dma.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined",
                "-I" + str(folder),
                "-I" + str(ops),
                str(folder / "test.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        for name in ("copy_in", "copy_out"):
            getattr(cls.lib, name).argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
        cls.lib.gather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_copy_directions_command_limit_and_byte_tails(self):
        for count in (0, 1, 3, 4, 127, 65531, 65532, 65533, 65535, 65536, 204800):
            for direction in ("copy_in", "copy_out"):
                for offset in (0, 1, 3):
                    with self.subTest(count=count, direction=direction, offset=offset):
                        data = bytes((i * 13 + 7) % 256 for i in range(count))
                        source = ctypes.create_string_buffer(data)
                        output = ctypes.create_string_buffer(
                            b"\xa5" * (count + offset + 8)
                        )
                        dest = ctypes.addressof(output) + offset + 4
                        getattr(self.lib, direction)(dest, source, count)
                        self.assertEqual(
                            output.raw[offset + 4 : offset + 4 + count], data
                        )
                        self.assertEqual(
                            output.raw[: offset + 4], b"\xa5" * (offset + 4)
                        )
                        self.assertEqual(
                            output.raw[offset + 4 + count : -1], b"\xa5" * 4
                        )
                        expected_commands = (count + 65531) // 65532
                        self.assertEqual(
                            ctypes.c_uint.in_dll(self.lib, "host_commands").value,
                            expected_commands,
                        )
                        self.assertEqual(
                            ctypes.c_uint.in_dll(self.lib, "host_waits").value,
                            expected_commands,
                        )
                        self.assertEqual(
                            ctypes.c_uint.in_dll(self.lib, "host_pending").value, 0
                        )

    def test_strided_channel_gather_and_split_planes(self):
        for pixels, channels in (
            (0, 3),
            (1, 7),
            (35, 1),
            (35, 5),
            (65537, 3),
            (3, 65535),
        ):
            with self.subTest(pixels=pixels, channels=channels):
                count = (pixels - 1) * channels + 1 if pixels else 0
                data = bytes((i * 11 + 3) % 256 for i in range(count))
                source = ctypes.create_string_buffer(data)
                output = ctypes.create_string_buffer(b"\xa5" * (pixels + 8))
                self.lib.gather(ctypes.addressof(output) + 4, source, pixels, channels)
                self.assertEqual(output.raw[4 : 4 + pixels], data[::channels])
                self.assertEqual(output.raw[:4], b"\xa5" * 4)
                self.assertEqual(output.raw[4 + pixels : -1], b"\xa5" * 4)
                self.assertEqual(
                    ctypes.c_uint.in_dll(self.lib, "host_waits").value,
                    (pixels + 65531) // 65532,
                )


if __name__ == "__main__":
    unittest.main()
