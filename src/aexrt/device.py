from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import struct
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RuntimeDeviceInfo:
    name: str
    device_type: str
    api: str
    vendor: str | None = None
    index: int = 0
    available: bool = True
    reason: str | None = None


@dataclass
class DeviceBuffer:
    device: RuntimeDeviceInfo
    nbytes: int
    dtype: str | None = None
    shape: Tuple[int, ...] | None = None
    label: str | None = None
    handle: Any = None


@dataclass
class PreparedDispatch:
    device: RuntimeDeviceInfo
    op: str
    input: DeviceBuffer
    output: DeviceBuffer
    element_count: int
    handle: Any = None


@dataclass
class PreparedConv2DDispatch:
    device: RuntimeDeviceInfo
    op: str
    input: DeviceBuffer
    weight: DeviceBuffer
    bias: DeviceBuffer
    output: DeviceBuffer
    desc: Tuple[int, ...]
    element_count: int
    handle: Any = None


@dataclass
class PreparedConv2DUploadDispatch:
    device: RuntimeDeviceInfo
    op: str
    weight: DeviceBuffer
    bias: DeviceBuffer
    output: DeviceBuffer
    desc: Tuple[int, ...]
    input_nbytes: int
    element_count: int
    ring_size: int
    handle: Any = None


@dataclass
class PreparedConv2DChainUploadDispatch:
    device: RuntimeDeviceInfo
    op: str
    output: DeviceBuffer
    descs: Tuple[Tuple[int, ...], ...]
    input_nbytes: int
    ring_size: int
    handle: Any = None


@dataclass
class PreparedGraphDispatch:
    device: RuntimeDeviceInfo
    op: str
    handle: Any = None


class RuntimeDevice(ABC):
    @abstractmethod
    def info(self) -> RuntimeDeviceInfo: ...

    @abstractmethod
    def allocate(
        self,
        nbytes: int,
        *,
        dtype: str | None = None,
        shape: Tuple[int, ...] | None = None,
        label: str | None = None,
    ) -> DeviceBuffer: ...

    @abstractmethod
    def upload(self, value: Any, *, label: str | None = None) -> DeviceBuffer: ...

    @abstractmethod
    def download(self, buffer: DeviceBuffer) -> np.ndarray: ...


class HostDevice(RuntimeDevice):
    """In-process device used to test the AEXRT HAL without vendor APIs."""

    def __init__(self) -> None:
        self._info = RuntimeDeviceInfo(name="Host Memory", device_type="cpu", api="aexrt_host")

    def info(self) -> RuntimeDeviceInfo:
        return self._info

    def allocate(
        self,
        nbytes: int,
        *,
        dtype: str | None = None,
        shape: Tuple[int, ...] | None = None,
        label: str | None = None,
    ) -> DeviceBuffer:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        return DeviceBuffer(self._info, int(nbytes), dtype=dtype, shape=shape, label=label, handle=bytearray(nbytes))

    def upload(self, value: Any, *, label: str | None = None) -> DeviceBuffer:
        arr = np.ascontiguousarray(value)
        buffer = self.allocate(arr.nbytes, dtype=str(arr.dtype), shape=tuple(arr.shape), label=label)
        buffer.handle[:] = arr.tobytes(order="C")
        return buffer

    def download(self, buffer: DeviceBuffer) -> np.ndarray:
        if buffer.dtype is None or buffer.shape is None:
            return np.frombuffer(bytes(buffer.handle), dtype=np.uint8).copy()
        return np.frombuffer(bytes(buffer.handle), dtype=np.dtype(buffer.dtype)).reshape(buffer.shape).copy()


class NativeD3D12Device(RuntimeDevice):
    """AEXRT native D3D12 entry point.

    This deliberately does not call DirectML. It expects a future native module
    that owns D3D12 device creation, resources, command queues, and compute
    dispatch. Until that module exists, construction fails loudly.
    """

    api = "aexrt_native_d3d12"

    def __init__(self, adapter_index: int = 0) -> None:
        self.adapter_index = int(adapter_index)
        self._native = _load_native_d3d12()
        if self._native is None:
            raise RuntimeError(
                "AEXRT native D3D12 runtime is not built. "
                "This path is intentionally separate from DirectML."
            )
        self._handle = self._native.create_device(self.adapter_index)

    @staticmethod
    def probe(adapter_index: int = 0) -> RuntimeDeviceInfo:
        native = _load_native_d3d12()
        if native is None:
            return RuntimeDeviceInfo(
                name="Native D3D12",
                device_type="d3d12",
                api=NativeD3D12Device.api,
                index=int(adapter_index),
                available=False,
                reason="native extension aexrt_native_d3d12 is not built",
            )
        try:
            info = native.probe_device(int(adapter_index))
            return RuntimeDeviceInfo(
                name=str(info.get("name", "Native D3D12")),
                device_type="d3d12",
                api=NativeD3D12Device.api,
                vendor=_vendor_name(info),
                index=int(adapter_index),
                available=True,
            )
        except Exception as e:
            return RuntimeDeviceInfo(
                name="Native D3D12",
                device_type="d3d12",
                api=NativeD3D12Device.api,
                index=int(adapter_index),
                available=False,
                reason=str(e),
            )

    def info(self) -> RuntimeDeviceInfo:
        info = self._native.device_info(self._handle)
        return RuntimeDeviceInfo(
            name=str(info.get("name", "Native D3D12")),
            device_type="d3d12",
            api=self.api,
            vendor=_vendor_name(info),
            index=self.adapter_index,
            available=True,
        )

    def allocate(
        self,
        nbytes: int,
        *,
        dtype: str | None = None,
        shape: Tuple[int, ...] | None = None,
        label: str | None = None,
    ) -> DeviceBuffer:
        handle = self._native.allocate_buffer(self._handle, int(nbytes), label or "")
        return DeviceBuffer(self.info(), int(nbytes), dtype=dtype, shape=shape, label=label, handle=handle)

    def allocate_uav(
        self,
        nbytes: int,
        *,
        dtype: str | None = None,
        shape: Tuple[int, ...] | None = None,
        label: str | None = None,
    ) -> DeviceBuffer:
        handle = self._native.allocate_uav_buffer(self._handle, int(nbytes), label or "")
        return DeviceBuffer(self.info(), int(nbytes), dtype=dtype, shape=shape, label=label, handle=handle)

    def upload(self, value: Any, *, label: str | None = None) -> DeviceBuffer:
        arr = np.ascontiguousarray(value)
        handle = self._native.upload_buffer(self._handle, arr.tobytes(order="C"), label or "")
        return DeviceBuffer(self.info(), arr.nbytes, dtype=str(arr.dtype), shape=tuple(arr.shape), label=label, handle=handle)

    def upload_into(self, buffer: DeviceBuffer, value: Any) -> None:
        arr = np.ascontiguousarray(value, dtype=np.dtype(buffer.dtype or "float32"))
        if arr.nbytes > buffer.nbytes:
            raise ValueError("upload value exceeds target buffer size")
        self._native.upload_buffer_into(self._handle, buffer.handle, arr.tobytes(order="C"))

    def create_buffer_view(
        self,
        buffer: DeviceBuffer,
        *,
        element_offset: int,
        element_count: int,
        dtype: str | None = None,
        shape: Tuple[int, ...] | None = None,
        label: str | None = None,
    ) -> DeviceBuffer:
        if int(element_offset) < 0 or int(element_count) <= 0:
            raise ValueError("buffer view offset/count must be non-negative/positive")
        handle = self._native.create_buffer_view(
            self._handle,
            buffer.handle,
            int(element_offset),
            int(element_count),
            label or "",
        )
        return DeviceBuffer(
            self.info(),
            int(element_count) * 4,
            dtype=dtype or buffer.dtype,
            shape=shape,
            label=label,
            handle=handle,
        )

    def download(self, buffer: DeviceBuffer) -> np.ndarray:
        raw = self._native.download_buffer(self._handle, buffer.handle, int(buffer.nbytes))
        if buffer.dtype is None or buffer.shape is None:
            return np.frombuffer(raw, dtype=np.uint8).copy()
        return np.frombuffer(raw, dtype=np.dtype(buffer.dtype)).reshape(buffer.shape).copy()

    def synchronize(self) -> None:
        self._native.synchronize(self._handle)

    def begin_batch(self) -> None:
        self._native.begin_batch(self._handle)

    def end_batch(self) -> None:
        self._native.end_batch(self._handle)

    def begin_prepared_graph(self) -> None:
        self._native.begin_prepared_graph(self._handle)

    def end_prepared_graph(self) -> PreparedGraphDispatch:
        handle = self._native.end_prepared_graph(self._handle)
        return PreparedGraphDispatch(self.info(), "PreparedGraph", handle=handle)

    def execute_prepared_graph(self, graph: PreparedGraphDispatch) -> None:
        self._native.execute_prepared_graph(self._handle, graph.handle)

    def buffer_info(self, buffer: DeviceBuffer) -> Dict[str, Any]:
        return dict(self._native.buffer_info(buffer.handle))

    def dispatch_relu_float32(self, buffer: DeviceBuffer, element_count: int) -> DeviceBuffer:
        handle = self._native.dispatch_relu_float32(self._handle, buffer.handle, int(element_count))
        return DeviceBuffer(
            self.info(),
            int(element_count) * 4,
            dtype="float32",
            shape=buffer.shape,
            label="relu",
            handle=handle,
        )

    def dispatch_relu_float32_into(
        self,
        input_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        element_count: int,
    ) -> None:
        self._native.dispatch_relu_float32_into(
            self._handle,
            input_buffer.handle,
            output_buffer.handle,
            int(element_count),
        )

    def prepare_relu_float32_dispatch(
        self,
        input_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        element_count: int,
    ) -> PreparedDispatch:
        handle = self._native.prepare_relu_float32_dispatch(
            self._handle,
            input_buffer.handle,
            output_buffer.handle,
            int(element_count),
        )
        return PreparedDispatch(
            self.info(),
            "Relu",
            input_buffer,
            output_buffer,
            int(element_count),
            handle=handle,
        )

    def execute_relu_float32_dispatch(self, dispatch: PreparedDispatch) -> None:
        self._native.execute_relu_float32_dispatch(self._handle, dispatch.handle)

    def dispatch_conv2d_silu_float32_into(
        self,
        input_buffer: DeviceBuffer,
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        desc: Dict[str, Any] | Tuple[int, ...],
    ) -> None:
        desc_tuple = _conv2d_desc_tuple(desc)
        self._native.dispatch_conv2d_silu_float32_into(
            self._handle,
            input_buffer.handle,
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            desc_tuple,
        )

    def prepare_conv2d_silu_float32_dispatch(
        self,
        input_buffer: DeviceBuffer,
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        desc: Dict[str, Any] | Tuple[int, ...],
    ) -> PreparedConv2DDispatch:
        desc_tuple = _conv2d_desc_tuple(desc)
        handle = self._native.prepare_conv2d_silu_float32_dispatch(
            self._handle,
            input_buffer.handle,
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            desc_tuple,
        )
        element_count = int(desc_tuple[0] * desc_tuple[4] * desc_tuple[5] * desc_tuple[6])
        return PreparedConv2DDispatch(
            self.info(),
            "Conv2D+SiLU",
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            desc_tuple,
            element_count,
            handle=handle,
        )

    def execute_conv2d_silu_float32_dispatch(self, dispatch: PreparedConv2DDispatch) -> None:
        self._native.execute_conv2d_silu_float32_dispatch(self._handle, dispatch.handle)

    def prepare_conv2d_silu_upload_float32_dispatch(
        self,
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        desc: Dict[str, Any] | Tuple[int, ...],
        *,
        ring_size: int = 2,
    ) -> PreparedConv2DUploadDispatch:
        desc_tuple = _conv2d_desc_tuple(desc)
        handle = self._native.prepare_conv2d_silu_upload_float32_dispatch(
            self._handle,
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            desc_tuple,
            int(ring_size),
        )
        input_nbytes = int(desc_tuple[0] * desc_tuple[1] * desc_tuple[2] * desc_tuple[3] * 4)
        element_count = int(desc_tuple[0] * desc_tuple[4] * desc_tuple[5] * desc_tuple[6])
        return PreparedConv2DUploadDispatch(
            self.info(),
            "Conv2D+SiLU+UploadRing",
            weight_buffer,
            bias_buffer,
            output_buffer,
            desc_tuple,
            input_nbytes,
            element_count,
            int(ring_size),
            handle=handle,
        )

    def execute_conv2d_silu_upload_float32_dispatch(self, dispatch: PreparedConv2DUploadDispatch, value: Any) -> None:
        arr = np.ascontiguousarray(value, dtype=np.float32)
        if arr.nbytes > dispatch.input_nbytes:
            raise ValueError("input value exceeds prepared upload dispatch size")
        self._native.execute_conv2d_silu_upload_float32_dispatch(self._handle, dispatch.handle, arr.tobytes(order="C"))

    def prepare_conv2d_silu_chain_upload_float32_dispatch(
        self,
        weight_buffers: Tuple[DeviceBuffer, ...] | list[DeviceBuffer],
        bias_buffers: Tuple[DeviceBuffer, ...] | list[DeviceBuffer],
        output_buffer: DeviceBuffer,
        descs: Tuple[Dict[str, Any] | Tuple[int, ...], ...] | list[Dict[str, Any] | Tuple[int, ...]],
        *,
        ring_size: int = 2,
    ) -> PreparedConv2DChainUploadDispatch:
        desc_tuples = tuple(_conv2d_desc_tuple(desc) for desc in descs)
        if not desc_tuples:
            raise ValueError("Conv2D chain requires at least one descriptor")
        if len(weight_buffers) != len(desc_tuples) or len(bias_buffers) != len(desc_tuples):
            raise ValueError("Conv2D chain weights, biases, and descriptors must have the same length")
        handle = self._native.prepare_conv2d_silu_chain_upload_float32_dispatch(
            self._handle,
            [b.handle for b in weight_buffers],
            [b.handle for b in bias_buffers],
            output_buffer.handle,
            list(desc_tuples),
            int(ring_size),
        )
        input_nbytes = int(desc_tuples[0][0] * desc_tuples[0][1] * desc_tuples[0][2] * desc_tuples[0][3] * 4)
        return PreparedConv2DChainUploadDispatch(
            self.info(),
            "Conv2D+SiLU+ChainUploadRing",
            output_buffer,
            desc_tuples,
            input_nbytes,
            int(ring_size),
            handle=handle,
        )

    def execute_conv2d_silu_chain_upload_float32_dispatch(self, dispatch: PreparedConv2DChainUploadDispatch, value: Any) -> None:
        arr = np.ascontiguousarray(value, dtype=np.float32)
        if arr.nbytes > dispatch.input_nbytes:
            raise ValueError("input value exceeds prepared chain upload dispatch size")
        self._native.execute_conv2d_silu_chain_upload_float32_dispatch(self._handle, dispatch.handle, arr.tobytes(order="C"))

    def dispatch_conv2d_float32_into(
        self,
        input_buffer: DeviceBuffer,
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        desc: Dict[str, Any] | Tuple[int, ...],
    ) -> None:
        desc_tuple = _conv2d_desc_tuple(desc)
        self._native.dispatch_conv2d_float32_into(
            self._handle,
            input_buffer.handle,
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            desc_tuple,
        )

    def dispatch_concat_conv1x1_float32_into(
        self,
        input_buffers: list[DeviceBuffer] | tuple[DeviceBuffer, ...],
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        *,
        activation: str = "linear",
    ) -> None:
        if not input_buffers:
            raise ValueError("concat-conv requires at least one input")
        if len(input_buffers) > 8:
            raise ValueError("native concat-conv currently supports up to 8 inputs")
        if output_buffer.shape is None or weight_buffer.shape is None:
            raise ValueError("concat-conv requires output and weight shape metadata")
        n, out_channels, out_h, out_w = (int(v) for v in output_buffer.shape)
        w_out, total_in_channels, kh, kw = (int(v) for v in weight_buffer.shape)
        if w_out != out_channels or kh != 1 or kw != 1:
            raise ValueError("concat-conv requires a [out_channels,total_in_channels,1,1] weight")
        channels: list[int] = []
        for buffer in input_buffers:
            if buffer.shape is None or len(buffer.shape) != 4:
                raise ValueError("concat-conv inputs must be rank-4 NCHW tensors")
            bn, c, h, w = (int(v) for v in buffer.shape)
            if bn != n or h != out_h or w != out_w:
                raise ValueError("concat-conv input shapes must match output N/H/W")
            channels.append(c)
        if sum(channels) != total_in_channels:
            raise ValueError("concat-conv input channels do not match weight")
        padded_channels = channels + [0] * (8 - len(channels))
        constants = [n, total_in_channels, out_h, out_w, out_channels, out_h, out_w, len(input_buffers), *padded_channels]
        activation_code = {"linear": 0, "silu": 1}[activation]
        self._native.dispatch_concat_conv1x1_float32_into(
            self._handle,
            [b.handle for b in input_buffers],
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            constants,
            activation_code,
        )

    def dispatch_concat_residual_conv1x1_float32_into(
        self,
        input_buffers: list[DeviceBuffer] | tuple[DeviceBuffer, ...],
        residual_buffers: list[DeviceBuffer] | tuple[DeviceBuffer, ...],
        residual_flags: list[int] | tuple[int, ...],
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        *,
        activation: str = "linear",
    ) -> None:
        if len(input_buffers) != len(residual_buffers) or len(input_buffers) != len(residual_flags):
            raise ValueError("concat-residual-conv inputs, residuals, and flags must have the same length")
        if not input_buffers:
            raise ValueError("concat-residual-conv requires at least one input")
        if len(input_buffers) > 8:
            raise ValueError("native concat-residual-conv currently supports up to 8 inputs")
        if output_buffer.shape is None or weight_buffer.shape is None:
            raise ValueError("concat-residual-conv requires output and weight shape metadata")
        n, out_channels, out_h, out_w = (int(v) for v in output_buffer.shape)
        w_out, total_in_channels, kh, kw = (int(v) for v in weight_buffer.shape)
        if w_out != out_channels or kh != 1 or kw != 1:
            raise ValueError("concat-residual-conv requires a [out_channels,total_in_channels,1,1] weight")
        channels: list[int] = []
        for input_buffer, residual_buffer, flag in zip(input_buffers, residual_buffers, residual_flags):
            if input_buffer.shape is None or len(input_buffer.shape) != 4:
                raise ValueError("concat-residual-conv inputs must be rank-4 NCHW tensors")
            bn, c, h, w = (int(v) for v in input_buffer.shape)
            if bn != n or h != out_h or w != out_w:
                raise ValueError("concat-residual-conv input shapes must match output N/H/W")
            if int(flag):
                if residual_buffer.shape is None or tuple(int(v) for v in residual_buffer.shape) != (bn, c, h, w):
                    raise ValueError("concat-residual-conv residual shape must match input when flag is set")
            channels.append(c)
        if sum(channels) != total_in_channels:
            raise ValueError("concat-residual-conv input channels do not match weight")
        padded_channels = channels + [0] * (8 - len(channels))
        padded_flags = [1 if int(flag) else 0 for flag in residual_flags] + [0] * (8 - len(residual_flags))
        constants = [n, total_in_channels, out_h, out_w, out_channels, out_h, out_w, len(input_buffers), *padded_channels, *padded_flags]
        activation_code = {"linear": 0, "silu": 1}[activation]
        self._native.dispatch_concat_residual_conv1x1_float32_into(
            self._handle,
            [b.handle for b in input_buffers],
            [b.handle for b in residual_buffers],
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            constants,
            activation_code,
        )

    def dispatch_concat_conv1x1_int8_float32_into(
        self,
        input_buffers: list[DeviceBuffer] | tuple[DeviceBuffer, ...],
        weight_buffer: DeviceBuffer,
        scale_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        *,
        activation_scale: float,
        activation: str = "linear",
    ) -> None:
        if not input_buffers:
            raise ValueError("int8 concat-conv requires at least one input")
        if len(input_buffers) > 8:
            raise ValueError("native int8 concat-conv currently supports up to 8 inputs")
        if output_buffer.shape is None or weight_buffer.shape is None:
            raise ValueError("int8 concat-conv requires output and weight shape metadata")
        n, out_channels, out_h, out_w = (int(v) for v in output_buffer.shape)
        w_out, total_in_channels, kh, kw = (int(v) for v in weight_buffer.shape)
        if w_out != out_channels or kh != 1 or kw != 1:
            raise ValueError("int8 concat-conv requires logical [out_channels,total_in_channels,1,1] weight shape")
        channels: list[int] = []
        for buffer in input_buffers:
            if buffer.shape is None or len(buffer.shape) != 4:
                raise ValueError("int8 concat-conv inputs must be rank-4 NCHW tensors")
            bn, c, h, w = (int(v) for v in buffer.shape)
            if bn != n or h != out_h or w != out_w:
                raise ValueError("int8 concat-conv input shapes must match output N/H/W")
            channels.append(c)
        if sum(channels) != total_in_channels:
            raise ValueError("int8 concat-conv input channels do not match weight")
        act_scale = float(activation_scale)
        if not np.isfinite(act_scale) or act_scale <= 0:
            raise ValueError("activation_scale must be a positive finite float")
        inv_bits = struct.unpack("<I", struct.pack("<f", np.float32(1.0 / act_scale)))[0]
        padded_channels = channels + [0] * (8 - len(channels))
        constants = [n, total_in_channels, out_h, out_w, out_channels, out_h, out_w, len(input_buffers), *padded_channels, inv_bits, 0, 0, 0]
        activation_code = {"linear": 0, "silu": 1}[activation]
        self._native.dispatch_concat_conv1x1_int8_float32_into(
            self._handle,
            [b.handle for b in input_buffers],
            weight_buffer.handle,
            scale_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            constants,
            activation_code,
        )

    def dispatch_c2f_bottleneck_tiled_float32_into(
        self,
        input_buffer: DeviceBuffer,
        w1_buffer: DeviceBuffer,
        b1_buffer: DeviceBuffer,
        w2_buffer: DeviceBuffer,
        b2_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
    ) -> None:
        if input_buffer.shape is None or w1_buffer.shape is None or w2_buffer.shape is None or output_buffer.shape is None:
            raise ValueError("C2f bottleneck requires input/weight/output shape metadata")
        n, in_channels, h, w = (int(v) for v in input_buffer.shape)
        mid_channels, w1_ic, kh1, kw1 = (int(v) for v in w1_buffer.shape)
        out_channels, w2_ic, kh2, kw2 = (int(v) for v in w2_buffer.shape)
        if tuple(int(v) for v in output_buffer.shape) != (n, out_channels, h, w):
            raise ValueError("C2f bottleneck output shape must match N/H/W")
        if in_channels != out_channels or w1_ic != in_channels or w2_ic != mid_channels or (kh1, kw1, kh2, kw2) != (3, 3, 3, 3):
            raise ValueError("C2f bottleneck requires 3x3 Conv shapes C->M->C for residual add")
        constants = [n, in_channels, h, w, mid_channels, out_channels, n * out_channels * h * w, 0]
        self._native.dispatch_c2f_bottleneck_tiled_float32_into(
            self._handle,
            input_buffer.handle,
            w1_buffer.handle,
            b1_buffer.handle,
            w2_buffer.handle,
            b2_buffer.handle,
            output_buffer.handle,
            constants,
        )

    def dispatch_sppf_tail_float32_into(
        self,
        input_buffer: DeviceBuffer,
        weight_buffer: DeviceBuffer,
        bias_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        *,
        activation: str = "linear",
    ) -> None:
        if input_buffer.shape is None or weight_buffer.shape is None or output_buffer.shape is None:
            raise ValueError("SPPF tail requires input/weight/output shape metadata")
        n, in_channels, h, w = (int(v) for v in input_buffer.shape)
        out_channels, total_in_channels, kh, kw = (int(v) for v in weight_buffer.shape)
        if tuple(int(v) for v in output_buffer.shape) != (n, out_channels, h, w):
            raise ValueError("SPPF tail output shape must match input N/H/W and weight out channels")
        if total_in_channels != in_channels * 4 or kh != 1 or kw != 1:
            raise ValueError("SPPF tail requires a logical [out_channels,4*in_channels,1,1] weight")
        constants = [n, in_channels, h, w, out_channels, n * out_channels * h * w, 0, 0]
        activation_code = {"linear": 0, "silu": 1}[activation]
        self._native.dispatch_sppf_tail_float32_into(
            self._handle,
            input_buffer.handle,
            weight_buffer.handle,
            bias_buffer.handle,
            output_buffer.handle,
            constants,
            activation_code,
        )

    def dispatch_unary_float32_into(self, op: str, input_buffer: DeviceBuffer, output_buffer: DeviceBuffer) -> None:
        op_code = {"Identity": 0, "Relu": 1, "Sigmoid": 2, "Tanh": 3, "Gelu": 4}[op]
        self._native.dispatch_unary_float32_into(
            self._handle,
            input_buffer.handle,
            output_buffer.handle,
            _numel(output_buffer.shape),
            op_code,
        )

    def dispatch_binary_broadcast_float32_into(
        self,
        op: str,
        a: DeviceBuffer,
        b: DeviceBuffer,
        output: DeviceBuffer,
    ) -> None:
        op_code = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3}[op]
        constants = _binary_broadcast_constants(op_code, a.shape, b.shape, output.shape)
        self._native.dispatch_binary_broadcast_float32_into(
            self._handle,
            a.handle,
            b.handle,
            output.handle,
            constants,
        )

    def dispatch_slice_float32_into(
        self,
        input_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        *,
        axis: int,
        start: int,
    ) -> None:
        rank = len(input_buffer.shape or ())
        axis_i = int(axis)
        if axis_i < 0:
            axis_i += rank
        padded_axis = axis_i + (4 - rank)
        in_shape = _shape4(input_buffer.shape)
        out_shape = _shape4(output_buffer.shape)
        constants = [_numel(output_buffer.shape), rank, 0, 0, *in_shape, *out_shape, padded_axis, int(start)]
        self._native.dispatch_slice_float32_into(self._handle, input_buffer.handle, output_buffer.handle, constants)

    def dispatch_concat_float32_into(
        self,
        input_buffers: list[DeviceBuffer] | tuple[DeviceBuffer, ...],
        output_buffer: DeviceBuffer,
        *,
        axis: int,
    ) -> None:
        if not input_buffers:
            raise ValueError("concat requires at least one input")
        if len(input_buffers) > 8:
            raise ValueError("native concat currently supports up to 8 inputs")
        rank = len(output_buffer.shape or ())
        out_shape = _shape4(output_buffer.shape)
        axis_i = int(axis)
        if axis_i < 0:
            axis_i += rank
        padded_axis = axis_i + (4 - rank)
        sizes = [int((b.shape or ())[axis_i]) for b in input_buffers]
        sizes.extend([sizes[-1]] * (8 - len(sizes)))
        constants = [_numel(output_buffer.shape), rank, padded_axis, len(input_buffers), *out_shape, *sizes, 0, 0]
        self._native.dispatch_concat_float32_into(
            self._handle,
            [b.handle for b in input_buffers],
            output_buffer.handle,
            constants,
        )

    def dispatch_resize_nearest_float32_into(self, input_buffer: DeviceBuffer, output_buffer: DeviceBuffer) -> None:
        n, c, in_h, in_w = (int(x) for x in input_buffer.shape)
        _, _, out_h, out_w = (int(x) for x in output_buffer.shape)
        self._native.dispatch_resize_nearest_float32_into(
            self._handle,
            input_buffer.handle,
            output_buffer.handle,
            [n, c, in_h, in_w, out_h, out_w],
        )

    def dispatch_maxpool2d_float32_into(
        self,
        input_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        *,
        kernel_shape: Tuple[int, int] | list[int],
        strides: Tuple[int, int] | list[int],
        pads: Tuple[int, int, int, int] | list[int],
        dilations: Tuple[int, int] | list[int],
    ) -> None:
        n, c, in_h, in_w = (int(x) for x in input_buffer.shape)
        _, _, out_h, out_w = (int(x) for x in output_buffer.shape)
        constants = [
            n, c, in_h, in_w, out_h, out_w,
            int(kernel_shape[0]), int(kernel_shape[1]),
            int(strides[0]), int(strides[1]),
            int(pads[0]), int(pads[1]),
            int(dilations[0]),
        ]
        self._native.dispatch_maxpool2d_float32_into(self._handle, input_buffer.handle, output_buffer.handle, constants)

    def dispatch_transpose_float32_into(
        self,
        input_buffer: DeviceBuffer,
        output_buffer: DeviceBuffer,
        axes: Tuple[int, ...] | list[int],
    ) -> None:
        rank = len(input_buffer.shape or ())
        axes_i = [int(x) for x in axes]
        in_shape = _shape4(input_buffer.shape)
        out_shape = _shape4(output_buffer.shape)
        axes4 = axes_i + list(range(len(axes_i), 4))
        constants = [_numel(output_buffer.shape), rank, 0, 0, *in_shape, *out_shape, *axes4, 0]
        self._native.dispatch_transpose_float32_into(self._handle, input_buffer.handle, output_buffer.handle, constants)

    def dispatch_softmax_axis1_float32_into(self, input_buffer: DeviceBuffer, output_buffer: DeviceBuffer) -> None:
        n, c, h, w = (int(x) for x in input_buffer.shape)
        groups = n * h * w
        self._native.dispatch_softmax_axis1_float32_into(
            self._handle,
            input_buffer.handle,
            output_buffer.handle,
            [n, c, h, w, 1, groups],
        )

    def dispatch_dfl_project_float32_into(self, input_buffer: DeviceBuffer, output_buffer: DeviceBuffer) -> None:
        _, bins, box_dims, anchors = (int(x) for x in input_buffer.shape)
        count = int(box_dims * anchors)
        self._native.dispatch_dfl_project_float32_into(
            self._handle,
            input_buffer.handle,
            output_buffer.handle,
            [bins, box_dims, anchors, count],
        )

    def dispatch_yolo_decode_filter_float32(
        self,
        yolo_output: DeviceBuffer,
        detections: DeviceBuffer,
        counter: DeviceBuffer,
        *,
        anchors: int,
        channels: int,
        classes: int,
        max_detections: int,
        conf_threshold: float = 0.25,
    ) -> None:
        self._native.dispatch_yolo_decode_filter_float32(
            self._handle,
            yolo_output.handle,
            detections.handle,
            counter.handle,
            int(anchors),
            int(channels),
            int(classes),
            int(max_detections),
            float(conf_threshold),
        )

    def allocate_yolo_detection_buffers(self, max_detections: int, *, label: str | None = None) -> tuple[DeviceBuffer, DeviceBuffer]:
        detections = self.allocate_uav(
            int(max_detections) * 6 * 4,
            dtype="float32",
            shape=(int(max_detections), 6),
            label=(label or "yolo") + "_detections",
        )
        counter = self.allocate_uav(
            4,
            dtype="uint32",
            shape=(1,),
            label=(label or "yolo") + "_counter",
        )
        return detections, counter

    def allocate_yolo_nms_buffers(
        self,
        max_candidates: int,
        max_detections: int,
        *,
        label: str | None = None,
    ) -> tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer]:
        prefix = label or "yolo_nms"
        candidates = self.allocate_uav(
            int(max_candidates) * 6 * 4,
            dtype="float32",
            shape=(int(max_candidates), 6),
            label=prefix + "_candidates",
        )
        candidate_counter = self.allocate_uav(4, dtype="uint32", shape=(1,), label=prefix + "_candidate_counter")
        keep_flags = self.allocate_uav(int(max_candidates) * 4, dtype="uint32", shape=(int(max_candidates),), label=prefix + "_keep")
        detections = self.allocate_uav(
            int(max_detections) * 6 * 4,
            dtype="float32",
            shape=(int(max_detections), 6),
            label=prefix + "_topk",
        )
        counter = self.allocate_uav(4, dtype="uint32", shape=(1,), label=prefix + "_counter")
        return candidates, candidate_counter, keep_flags, detections, counter

    def dispatch_yolo_decode_nms_float32(
        self,
        yolo_output: DeviceBuffer,
        candidates: DeviceBuffer,
        candidate_counter: DeviceBuffer,
        keep_flags: DeviceBuffer,
        detections: DeviceBuffer,
        counter: DeviceBuffer,
        *,
        anchors: int,
        channels: int,
        classes: int,
        max_candidates: int,
        max_detections: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._native.dispatch_yolo_decode_nms_float32(
            self._handle,
            yolo_output.handle,
            candidates.handle,
            candidate_counter.handle,
            keep_flags.handle,
            detections.handle,
            counter.handle,
            int(anchors),
            int(channels),
            int(classes),
            int(max_candidates),
            int(max_detections),
            float(conf_threshold),
            float(iou_threshold),
        )

    def dispatch_yolo_head_decode_nms_float32(
        self,
        box_outputs: list[DeviceBuffer] | tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
        class_outputs: list[DeviceBuffer] | tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
        candidates: DeviceBuffer,
        candidate_counter: DeviceBuffer,
        keep_flags: DeviceBuffer,
        detections: DeviceBuffer,
        counter: DeviceBuffer,
        *,
        classes: int,
        max_candidates: int,
        max_detections: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        strides: tuple[float, float, float] = (8.0, 16.0, 32.0),
    ) -> None:
        if len(box_outputs) != 3 or len(class_outputs) != 3:
            raise ValueError("YOLO head decode requires exactly three bbox and three class outputs")
        dims = []
        classes_i = int(classes)
        for box, cls in zip(box_outputs, class_outputs):
            if box.shape is None or cls.shape is None or len(box.shape) != 4 or len(cls.shape) != 4:
                raise ValueError("YOLO head tensors must be rank-4 NCHW buffers")
            if int(box.shape[0]) != 1 or int(cls.shape[0]) != 1 or int(box.shape[1]) != 64 or int(cls.shape[1]) != classes_i:
                raise ValueError("YOLO head expects bbox [1,64,H,W] and class [1,classes,H,W] buffers")
            if tuple(int(v) for v in box.shape[2:]) != tuple(int(v) for v in cls.shape[2:]):
                raise ValueError("YOLO bbox/class head tensors must share H/W per scale")
            dims.extend([int(box.shape[2]), int(box.shape[3])])
        self._native.dispatch_yolo_head_decode_nms_float32(
            self._handle,
            box_outputs[0].handle,
            class_outputs[0].handle,
            box_outputs[1].handle,
            class_outputs[1].handle,
            box_outputs[2].handle,
            class_outputs[2].handle,
            candidates.handle,
            candidate_counter.handle,
            keep_flags.handle,
            detections.handle,
            counter.handle,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            dims[4],
            dims[5],
            classes_i,
            int(max_candidates),
            int(max_detections),
            float(conf_threshold),
            float(iou_threshold),
            float(strides[0]),
            float(strides[1]),
            float(strides[2]),
        )

    def download_yolo_topk(self, detections: DeviceBuffer, counter: DeviceBuffer) -> np.ndarray:
        count = int(self.download(counter).reshape(-1)[0])
        if count <= 0:
            return np.empty((0, 6), dtype=np.float32)
        max_rows = detections.shape[0] if detections.shape else count
        count = min(count, int(max_rows))
        return self.download(detections).reshape(int(max_rows), 6)[:count].copy()


def _load_native_d3d12() -> Any | None:
    try:
        import aexrt_native_d3d12
    except Exception:
        return None
    return aexrt_native_d3d12


def _conv2d_desc_tuple(desc: Dict[str, Any] | Tuple[int, ...]) -> Tuple[int, ...]:
    keys = (
        "batch",
        "in_channels",
        "in_h",
        "in_w",
        "out_channels",
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_top",
        "pad_left",
        "dilation_h",
        "dilation_w",
        "groups",
    )
    if isinstance(desc, dict):
        values = tuple(int(desc[k]) for k in keys)
    else:
        values = tuple(int(x) for x in desc)
    if len(values) != len(keys):
        raise ValueError(f"Conv2D descriptor must have {len(keys)} values")
    return values


def _numel(shape: Tuple[int, ...] | None) -> int:
    total = 1
    for dim in shape or ():
        total *= int(dim)
    return int(total)


def _shape4(shape: Tuple[int, ...] | None) -> tuple[int, int, int, int]:
    dims = [int(x) for x in (shape or ())]
    if len(dims) > 4:
        raise ValueError("native D3D12 rank<=4 primitive expected")
    dims = [1] * (4 - len(dims)) + dims
    return tuple(dims)  # type: ignore[return-value]


def _strides4(shape4: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        int(shape4[1] * shape4[2] * shape4[3]),
        int(shape4[2] * shape4[3]),
        int(shape4[3]),
        1,
    )


def _binary_broadcast_constants(
    op_code: int,
    a_shape: Tuple[int, ...] | None,
    b_shape: Tuple[int, ...] | None,
    out_shape: Tuple[int, ...] | None,
) -> list[int]:
    out4 = _shape4(out_shape)
    a4 = _shape4(a_shape)
    b4 = _shape4(b_shape)
    astr = _strides4(a4)
    bstr = _strides4(b4)
    return [
        _numel(out_shape),
        int(op_code),
        len(out_shape or ()),
        0,
        *out4,
        *a4,
        *b4,
        *astr,
        *bstr,
    ]


def _vendor_name(info: Dict[str, Any]) -> str | None:
    vendor_id = info.get("vendor_id")
    if vendor_id is None:
        return info.get("vendor")
    known = {
        0x1002: "AMD",
        0x10DE: "NVIDIA",
        0x8086: "Intel",
        0x1414: "Microsoft",
        0x13B5: "ARM",
        0x5143: "Qualcomm",
    }
    try:
        vendor_int = int(vendor_id)
    except Exception:
        return str(vendor_id)
    return known.get(vendor_int, f"0x{vendor_int:04X}")
