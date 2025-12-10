from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from typing import Callable, Dict, Iterable, Tuple

import neural_compressor.adaptor.torch_utils.util as nc_util

MseKey = Tuple[str, str]
MseMap = Dict[MseKey, float]


@contextmanager
def capture_mse_v2_sensitivity() -> Iterable[Tuple[MseMap, MseMap]]:
    """Monkeypatch INC PyTorch FX helpers to capture per-op MSE.

    This context manager temporarily wraps
    ``neural_compressor.adaptor.torch_utils.util.get_mse_order_per_fp32``
    and ``get_mse_order_per_int8`` so that callers can inspect the
    per-op MSE values computed during an ``mse_v2`` tuning run.

    The yielded value is a pair of dictionaries:
    ``(fp32_mse_map, int8_mse_map)``, mapping
    ``(op_name, op_type)`` to the last-observed MSE value for that op.

    The current implementation only establishes the patching scaffold;
    the sensitivity driver script is expected to populate and consume
    these maps as needed for reporting.
    """
    original_fp32: Callable[..., object] = nc_util.get_mse_order_per_fp32
    original_int8: Callable[..., object] = nc_util.get_mse_order_per_int8

    fp32_mse: MseMap = {}
    int8_mse: MseMap = {}

    torch = nc_util.torch

    def wrapped_fp32(adaptor, model, example_inp, tune_cfg):  # type: ignore[no-untyped-def]
        """Patched get_mse_order_per_fp32 that also records MSE values."""
        inner_output = None

        nc_util.logger.info("[MSE_PATCH] wrapped_fp32 invoked for mse_v2 sensitivity capture.")

        def output_hook(self, input, output):  # type: ignore[no-untyped-def]
            nonlocal inner_output
            inner_output = output
            return output

        op_type_dict = {}
        for k, v in tune_cfg["op"].keys():
            op_type_dict[k] = v

        from neural_compressor.adaptor.pytorch import (  # type: ignore[import]
            PyTorch_FXAdaptor,
            _cfg_to_qconfig,
            _cfgs_to_fx_cfgs,
        )

        op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
        last_module_name = list(op_cfgs.keys())[-1]
        module = nc_util.fetch_module(model, last_module_name)
        module.register_forward_hook(output_hook)
        nc_util.simple_inference(model, example_inp)
        inner_output_fp32 = inner_output

        fx_op_cfgs = {}
        fallback_order: Dict[Tuple[str, str], float] = {}
        nc_util.logger.info("Evaluate the sensitivity for each int8 operation")

        max_ops_env = os.environ.get("INC_MSE_MAX_OPS", "")
        try:
            max_ops = int(max_ops_env) if max_ops_env else 0
        except ValueError:
            max_ops = 0

        for idx, (op_name, qconfig) in enumerate(nc_util.tqdm(op_cfgs.items())):
            if max_ops > 0 and idx >= max_ops:
                break
            if op_name == "bf16_ops_list":
                continue
            global_op_cfg_mapping = nc_util.op_cfg_mapping
            if op_name not in global_op_cfg_mapping:
                global_op_cfg_mapping[op_name] = qconfig

            tmp_model = copy.deepcopy(model)
            if not qconfig:
                nc_util.logger.debug(f"No qconfig for {op_name}, next op.")
                continue

            op_cfgs[op_name] = None
            fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
            op_cfgs[op_name] = qconfig

            from torch.quantization.quantize_fx import convert_fx, prepare_fx

            if adaptor.sub_module_list is None:
                if adaptor.version.release >= nc_util.Version("1.13.0").release:  # type: ignore[attr-defined]
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
                else:
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs)
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(
                    adaptor.sub_module_list, fx_op_cfgs, tmp_model, prefix=""
                )
            nc_util.simple_inference(tmp_model, example_inp)
            if adaptor.sub_module_list is None:
                tmp_model = convert_fx(tmp_model)
            else:
                PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, tmp_model, prefix="")

            module = nc_util.fetch_module(tmp_model, list(op_cfgs.keys())[-1])
            module.register_forward_hook(output_hook)
            nc_util.simple_inference(tmp_model, example_inp)
            inner_output_int8 = inner_output.dequantize() if inner_output.dtype == torch.quint8 else inner_output
            mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
            key = (op_name, op_type_dict[op_name])
            fallback_order[key] = float(mse_val.item())
            fp32_mse[key] = float(mse_val.item())

        nc_util.logger.debug(f"fallback order: {fallback_order}")
        ordered_ops = sorted(fallback_order.keys(), key=lambda key: fallback_order[key], reverse=False)
        if not ordered_ops:
            return ordered_ops

        min_mse, max_mse = fallback_order[ordered_ops[0]], fallback_order[ordered_ops[-1]]
        if min_mse < 0.8 * max_mse:
            nc_util.logger.debug("Return the sorted ops early.")
            return ordered_ops

        double_check_list = []
        for op_name in ordered_ops:
            if min_mse <= fallback_order[op_name] <= (max_mse - min_mse) * 0.1 + min_mse:
                double_check_list.append(op_name)

        check_num = min(len(ordered_ops) // 10 + 1, 5)
        double_check_list = ordered_ops[:check_num]
        nc_util.logger.debug(f"double check list: {double_check_list}")
        worst_op_name = ordered_ops[-1]
        op_cfgs[worst_op_name[0]] = None
        new_fallback_order: Dict[Tuple[str, str], float] = {}

        nc_util.logger.info("Evaluate the sensitivity gradient for selected operations")
        for op_name, op_type in nc_util.tqdm(double_check_list):
            tmp_model = copy.deepcopy(model)
            qconfig = op_cfgs[op_name]
            op_cfgs[op_name] = None
            fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
            op_cfgs[op_name] = qconfig

            from torch.quantization.quantize_fx import convert_fx, prepare_fx

            if adaptor.sub_module_list is None:
                if adaptor.version.release >= nc_util.Version("1.13.0").release:  # type: ignore[attr-defined]
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
                else:
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs)
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(
                    adaptor.sub_module_list, fx_op_cfgs, tmp_model, prefix=""
                )
            nc_util.simple_inference(tmp_model, example_inp)
            if adaptor.sub_module_list is None:
                tmp_model = convert_fx(tmp_model)
            else:
                PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, tmp_model, prefix="")

            module = nc_util.fetch_module(tmp_model, last_module_name)
            module.register_forward_hook(output_hook)
            nc_util.simple_inference(tmp_model, example_inp)
            inner_output_int8 = inner_output.dequantize() if inner_output.dtype == torch.quint8 else inner_output
            mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
            key = (op_name, op_type_dict[op_name])
            new_fallback_order[key] = float(mse_val.item())
            fp32_mse[key] = float(mse_val.item())

        ordered_ops = sorted(new_fallback_order.keys(), key=lambda key: new_fallback_order[key], reverse=False)
        return ordered_ops

    def wrapped_int8(adaptor, fp32_model, example_input, tune_cfg):  # type: ignore[no-untyped-def]
        """Patched get_mse_order_per_int8 that also records MSE values."""
        inner_output = None

        nc_util.logger.info("[MSE_PATCH] wrapped_int8 invoked for mse_v2 sensitivity capture.")

        def output_hook(self, input, output):  # type: ignore[no-untyped-def]
            nonlocal inner_output
            inner_output = output
            return output

        op_type_dict = {}
        for k, v in tune_cfg["op"].keys():
            op_type_dict[k] = v

        example_inp = example_input

        from neural_compressor.adaptor.pytorch import _cfg_to_qconfig  # type: ignore[import]

        op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
        module = nc_util.fetch_module(fp32_model, list(op_cfgs.keys())[-1])
        module.register_forward_hook(output_hook)
        nc_util.simple_inference(fp32_model, example_inp)
        inner_output_fp32 = inner_output

        quant_list = []
        for k, v in tune_cfg["op"].items():
            if k[1] in ["LayerNorm", "Dropout", "InstanceNorm3d"]:
                continue
            if v["weight"]["dtype"] == "fp32":
                quant_list.append(k)

        max_ops_env = os.environ.get("INC_MSE_MAX_OPS", "")
        try:
            max_ops = int(max_ops_env) if max_ops_env else 0
        except ValueError:
            max_ops = 0

        fallback_order: Dict[Tuple[str, str], float] = {}
        nc_util.logger.info("Evaluate the sensitivity for each fp32 operation")
        for idx, (op_name, op_type) in enumerate(nc_util.tqdm(quant_list)):
            if max_ops > 0 and idx >= max_ops:
                break
            if op_name in nc_util.op_cfg_mapping:
                tmp_model = copy.deepcopy(fp32_model)
                from neural_compressor.adaptor.pytorch import (  # type: ignore[import]
                    PyTorch_FXAdaptor,
                    _cfg_to_qconfig,
                    _cfgs_to_fx_cfgs,
                )

                op_cfgs[op_name] = nc_util.op_cfg_mapping[op_name]
                fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])

                from torch.quantization.quantize_fx import convert_fx, prepare_fx

                if adaptor.sub_module_list is None:
                    if adaptor.version.release >= nc_util.Version("1.13.0").release:  # type: ignore[attr-defined]
                        tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
                    else:
                        tmp_model = prepare_fx(tmp_model, fx_op_cfgs)
                else:
                    PyTorch_FXAdaptor.prepare_sub_graph(
                        adaptor.sub_module_list, fx_op_cfgs, tmp_model, prefix=""
                    )
                nc_util.simple_inference(tmp_model, example_inp)
                if adaptor.sub_module_list is None:
                    tmp_model = convert_fx(tmp_model)
                else:
                    PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, tmp_model, prefix="")

                module = nc_util.fetch_module(tmp_model, list(op_cfgs.keys())[-1])
                module.register_forward_hook(output_hook)
                nc_util.simple_inference(tmp_model, example_inp)
                inner_output_int8 = inner_output
                if inner_output_fp32.dtype == torch.quint8:
                    inner_output_fp32_local = inner_output_fp32.dequantize()
                else:
                    inner_output_fp32_local = inner_output_fp32
                if inner_output_int8.dtype == torch.quint8:
                    inner_output_int8 = inner_output_int8.dequantize()

                mse_val = (inner_output_fp32_local - inner_output_int8).pow(2).sum()
                key = (op_name, op_type_dict[op_name])
                fallback_order[key] = float(mse_val.item())
                int8_mse[key] = float(mse_val.item())

        ordered_ops = sorted(fallback_order.keys(), key=lambda key: fallback_order[key], reverse=False)
        return ordered_ops

    try:
        nc_util.get_mse_order_per_fp32 = wrapped_fp32  # type: ignore[assignment]
        nc_util.get_mse_order_per_int8 = wrapped_int8  # type: ignore[assignment]
        yield fp32_mse, int8_mse
    finally:
        nc_util.get_mse_order_per_fp32 = original_fp32  # type: ignore[assignment]
        nc_util.get_mse_order_per_int8 = original_int8  # type: ignore[assignment]


def run_single_mse_v2_sensitivity_pass(
    model,  # torch.nn.Module, but kept untyped to avoid hard torch dependency here
    conf,
    calib_dataloader,
    confidence_batches: int = 1,
) -> Tuple[MseMap, MseMap]:
    """Run a single bounded MSE_V2 sensitivity pass via the PyTorch FX adaptor.

    This helper wraps the common pattern needed to force INC to execute its
    per-op MSE helpers (``get_mse_order_per_fp32`` / ``get_mse_order_per_int8``)
    at least once, regardless of whether ``quantization.fit`` finds a good
    quantized model.

    It:
    - Wraps the HF model in an INC ``Model`` using the provided PTQ config.
    - Constructs an ``MSE_V2TuneStrategy`` with the calibration dataloader.
    - Initializes the adaptor, capability, and tuning space.
    - Builds an initial tuning config and passes it through
      ``adaptor.calculate_op_sensitivity(...)`` for both fallback and
      re-quant phases under ``capture_mse_v2_sensitivity()``, so the
      monkeypatched helpers can populate MSE maps.

    Returns:
        (fp32_mse_map, int8_mse_map): dictionaries mapping (op_name, op_type)
        to MSE values for fallback and re-quant sensitivity, respectively.
    """
    # Local imports to avoid imposing these dependencies on importers that
    # only need the monkeypatch context manager.
    from copy import deepcopy

    from neural_compressor.model import Model as NcModel  # type: ignore[import]
    from neural_compressor.quantization import _Config as NcInternalConfig  # type: ignore[import]
    from neural_compressor.strategy.mse_v2 import (  # type: ignore[import]
        MSE_V2TuneStrategy,
    )

    internal_conf = NcInternalConfig(
        quantization=conf,
        benchmark=None,
        pruning=None,
        distillation=None,
        nas=None,
    )
    wrapped_model = NcModel(model, conf=conf)

    with capture_mse_v2_sensitivity() as (fp32_mse_map, int8_mse_map):
        strategy = MSE_V2TuneStrategy(
            model=wrapped_model,
            conf=internal_conf,
            q_dataloader=calib_dataloader,
            q_func=None,
            eval_func=None,
            eval_dataloader=None,
            eval_metric=None,
            resume=None,
            q_hooks=None,
        )
        # Initialize adaptor, capability, tuning space, and (if needed) baseline.
        strategy._prepare_tuning()

        # Build a single initial tuning configuration covering all quantizable ops.
        _, _, initial_op_tuning_cfg = strategy.initial_tuning_cfg()
        base_tune_cfg = strategy._tune_cfg_converter(initial_op_tuning_cfg)

        # Fallback phase: treat INT8 ops as candidates for FP32 fallback.
        _ = strategy.adaptor.calculate_op_sensitivity(
            model=strategy.model,
            dataloader=calib_dataloader,
            tune_cfg=deepcopy(base_tune_cfg),
            output_op_names=None,
            confidence_batches=confidence_batches,
            fallback=True,
        )

        # Re-quant phase: treat FP32 ops as candidates for re-quantization.
        _ = strategy.adaptor.calculate_op_sensitivity(
            model=strategy.model,
            dataloader=calib_dataloader,
            tune_cfg=deepcopy(base_tune_cfg),
            output_op_names=None,
            confidence_batches=confidence_batches,
            fallback=False,
            requantize_cfgs=deepcopy(base_tune_cfg["op"]),
        )

    return fp32_mse_map, int8_mse_map
