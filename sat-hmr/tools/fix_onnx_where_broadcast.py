#!/usr/bin/env python3
"""Patch SAT-HMR ONNX graph for runtime compatibility.

Passes:
1) Fix Where broadcast mismatch (cond [N] with y [N, ...])
2) Rewrite the problematic bool Where pattern to arithmetic ops (no Where kernel needed)
3) Replace Split nodes with single output by Identity (B=1 export path)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx import TensorProto


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    inferred = shape_inference.infer_shapes(model)
    out: dict[str, list[int | str]] = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        t = v.type.tensor_type
        if not t.HasField("shape"):
            continue
        dims: list[int | str] = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
        out[v.name] = dims
    return out


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = shape_inference.infer_shapes(model)
    out: dict[str, int] = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        t = v.type.tensor_type
        if not t.HasField("elem_type"):
            continue
        out[v.name] = int(t.elem_type)
    return out


def _rank(shape: list[int | str] | None) -> int:
    return -1 if shape is None else len(shape)


def patch_where_broadcast(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    shape_map = _shape_map(model)
    nodes = model.graph.node
    patched = 0

    # Append new initializers/inputs after graph traversal to avoid mutating while iterating.
    nodes_to_insert: list[tuple[int, onnx.NodeProto]] = []
    new_inits: list[onnx.TensorProto] = []

    for i, node in enumerate(nodes):
        if node.op_type != "Where" or len(node.input) != 3:
            continue

        cond_name, x_name, y_name = node.input
        cond_shape = shape_map.get(cond_name)
        x_shape = shape_map.get(x_name)
        y_shape = shape_map.get(y_name)
        cond_rank = _rank(cond_shape)
        x_rank = _rank(x_shape)
        y_rank = _rank(y_shape)
        target_rank = max(x_rank, y_rank)

        # Patch only the known problematic pattern:
        # cond: [N], x: scalar/[], y: [N, ...]  -> need cond: [N,1,...]
        if cond_rank != 1 or target_rank <= 1:
            continue
        if y_shape is None or len(y_shape) < 2:
            continue
        if cond_shape is None:
            continue
        if cond_shape[0] != y_shape[0]:
            continue

        axes = list(range(1, target_rank))
        axes_name = f"{node.name or 'Where'}_axes_const_{i}"
        unsq_out = f"{node.name or 'Where'}_cond_unsq_{i}"
        unsq_name = f"{node.name or 'Where'}_unsqueeze_{i}"

        new_inits.append(
            numpy_helper.from_array(
                np.array(axes, dtype="int64"),
                name=axes_name,
            )
        )
        unsq_node = helper.make_node(
            "Unsqueeze",
            inputs=[cond_name, axes_name],
            outputs=[unsq_out],
            name=unsq_name,
        )
        nodes_to_insert.append((i, unsq_node))
        node.input[0] = unsq_out
        patched += 1

    if patched == 0:
        return model, 0

    # Insert in reverse so indices remain valid.
    for idx, n in reversed(nodes_to_insert):
        nodes.insert(idx, n)
    model.graph.initializer.extend(new_inits)
    return model, patched


def rewrite_problematic_bool_where(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Rewrite bool Where(cond, scalar_bool, bool_tensor) to arithmetic ops.

    Some runtime stacks fail for this specific Where signature even after broadcast fix.
    Equivalent logic:
        out = cond ? x : y
    is rewritten as:
        out_i32 = cond_i32 * x_i32 + (1-cond_i32) * y_i32
        out = Cast(out_i32, BOOL)
    """
    shape_map = _shape_map(model)
    type_map = _type_map(model)

    one_name = "__fix_where_one_i32"
    if not any(init.name == one_name for init in model.graph.initializer):
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(1, dtype=np.int32), name=one_name)
        )

    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Where" or len(node.input) != 3:
            new_nodes.append(node)
            continue

        cond_name, x_name, y_name = node.input
        cond_shape = shape_map.get(cond_name)
        x_shape = shape_map.get(x_name)
        y_shape = shape_map.get(y_name)

        cond_type = type_map.get(cond_name)
        x_type = type_map.get(x_name)
        y_type = type_map.get(y_name)

        x_rank = _rank(x_shape)
        y_rank = _rank(y_shape)

        # SAT-HMR problematic pattern:
        # cond: bool [N,1] (after pass1), x: bool scalar, y: bool [N,M]
        if not (
            cond_type == TensorProto.BOOL
            and x_type == TensorProto.BOOL
            and y_type == TensorProto.BOOL
            and x_rank == 0
            and y_rank >= 2
        ):
            new_nodes.append(node)
            continue

        base = node.name or "Where"
        ccond = f"{base}_cond_i32"
        cx = f"{base}_x_i32"
        cy = f"{base}_y_i32"
        notc = f"{base}_notc_i32"
        a = f"{base}_a_i32"
        b = f"{base}_b_i32"
        s = f"{base}_s_i32"
        out = node.output[0]

        new_nodes.extend(
            [
                helper.make_node("Cast", [cond_name], [ccond], name=f"{base}/CastCond", to=TensorProto.INT32),
                helper.make_node("Cast", [x_name], [cx], name=f"{base}/CastX", to=TensorProto.INT32),
                helper.make_node("Cast", [y_name], [cy], name=f"{base}/CastY", to=TensorProto.INT32),
                helper.make_node("Sub", [one_name, ccond], [notc], name=f"{base}/NotCond"),
                helper.make_node("Mul", [ccond, cx], [a], name=f"{base}/MulA"),
                helper.make_node("Mul", [notc, cy], [b], name=f"{base}/MulB"),
                helper.make_node("Add", [a, b], [s], name=f"{base}/Add"),
                helper.make_node("Cast", [s], [out], name=f"{base}/CastOut", to=TensorProto.BOOL),
            ]
        )
        rewritten += 1

    if rewritten == 0:
        return model, 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, rewritten


def rewrite_single_output_split(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Split with one output is a no-op for B=1 export path; replace by Identity."""
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0
    for node in model.graph.node:
        if node.op_type == "Split" and len(node.output) == 1:
            new_nodes.append(
                helper.make_node(
                    "Identity",
                    [node.input[0]],
                    [node.output[0]],
                    name=f"{node.name or 'Split'}/Identity",
                )
            )
            rewritten += 1
            continue
        new_nodes.append(node)

    if rewritten == 0:
        return model, 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, rewritten


def rewrite_const_single_index_gather(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Rewrite Gather(data, const_single_index) into Slice (+ Squeeze for scalar index).

    This avoids relying on Gather kernels for trivial index patterns and helps stricter
    TensorRT precision flows.
    """
    const_values: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        const_values[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                const_values[node.output[0]] = numpy_helper.to_array(attr.t)
                break

    new_inits: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0

    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Gather" or len(node.input) < 2:
            new_nodes.append(node)
            continue

        index_name = node.input[1]
        if index_name not in const_values:
            new_nodes.append(node)
            continue

        index_arr = np.asarray(const_values[index_name])
        flat = index_arr.reshape(-1)
        if flat.size != 1:
            new_nodes.append(node)
            continue

        gather_axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                gather_axis = int(attr.i)
                break

        index_val = int(flat[0])
        if index_val >= 0:
            end_val = index_val + 1
        else:
            # Keep ONNX negative-index behavior for last elements.
            end_val = np.iinfo(np.int64).max

        base = node.name or f"Gather_{idx}"
        starts_name = f"__gather2slice_starts_{idx}"
        ends_name = f"__gather2slice_ends_{idx}"
        axes_name = f"__gather2slice_axes_{idx}"
        steps_name = f"__gather2slice_steps_{idx}"
        new_inits.extend(
            [
                numpy_helper.from_array(np.asarray([index_val], dtype=np.int64), name=starts_name),
                numpy_helper.from_array(np.asarray([end_val], dtype=np.int64), name=ends_name),
                numpy_helper.from_array(np.asarray([gather_axis], dtype=np.int64), name=axes_name),
                numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=steps_name),
            ]
        )

        # Scalar index (shape=[]) reduces rank by 1 in Gather, so append Squeeze after Slice.
        if index_arr.ndim == 0:
            slice_out = f"{node.output[0]}__gather2slice"
            squeeze_axes_name = f"__gather2slice_squeeze_axes_{idx}"
            new_inits.append(
                numpy_helper.from_array(np.asarray([gather_axis], dtype=np.int64), name=squeeze_axes_name)
            )
            new_nodes.append(
                helper.make_node(
                    "Slice",
                    inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                    outputs=[slice_out],
                    name=f"{base}/Slice",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[slice_out, squeeze_axes_name],
                    outputs=[node.output[0]],
                    name=f"{base}/Squeeze",
                )
            )
        else:
            new_nodes.append(
                helper.make_node(
                    "Slice",
                    inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                    outputs=[node.output[0]],
                    name=f"{base}/Slice",
                )
            )
        rewritten += 1

    if rewritten == 0:
        return model, 0

    model.graph.initializer.extend(new_inits)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, rewritten


def rewrite_redundant_cast(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace Cast nodes whose input/output elem type is identical with Identity."""
    type_map = _type_map(model)
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            new_nodes.append(node)
            continue
        in_name = node.input[0]
        out_name = node.output[0]
        in_type = type_map.get(in_name)
        out_type = type_map.get(out_name)
        if in_type is None or out_type is None or in_type != out_type:
            new_nodes.append(node)
            continue
        new_nodes.append(
            helper.make_node(
                "Identity",
                inputs=[in_name],
                outputs=[out_name],
                name=f"{node.name or f'Cast_{idx}'}/Identity",
            )
        )
        rewritten += 1

    if rewritten == 0:
        return model, 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, rewritten


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix ONNX Where broadcast for TensorRT parser.")
    p.add_argument("--in", dest="in_path", required=True, help="Input ONNX path")
    p.add_argument("--out", dest="out_path", required=True, help="Output ONNX path")
    p.add_argument(
        "--skip-bool-where-rewrite",
        action="store_true",
        help="Skip rewriting problematic bool Where to arithmetic ops.",
    )
    p.add_argument(
        "--skip-single-split-rewrite",
        action="store_true",
        help="Skip replacing one-output Split with Identity.",
    )
    p.add_argument(
        "--skip-const-gather-rewrite",
        action="store_true",
        help="Skip rewriting Gather(data, const_single_index) to Slice/Squeeze.",
    )
    p.add_argument(
        "--skip-redundant-cast-rewrite",
        action="store_true",
        help="Skip replacing redundant Cast nodes (same in/out type) with Identity.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_path))
    model, n_where_broadcast = patch_where_broadcast(model)
    n_bool_where = 0
    n_split = 0
    if not args.skip_bool_where_rewrite:
        model, n_bool_where = rewrite_problematic_bool_where(model)
    if not args.skip_single_split_rewrite:
        model, n_split = rewrite_single_output_split(model)
    n_const_gather = 0
    if not args.skip_const_gather_rewrite:
        model, n_const_gather = rewrite_const_single_index_gather(model)
    n_redundant_cast = 0
    if not args.skip_redundant_cast_rewrite:
        model, n_redundant_cast = rewrite_redundant_cast(model)

    # Validate quickly via ONNX checker + shape inference.
    onnx.checker.check_model(model)
    _ = shape_inference.infer_shapes(model)
    onnx.save(model, str(out_path))
    print(
            "[DONE] Fix summary: "
            f"where_broadcast={n_where_broadcast}, "
            f"bool_where_rewrite={n_bool_where}, "
            f"split1_rewrite={n_split}, "
            f"const_gather_rewrite={n_const_gather}, "
            f"redundant_cast_rewrite={n_redundant_cast}"
    )
    print(f"[DONE] Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
