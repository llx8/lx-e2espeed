"""Phase 0: ground-truth the decomposed attention subgraphs in the ColorAdjustor ONNX.

Finds every Softmax whose input is a MatMul (QK) and whose single consumer is a
MatMul (AV), then prints operand shapes, the scale node (Mul/Div) feeding QK, and
the transpose layout of K. This fixes the IO contract for the TRT plugin.
"""
import argparse
from pathlib import Path

import onnx
from onnx import shape_inference

def shape_of(vi):
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    return dims


def main():
    parser = argparse.ArgumentParser(description="Inspect decomposed attention subgraphs in ONNX.")
    parser.add_argument("onnx")
    args = parser.parse_args()
    path = args.onnx
    print(f"Loading {path}")
    model = onnx.load(path)
    model = shape_inference.infer_shapes(model)
    g = model.graph

    vinfo = {vi.name: vi for vi in list(g.value_info) + list(g.input) + list(g.output)}

    def vshape(name):
        return shape_of(vinfo[name]) if name in vinfo else "?"

    producer = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    triplets = []
    for sm in g.node:
        if sm.op_type != "Softmax" or not sm.input or not sm.output:
            continue
        qk = producer.get(sm.input[0])
        cons = consumers.get(sm.output[0], [])
        if qk is None or qk.op_type != "MatMul":
            continue
        if len(cons) != 1 or cons[0].op_type != "MatMul":
            continue
        triplets.append((qk, sm, cons[0]))

    print(f"\nFound {len(triplets)} decomposed attention triplets\n")
    for idx, (qk, sm, av) in enumerate(triplets):
        print(f"===== attn[{idx}] =====")
        print(f"  QK MatMul {qk.name}")
        for i, inp in enumerate(qk.input):
            prod = producer.get(inp)
            print(f"    in[{i}] {inp}  shape={vshape(inp)}  <- {prod.op_type if prod else 'graph-input/init'}")
        print(f"    out    {qk.output[0]}  shape={vshape(qk.output[0])}")
        print(f"  Softmax {sm.name} axis={[a.i for a in sm.attribute if a.name=='axis']}")
        print(f"    out    {sm.output[0]}  shape={vshape(sm.output[0])}")
        print(f"  AV MatMul {av.name}")
        for i, inp in enumerate(av.input):
            prod = producer.get(inp)
            print(f"    in[{i}] {inp}  shape={vshape(inp)}  <- {prod.op_type if prod else 'graph-input/init'}")
        print(f"    out    {av.output[0]}  shape={vshape(av.output[0])}")

        # Trace scale: walk back from QK inputs through Mul/Div/Transpose/Reshape.
        for i, inp in enumerate(qk.input):
            chain = []
            cur = inp
            for _ in range(8):
                p = producer.get(cur)
                if p is None:
                    chain.append(f"{cur}(input/init)")
                    break
                chain.append(f"{p.op_type}[{p.name}]")
                if p.op_type in ("Mul", "Div"):
                    # report the scalar operand if it's an initializer/constant
                    chain[-1] += f" inputs={list(p.input)}"
                cur = p.input[0] if p.input else None
                if cur is None:
                    break
            print(f"  QK in[{i}] backtrace: " + " <- ".join(chain))
        print()

    # Dump constant/initializer values that look like scales (scalars near attention).
    print("===== scalar initializers (potential sm_scale) =====")
    from onnx import numpy_helper
    seen = set()
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size == 1:
            v = float(arr.reshape(-1)[0])
            key = round(v, 6)
            if key not in seen and (0.0 < abs(v) < 2.0):
                seen.add(key)
                print(f"  {init.name} = {v}")


if __name__ == "__main__":
    main()
