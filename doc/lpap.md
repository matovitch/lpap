# Linear Probing Amplitude Pooling

Linear Probing Amplitude Pooling, or LPAP, is a pooling operator for reducing a flat tensor of `N` values into a compact table of `C` buckets, where `N` is a multiple of `C`. The operator also receives a maximum roll count, `k_max`, which bounds how far the probing process may advance.

The operator keeps two small table tensors. The table is always semantically full, even when it contains zeros:

- `buckets`: `C` pooled values selected by largest amplitude.
- `dibs`: `C` integer distances to the selected values' initial buckets.

It also uses an integer work tensor shaped like the projected input view. This tensor stores per-position offsets, called `dibs_diff` here, that are later combined with a roll counter to recover the distance information for values that are displaced and then re-enter the table.

In practice, LPAP is intended to run batched. A single example has input shape `N`, bucket shape `C`, and dib shape `C`; a batched call can be understood as applying that same operator independently across a leading batch dimension, like the result of `vmap`:

```text
values:  B x N
buckets: B x C
dibs:    B x C
```

Additional leading dimensions can be treated the same way, as long as each independent item still satisfies `N % C == 0`.

## Tensor View

Given an input tensor with `N` elements and a bucket count `C`, LPAP views the input as:

```text
C x (N // C)
```

Each row corresponds to one initial bucket. Each column corresponds to one probe position for that bucket.

For example, if `C = 4` and `N = 16`, the input is interpreted as four bucket lanes with four probe positions each.

## Projection Step

At each projection step, LPAP compares values by amplitude, using `abs(value)`, along the probe dimension of each bucket lane.

For every bucket lane:

1. Find the index of the value with the largest absolute amplitude.
2. Compare that value against the current table value for the corresponding bucket.
3. Use the integer work tensor at the selected position as that value's local displacement.

The integer work tensor is initialized with zeros. When a table value is swapped back into the projected view, the work tensor stores the adjusted distance state needed to reconstruct that displaced value's DIB on a later projection.

## Collision And Swap-Back

The table always contains values. There is no empty-bucket or full-bucket state: a table initialized with zeros is still full of zero values.

When a projected value has amplitude greater than or equal to the current table value, LPAP swaps the existing table content back into the input tensor in place. This preserves displaced values so they can be considered by later projection steps.

The swap-back updates both value and distance state:

- The old bucket value is written back into the input tensor.
- Its associated distance information is preserved through the integer work tensor.
- If the displaced value later returns to the table, its final dib can be reconstructed.

## Rolling And DIB Computation

After each projection attempt, LPAP advances the probing process by rolling the projected tensors by one position and incrementing a roll counter `k`. The process stops when `k` reaches `k_max`.

Conceptually:

```text
k = k + 1
roll(value_view, shift=1)
roll(dibs_diff_view, shift=1)
```

When a value is projected into a bucket after rolling, its distance to its initial bucket is computed as:

```text
dib = k + dibs_diff
```

Here:

- `k` counts how many probing rolls have occurred.
- `dibs_diff` is the local offset stored in the integer work tensor.

Together, they describe how far the value has traveled from its initial bucket before landing in the table.

## Equivalent Rolling Strategy

Instead of rolling the larger `C x (N // C)` projected views, an implementation may roll the smaller `buckets` and `dibs` tensors. The two strategies describe the same probing process, but rolling the smaller tensors may be preferable for an optimized implementation.

## High-Level Algorithm

```text
input: values with N elements, bucket count C, maximum roll count k_max
require: N % C == 0

view values as C x (N // C)
initialize buckets as zeros with shape C
initialize dibs as zeros with shape C
initialize dibs_diff as zeros with shape C x (N // C)
initialize roll counter k = 0

repeat while k < k_max:
    for each bucket lane:
        index = argmax(abs(values in lane))
        candidate = values[lane, index]
        candidate_dib = k + dibs_diff[lane, index]

        if abs(candidate) >= abs(current bucket value):
            swap current bucket value and dib back into values in place
            preserve its adjusted distance in dibs_diff
            write candidate into target bucket
            write candidate_dib into target dib

    roll probing state by one position
    k = k + 1

output: buckets, dibs, modified input values
```

## Implementation Notes

- Amplitude comparison uses absolute value, but the original signed value is preserved in the bucket.
- The input tensor is modified in place during swap-back.
- There are no empty table entries. Zero is a valid table value.
- The integer `dibs_diff` tensor is working state, not the final dib output.
- The final `dibs` tensor records distances to initial buckets for values that land in the table.
- `k_max` bounds the probing work and prevents unbounded rolling when values keep being displaced.
- Batched execution should apply the same logic independently for each batch item.
- A GPU implementation should avoid materializing expensive rolls when index arithmetic or rolling the smaller table state is enough.
