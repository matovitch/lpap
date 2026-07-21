import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import subprocess
    import sys

    import marimo as mo
    import torch

    install_spec = (
        "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
    )

    try:
        import lpap
        install_note = "lpap already importable"
    except ImportError:
        # --no-deps: use molab's preinstalled torch; avoid resolver fights.
        # jaxtyping is required by lpap but is not a molab default.
        lpap_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            install_spec,
        ]
        deps_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "jaxtyping>=0.3.7",
        ]
        subprocess.check_call(lpap_cmd)
        subprocess.check_call(deps_cmd)
        import lpap

        install_note = (
            f"installed via: {' '.join(lpap_cmd)} ; {' '.join(deps_cmd)}"
        )

    return install_note, install_spec, lpap, mo, torch


@app.cell
def _(mo):
    mo.md("""
    # LPAP molab POC

    Smoke test for remote GPU + package install from `molab-summer`.
    Attach an **RTX Pro 6000**, run all cells, then pair Cursor via
    **Actions → Pair with an agent**.
    """)
    return


@app.cell
def _(install_note, mo, torch):
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
    gpu_md = mo.md(
        f"""
## GPU smoke

- install: `{install_note}`
- torch: `{torch.__version__}`
- cuda available: **{cuda_ok}**
- device: `{device}` (`{device_name}`)
"""
    )
    gpu_md
    return cuda_ok, device, device_name


@app.cell
def _(cuda_ok, device, lpap, mo, torch):
    batch = 4
    n = 256
    buckets = 32
    k_max = 2
    values = torch.randn(batch, n, device=device, dtype=torch.float32)
    amplitudes, dibs, soft = lpap.lpap_torch(
        values, bucket_count=buckets, k_max=k_max
    )
    amp_sum = float(amplitudes.abs().sum().item())
    finite = bool(
        torch.isfinite(amplitudes).all()
        and torch.isfinite(soft).all()
    )
    lpap_md = mo.md(
        f"""
## LPAP smoke (`lpap_torch`)

- shapes: amplitudes `{tuple(amplitudes.shape)}`, dibs `{tuple(dibs.shape)}`, soft `{tuple(soft.shape)}`
- |amplitudes| sum: `{amp_sum:.4f}`
- all finite: **{finite}**
- ran on CUDA: **{cuda_ok and values.is_cuda}**
"""
    )
    lpap_md
    return amp_sum, amplitudes, dibs, finite, soft, values



@app.cell
def _(amp_sum, cuda_ok, device_name, finite, mo):
    ok = cuda_ok and finite and amp_sum > 0.0
    status = "PASS" if ok else "FAIL"
    mo.md(
        f"""
## Status: **{status}**

Checklist:

- [x] `lpap` importable from git `@molab-summer`
- [{"x" if cuda_ok else " "}] CUDA available (`{device_name}`)
- [{"x" if finite else " "}] `lpap_torch` returned finite tensors
- [{"x" if amp_sum > 0 else " "}] non-trivial amplitude mass

Next: Actions → **Pair with an agent**, then ask Cursor to re-run this notebook.
"""
    )
    return (ok,)


if __name__ == "__main__":
    app.run()
