# Policy exports

`python -m r2p.policy.train` writes one directory per policy with `config.json`,
`norm.json`, `weights.npz`, and standalone NumPy `runtime.py`. Training records
closed-loop clean, real, and realx0.5 evaluations before returning. The config
declares the start pose, task, observation layout, constant clamps, seed ensemble,
absolute action convention and weight hash. Use only a measured/gated candidate
for deployment; the automated tiny CPU tests create temporary smoke exports.
