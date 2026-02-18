# QSC-EOM Examples

Run the H2 reference workflow:

```bash
python qsceom/examples/H2/run_h2.py --ground-method uccsd --basis sto-3g
```

Use ADAPT-VQE for the ground state:

```bash
python qsceom/examples/H2/run_h2.py --ground-method adaptvqe --basis sto-6g
```

Optional controls:
- `--bond-length` sets H-H distance in Angstrom (default `0.735`)
- `--ground-shots` and `--excited-shots` set shot-based mode (`0` means analytic)
- `--uccsd-max-iter`, `--adapt-it`, and `--adapt-optimizer-maxiter` control optimization effort
