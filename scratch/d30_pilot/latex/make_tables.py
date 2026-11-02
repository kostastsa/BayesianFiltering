"""LaTeX of Tables I and III (Exp. A, wrapped bearing) from the Error summary.txt files of 'main.py errors'.
Writes table_I.tex and table_III.tex next to this script. Columns for each noise value: MSE, LPE, NEES, Lost.
Usage (repository root): python scratch/d30_pilot/latex/make_tables.py"""
import os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
E = ROOT + "/data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}_wrap/{f}/Error summary.txt"
FAM = [("L-GSF", "gsf/lgsf/ncomps_{M}_iters_1", [(1, "M=1 (EKF)"), (100, "M=100"), (1000, "M=1000")]),
       ("U-GSF", "gsf/ugsf/ncomps_{M}_iters_1_alpha_1.0_beta_0.0_kappa_0.0", [(1, "M=1 (UKF)"), (100, "M=100"), (1000, "M=1000")]),
       ("BPF", "pf/bpf/numprt_{M}_ess_0.5", [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000"), (100000, "M=100000")]),
       ("APF", "pf/apf/numprt_{M}_ess_0.5", [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000"), (100000, "M=100000")]),
       ("RPF", "pf/rpf/numprt_{M}_ess_0.5", [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000")]),
       ("UPF", "pf/upf/numprt_{M}_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0", [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000")]),
       ("L-AGSF", "agsf/lagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [(2, "M=2, N=5, L=5"), (10, "M=10, N=5, L=5"), (100, "M=100, N=5, L=5")]),
       ("U-AGSF", "agsf/uagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [(2, "M=2, N=5, L=5"), (10, "M=10, N=5, L=5"), (100, "M=100, N=5, L=5")])]
SET = [("2.5", r"25\times 10^{-1}"), ("0.025", r"25\times 10^{-3}"), ("2.5e-05", r"25\times 10^{-6}")]


def summ(p):
    s = open(p).read(); o = {}
    o["n"] = int(re.search(r"^trials: (\d+)", s, re.M).group(1))
    o["lost"] = int(re.search(r"trials with mse > 10: (\d+)", s).group(1))
    for k in ("mse", "lpe", "outside"):
        m = re.search(rf"^{k}: (\S+) \+- (\S+), (\S+)", s, re.M); o[k] = (float(m.group(1)), float(m.group(2)))
    o["empty"] = int(re.search(r"trials with > 20 % of steps outside: (\d+) of", s).group(1))
    return o


def sci(x):
    return x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-2)


def pm(m, sd):
    """mean \pm sd; the sd uses the a x 10^b form also when the mean uses it (as in the paper)."""
    return f"{num(m)}\\pm{num(sd, force=sci(m) and abs(sd) >= 1e-2)}"


def num(x, force=False):
    ax = abs(x)
    if x != 0 and (force or ax >= 1e4 or ax < 1e-2):
        e = int(f"{x:.2e}".split("e")[1]); m = x / 10 ** e
        return rf"{m:.2f}\times 10^{{{e}}}"
    return f"{x:.2f}"


def table(a, label, appendix_ref):
    L = []
    L.append(r"\begin{table*}")
    L.append(r"\centering")
    L.append(r"\caption{{\textbf{{Experiment A}.} Comparison between the MSE and LPE for various settings of the L-GSF, U-GSF, BPF, \cblue{APF, RPF and UPF} with the proposed L-AGSF and U-AGSF algorithms. We compare the error metrics for $a=" + a + r"$ and three observation noise values. Lower values are better. "
             r"\cblue{MSE and LPE: mean $\pm$ the standard deviation of $100$ bootstrap means over $20$ (GSF, AGSF) or $30$ (BPF, APF, RPF, UPF) simulations. "
             r"NEES: percentage of time steps at which the NEES exceeds the $0.999$ quantile of the $\chi^2_4$ distribution, i.e., at which the true state is outside the $99.9\%$ region of the filter; in parentheses, the number of simulations with more than $20\%$ of such steps. "
             r"Lost: number of simulations with an MSE above $10$. A few simulations in which the filter loses the target dominate the means " + appendix_ref + r".}}}")
    L.append(r"\label{" + label + "}")
    L.append(r"\resizebox{\textwidth}{!}{")
    L.append(r"\begin{tabular}{ll" + "c" * 12 + "}")
    L.append(r"\toprule")
    L.append("& & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$a={a}, \sigma^2={tex}$}}" for _, tex in SET) + r" \\")
    L.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}")
    L.append(r"& \textsc{parameters} & " + " & ".join(["MSE & LPE & \cblue{NEES} & \cblue{Lost}"] * 3) + r" \\")
    for name, folder, sizes in FAM:
        L.append(r"\midrule")
        for j, (M, lab) in enumerate(sizes):
            head = rf"\multirow{{{len(sizes)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{name}}}}}" if j == 0 else ""
            cells = []
            for s, _ in SET:
                v = summ(E.format(a=a, s=s, f=folder.format(M=M)))
                cells += [rf"\cblue{{${pm(*v['mse'])}$}}",
                          rf"\cblue{{${pm(*v['lpe'])}$}}",
                          rf"\cblue{{${100 * v['outside'][0]:.1f}\%$ ({v['empty']})}}",
                          rf"\cblue{{${v['lost']}/{v['n']}$}}"]
            L.append(f"{head} & ${lab}$ & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "table_I.tex"), "w").write(table("0.5", "table::Exp A.1", "(see Appendix D of the supplementary material)"))
open(os.path.join(here, "table_III.tex"), "w").write(table("0.05", "table::Exp A.2", "(see Appendix D)"))
print("written table_I.tex and table_III.tex")
