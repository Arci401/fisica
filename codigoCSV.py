from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

COL_TEMP = "CPU (Tctl/Tdie) [°C]"
COL_POTENCIA = "Potencia total de CPU [W]"
COL_HTC = "Desaceleración térmica (HTC) [Yes/No]"



def cargar_columnas_clave(ruta_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrae tiempo relativo (s), temperatura, potencia y flag HTC desde las 3 columnas clave."""
    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    t_s = np.arange(len(df), dtype=float)
    temp_c = df[COL_TEMP].to_numpy(dtype=float)
    power_w = df[COL_POTENCIA].to_numpy(dtype=float)
    htc_flag = (
        df[COL_HTC]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .to_numpy(dtype=int)
    )
    return t_s, temp_c, power_w, htc_flag


def promedio_en_ventana(t_s: np.ndarray, señal: np.ndarray, t_ini: float, t_fin: float) -> float:
    """Promedio de una señal en [t_ini, t_fin]."""
    mascara = (t_s >= t_ini) & (t_s <= t_fin)
    return float(np.mean(señal[mascara]))


def r_termica(t_carga: float, t_reposo: float, p_carga: float, p_reposo: float) -> float:
    """R_th = (T_carga - T_reposo) / (P_carga - P_reposo)."""
    return (t_carga - t_reposo) / (p_carga - p_reposo)


def curva_exponencial(t_rel: np.ndarray, t_eq: float, t_ini: float, tau: float) -> np.ndarray:
    """T(t_rel) = T_eq + (T_ini - T_eq) * exp(-t_rel / tau)."""
    return t_eq + (t_ini - t_eq) * np.exp(-t_rel / tau)


def ajustar_tau(t_rel: np.ndarray, temp_tramo: np.ndarray, tau_semilla: float = 30.0) -> tuple[float, float, float]:
    """Ajusta tau, T_ini y T_eq sobre un tramo."""
    popt, _ = curve_fit(
        lambda x, t_eq, t_ini, tau: curva_exponencial(x, t_eq, t_ini, tau),
        t_rel,
        temp_tramo,
        p0=[float(temp_tramo[-1]), float(temp_tramo[0]), tau_semilla],
        bounds=([0.0, 0.0, 0.1], [130.0, 130.0, 3000.0]),
        maxfev=20000,
    )
    return float(popt[2]), float(popt[1]), float(popt[0])


def c_termica(tau: float, r_th: float) -> float:
    """C_th = tau / R_th."""
    return tau / r_th


def graficar_sensores(
    t_s: np.ndarray,
    temp_c: np.ndarray,
    power_w: np.ndarray,
    t_cal: float,
    t_enf: float,
    salida: Path,
) -> None:
    """Grafica temperatura y potencia total de CPU en el tiempo."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t_s, temp_c, color="#1f77b4", lw=1.4, label=COL_TEMP)
    ax1.axvline(t_cal, color="#d62728", ls="--", lw=1.1, label=f"Calentamiento ({int(t_cal)} s)")
    ax1.axvline(t_enf, color="#2ca02c", ls="--", lw=1.1, label=f"Enfriamiento ({int(t_enf)} s)")
    ax1.set_ylabel("Temperatura [°C]")
    ax1.set_title("Temperatura y Potencia Total de CPU")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_s, power_w, color="#ff7f0e", lw=1.2, label=COL_POTENCIA)
    ax2.axvline(t_cal, color="#d62728", ls="--", lw=1.1)
    ax2.axvline(t_enf, color="#2ca02c", ls="--", lw=1.1)
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Potencia [W]")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(salida, dpi=150)
    plt.close(fig)


def main() -> None:
    ruta_csv = Path("piloto.CSV")
    t_cal = float(3 * 60 + 11)   # 191 s
    t_enf = float(18 * 60 + 8)  # 1088 s

    t_s, temp_c, power_w, htc_flag = cargar_columnas_clave(ruta_csv)

    t_rep = promedio_en_ventana(t_s, temp_c, t_cal - 60.0, t_cal - 1.0)
    p_rep = promedio_en_ventana(t_s, power_w, t_cal - 60.0, t_cal - 1.0)
    t_car = promedio_en_ventana(t_s, temp_c, t_enf - 60.0, t_enf - 1.0)
    p_car = promedio_en_ventana(t_s, power_w, t_enf - 60.0, t_enf - 1.0)

    r_th = r_termica(t_car, t_rep, p_car, p_rep)

    mascara_enf = t_s >= t_enf
    tau_enf, _, _ = ajustar_tau(t_s[mascara_enf] - t_enf, temp_c[mascara_enf], tau_semilla=30.0)
    c_th = c_termica(tau_enf, r_th)

    graficar_sensores(t_s, temp_c, power_w, t_cal, t_enf, Path("sensores_cpu.png"))

    print(f"Eventos HTC activos: {int(np.sum(htc_flag))}")
    print(f"R_th = {r_th:.4f} °C/W | tau_enf = {tau_enf:.2f} s | C_th = {c_th:.2f} J/°C")


if __name__ == "__main__":
    main()
    
    
"""
Eventos HTC activos: 0
R_th = 1.5399 °C/W | tau_enf = 48.30 s | C_th = 31.36 J/°C
"""