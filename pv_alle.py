import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import matplotlib.animation as animation
import tempfile

st.set_page_config(page_title="Poissonverteilung – Multi-Tab", layout="centered")
st.title("Poissonverteilung – Mehrere Tabs")

# Gemeinsame Sidebar-Parameter, werden in allen Tabs verwendet!
st.sidebar.header("Gemeinsame Parameter")
lambda_val = st.sidebar.slider("λ (Erwartungswert)", min_value=0.1, max_value=100.0, value=12.0, step=0.1)
x_min = st.sidebar.number_input("x-Minimum", min_value=0, value=0, step=1)
x_max = st.sidebar.number_input("x-Maximum", min_value=1, value=30, step=1)
k = st.sidebar.number_input("k-Wert für P(X = k), P(X ≤ k), P(X ≥ k)", min_value=0, value=10, step=1)

tab_names = [
    "Dichtefunktion |",
    "Kumulative und Komplementäre kumulative |",
    "Normalapproximation |",
    "Animation λ"
]
tabs = st.tabs(tab_names)

# Tab 1: Grundlagen
with tabs[0]:
    st.header("Wahrscheinlichkeitsfunktion")
    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        x = np.arange(x_min, x_max + 1)
        y = poisson.pmf(x, mu=lambda_val)
        fig, ax = plt.subplots()
        ax.bar(x, y, color='skyblue', edgecolor='black')
        ax.set_title(f"Poisson-Verteilung (λ = {lambda_val})")
        ax.set_xlabel("x")
        ax.set_ylabel("Wahrscheinlichkeit P(X = x)")
        ax.grid(True, linestyle='--', alpha=0.5)
        if x_min <= k <= x_max:
            ax.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X={k})")
            ax.legend()
        st.pyplot(fig)
        st.subheader("Wahrscheinlichkeiten")
        st.dataframe({"x": x, f"P(X=x) bei λ={lambda_val}": np.round(y, 4)})
        st.subheader(f"Wahrscheinlichkeit für k = {k}")
        prob_k = poisson.pmf(k, mu=lambda_val)
        st.write(f"**P(X = {k}) = {prob_k:.4f}**")

# Tab 2: PMF, CDF, CCDF
with tabs[1]:
    st.header(" ")
    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        x = np.arange(x_min, x_max + 1)
        pmf_y = poisson.pmf(x, mu=lambda_val)
        cdf_y = poisson.cdf(x, mu=lambda_val)
        ccdf_y = 1 - cdf_y + pmf_y

        bar_colors = ['royalblue' if val <= k else 'skyblue' for val in x]
        fig1, ax1 = plt.subplots()
        ax1.bar(x, pmf_y, color=bar_colors, edgecolor='black')
        if x_min <= k <= x_max:
            ax1.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X = {k})")
            ax1.legend()
        ax1.set_title(f"Wahrscheinlichkeitsfunktion (genau) (λ = {lambda_val})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("P(X = x)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(x, cdf_y, marker='o', linestyle='-', color='green', label="P(X ≤ x)")
        ax2.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
        ax2.axhline(poisson.cdf(k, mu=lambda_val), color='gray', linestyle=':', label=f"P(X ≤ {k}) = {poisson.cdf(k, mu=lambda_val):.4f}")
        ax2.set_title("Kumulative Verteilung (P(X ≤ k))")
        ax2.set_xlabel("x")
        ax2.set_ylabel("P(X ≤ x)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(x, ccdf_y, marker='o', linestyle='-', color='purple', label="P(X ≥ x)")
        ax3.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
        ax3.axhline(ccdf_y[k - x_min] if x_min <= k <= x_max else 0, color='gray', linestyle=':', label=f"P(X ≥ {k}) = {ccdf_y[k - x_min]:.4f}")
        ax3.set_title("Komplementäre kumulative Verteilung (P(X ≥ k))")
        ax3.set_xlabel("x")
        ax3.set_ylabel("P(X ≥ x)")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend()
        st.pyplot(fig3)

        st.subheader("Wahrscheinlichkeiten für k")
        prob_k = poisson.pmf(k, mu=lambda_val)
        cdf_k = poisson.cdf(k, mu=lambda_val)
        ccdf_k = 1 - cdf_k + prob_k
        st.write(f"**P(X = {k}) = {prob_k:.4f}**")
        st.write(f"**P(X ≤ {k}) = {cdf_k:.4f}**")
        st.write(f"**P(X ≥ {k}) = {ccdf_k:.4f}**")

        st.subheader("Tabelle: Wahrscheinlichkeiten")
        table_combined = {
            "x": x,
            f"P(X = x) bei λ = {lambda_val}": np.round(pmf_y, 6),
            f"P(X ≤ x) bei λ = {lambda_val}": np.round(cdf_y, 6),
            f"P(X ≥ x) bei λ = {lambda_val}": np.round(ccdf_y, 6)
        }
        st.dataframe(table_combined)

# Tab 3: Neue Normalapproximation mit Stetigkeitskorrektur (dein Code)
with tabs[2]:
    st.header("Poissonverteilung & Normalapproximation")
    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        x = np.arange(x_min, x_max + 1)
        pmf_y = poisson.pmf(x, mu=lambda_val)
        cdf_y = poisson.cdf(x, mu=lambda_val)
        ccdf_y = 1 - cdf_y + pmf_y

        # Normalverteilungsparameter
        mu = lambda_val
        sigma = np.sqrt(lambda_val)
        norm_x = np.linspace(x_min, x_max, 500)
        norm_pdf = norm.pdf(norm_x, mu, sigma)
        norm_cdf = norm.cdf(norm_x, mu, sigma)
        norm_ccdf = 1 - norm_cdf

        bar_colors = ['royalblue' if val <= k else 'skyblue' for val in x]

        # Einzelwahrscheinlichkeit P(X = k) mit Stetigkeitskorrektur
        st.subheader(" Punktwahrscheinlichkeit $P(X = k)$")
        fig1, ax1 = plt.subplots()
        ax1.bar(x, pmf_y, color=bar_colors, edgecolor='black', label='Poisson')
        ax1.plot(norm_x, norm_pdf, 'r-', lw=2, label='Normalverteilung')
        if x_min <= k <= x_max:
            ax1.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X = {k})")
            # Stetigkeitskorrektur-Bereich visualisieren
            x_fill = np.linspace(k - 0.5, k + 0.5, 100)
            y_fill = norm.pdf(x_fill, mu, sigma)
            ax1.fill_between(x_fill, y_fill, color='orange', alpha=0.4, label='Stetigkeitskorrektur')
            ax1.legend()
        ax1.set_title(f"Wahrscheinlichkeitsfunktion und Normalapproximation (λ = {lambda_val})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("P(X = x)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

        # Berechnung und Anzeige der Wahrscheinlichkeiten mit Stetigkeitskorrektur
        p_pois = poisson.pmf(k, mu=lambda_val)
        p_norm_stetig = norm.cdf(k + 0.5, mu, sigma) - norm.cdf(k - 0.5, mu, sigma)
        st.latex(rf"P(X = {k}) = P({k} - 0.5 < Y < {k} + 0.5)")
        st.markdown(f"**Poisson:** P(X = {k}) = `{p_pois:.5f}`")
        st.markdown(f"**Normalapproximation mit Stetigkeitskorrektur:** P({k}-0.5 < Y < {k}+0.5) = `{p_norm_stetig:.5f}`")

        # Kumulative Wahrscheinlichkeit P(X ≤ k)
        st.subheader(" Kumulative Wahrscheinlichkeit $P(X \leq k)$ mit Stetigkeitskorrektur")
        fig2, ax2 = plt.subplots()
        ax2.plot(x, cdf_y, marker='o', linestyle='-', color='green', label="Poisson")
        ax2.plot(norm_x, norm_cdf, 'r-', lw=2, label='Normalverteilung')
        ax2.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
        ax2.set_title("Kumulative Verteilung $P(X \leq x)$")
        ax2.set_xlabel("x")
        ax2.set_ylabel("P(X ≤ x)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

        # Berechnung und Anzeige der Wahrscheinlichkeiten mit Stetigkeitskorrektur
        p_pois_leq = poisson.cdf(k, mu=lambda_val)
        p_norm_leq_stetig = norm.cdf(k + 0.5, mu, sigma)
        st.latex(rf"P(X \leq {k}) \approx P(Y < {k} + 0.5)")
        st.markdown(f"**Poisson:** P(X ≤ {k}) = `{p_pois_leq:.5f}`")
        st.markdown(f"**Normalapproximation mit Stetigkeitskorrektur:** P(Y < {k}+0.5) = `{p_norm_leq_stetig:.5f}`")

        # Komplementäre kumulative Wahrscheinlichkeit P(X ≥ k)
        st.subheader(" Komplementäre kumulative Wahrscheinlichkeit $P(X \geq k)$ mit Stetigkeitskorrektur")
        fig3, ax3 = plt.subplots()
        ax3.plot(x, ccdf_y, marker='o', linestyle='-', color='purple', label="Poisson")
        ax3.plot(norm_x, norm_ccdf, 'r-', lw=2, label='Normalverteilung')
        ax3.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
        ax3.set_title("Komplementäre kumulative Verteilung $P(X \geq x)$")
        ax3.set_xlabel("x")
        ax3.set_ylabel("P(X ≥ x)")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig3)

        # Berechnung und Anzeige der Wahrscheinlichkeiten mit Stetigkeitskorrektur
        p_pois_geq = 1 - poisson.cdf(k - 1, mu=lambda_val)
        p_norm_geq_stetig = 1 - norm.cdf(k - 0.5, mu, sigma)
        st.latex(rf"P(X \geq {k}) \approx P(Y > {k} - 0.5)")
        st.markdown(f"**Poisson:** P(X ≥ {k}) = `{p_pois_geq:.5f}`")
        st.markdown(f"**Normalapproximation mit Stetigkeitskorrektur:** P(Y > {k}-0.5) = `{p_norm_geq_stetig:.5f}`")

        # Wahrscheinlichkeiten für k anzeigen
        st.subheader("Zusammenfassung für k")
        st.write(f"**P(X = {k}) (Poisson):** {p_pois:.5f}")
        st.write(f"**P(X = {k}) (Normal, Stetigkeitskorrektur):** {p_norm_stetig:.5f}")
        st.write(f"**P(X ≤ {k}) (Poisson):** {p_pois_leq:.5f}")
        st.write(f"**P(X ≤ {k}) (Normal, Stetigkeitskorrektur):** {p_norm_leq_stetig:.5f}")
        st.write(f"**P(X ≥ {k}) (Poisson):** {p_pois_geq:.5f}")
        st.write(f"**P(X ≥ {k}) (Normal, Stetigkeitskorrektur):** {p_norm_geq_stetig:.5f}")

        # Tabelle: Wahrscheinlichkeiten
        st.subheader("Tabelle: Wahrscheinlichkeiten")
        table_combined = {
            "x": x,
            f"P(X = x) bei λ = {lambda_val}": np.round(pmf_y, 6),
            f"P(X ≤ x) bei λ = {lambda_val}": np.round(cdf_y, 6),
            f"P(X ≥ x) bei λ = {lambda_val}": np.round(ccdf_y, 6)
        }
        st.dataframe(table_combined)

        # Erklärung Stetigkeitskorrektur
        st.markdown("""
        ---
        **Hinweis zur Stetigkeitskorrektur:**  
        Die Stetigkeitskorrektur gleicht den Unterschied zwischen der diskreten Poissonverteilung und der stetigen Normalverteilung aus.  
        - Für $P(X = k)$: Fläche von $k-0{,}5$ bis $k+0{,}5$  
        - Für $P(X ≤ k)$: bis $k+0{,}5$  
        - Für $P(X ≥ k)$: ab $k-0{,}5$  
        """)

# Tab 4: Animation
with tabs[3]:
    st.header("Poissonverteilung – Animation von λ (1.1 bis 60)")
    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        fig, ax = plt.subplots()

        def animate(frame):
            ax.clear()
            lambda_anim = frame
            x = np.arange(x_min, x_max + 1)
            y = poisson.pmf(x, mu=lambda_anim)
            ax.bar(x, y, color='skyblue', edgecolor='black')
            ax.set_title(f"Poisson-Verteilung (λ = {lambda_anim:.1f})")
            ax.set_xlabel("x")
            ax.set_ylabel("P(X = x)")
            ax.set_ylim(0, 0.25)
            ax.grid(True, linestyle='--', alpha=0.5)

        lambda_frames = np.linspace(1.1, 60, 100)
        ani = animation.FuncAnimation(fig, animate, frames=lambda_frames, interval=150)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
            ani.save(tmpfile.name, writer="pillow")
            gif_path = tmpfile.name

        st.image(gif_path, caption="Animation der Poisson-Verteilung für λ von 1.1 bis 60", use_container_width=True)
