// Animations légères, compteurs et formulaires de simulation
document.addEventListener("DOMContentLoaded", () => {
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.08 });
  document.querySelectorAll(".reveal").forEach(el => io.observe(el));

  document.querySelectorAll("[data-count]").forEach(el => {
    const end = +el.dataset.count, t0 = performance.now();
    const tick = t => { const p = Math.min((t - t0) / 900, 1); el.textContent = Math.round(end * p); if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  });

  document.querySelectorAll('input[type="range"]').forEach(r => {
    const out = r.parentElement.querySelector("output");
    r.addEventListener("input", () => out.textContent = (+r.value).toFixed(1));
  });

  const fmt = n => n.toLocaleString("fr-FR").replace(/\u202f|\u00a0/g, " ");
  document.querySelectorAll("form[data-predict]").forEach(form => {
    form.addEventListener("submit", async e => {
      e.preventDefault();
      const box = document.querySelector(form.dataset.target), full = form.dataset.mode === "full";
      box.className = "result-box";
      box.textContent = "Calcul en cours…";
      try {
        const resp = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(Object.fromEntries(new FormData(form))) });
        const d = await resp.json();
        if (!resp.ok) { box.className = "result-box show error"; box.textContent = "⚠️ " + d.error; return; }
        box.textContent = "";
        const add = (cls, html) => { const div = document.createElement("div"); div.className = cls; div.innerHTML = html; box.appendChild(div); return div; };
        add("res-card salary", `<span>Salaire mensuel estimé</span><b>${fmt(d.salary)} MAD</b>`);
        if (full) {
          const p = d.proba, risk = p === null ? (d.attrition ? "high" : "low") : p >= 0.6 ? "high" : p >= 0.4 ? "mid" : "low";
          const label = d.attrition ? "Risque de départ" : "Reste probablement";
          add(`res-card risk ${risk}`, `<span>Prédiction d'attrition (${d.model})</span><b>${label}</b>` +
            (p !== null ? `<div class="bar"><i style="width:${Math.round(p * 100)}%"></i></div><small>Probabilité d'attrition : ${(p * 100).toFixed(0)}%</small>` : "<small>Probabilité non disponible pour ce modèle.</small>"));
          const ex = add("res-card explain", `<span>Explication</span><p></p><ul></ul><small>Prédiction expérimentale : à ne pas utiliser seule pour décider.</small>`);
          ex.querySelector("p").textContent = d.explanation;
          d.notes.forEach(n => { const li = document.createElement("li"); li.textContent = n; ex.querySelector("ul").appendChild(li); });
        }
        box.classList.add("show");
      } catch (err) { box.className = "result-box show error"; box.textContent = "⚠️ Erreur de communication avec le serveur."; }
    });
  });
});
