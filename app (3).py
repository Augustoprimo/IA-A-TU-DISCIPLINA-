import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="FlujIA", page_icon="💰", layout="wide")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def ia(prompt):
    r = model.generate_content(prompt)
    return r.text

def fp(n):
    return f"${n:,.0f}".replace(",",".")

def cp():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c8d0e7",
        xaxis=dict(gridcolor="#2a3a55", linecolor="#2a3a55"),
        yaxis=dict(gridcolor="#2a3a55", linecolor="#2a3a55")
    )

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0f1117;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1f2e,#131720);border-right:1px solid #2a3044;}
section[data-testid="stSidebar"] *{color:#c8d0e7!important;}
.app-header{background:linear-gradient(135deg,#1e2d4a,#0d1a2e);border:1px solid #2a4a7a;border-radius:16px;padding:28px 36px;margin-bottom:24px;}
.app-header h1{font-family:'Space Grotesk',sans-serif;color:#e8f0ff;font-size:1.9rem;font-weight:700;margin:0 0 6px 0;}
.app-header p{color:#7a90b8;font-size:0.9rem;margin:0;}
.badge{display:inline-block;background:rgba(56,139,253,0.15);border:1px solid rgba(56,139,253,0.4);color:#388bfd;font-size:0.7rem;font-weight:700;padding:3px 10px;border-radius:20px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;}
.mc{background:#1a2035;border:1px solid #2a3a55;border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;margin-bottom:8px;}
.mc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;}
.mc.b::after{background:linear-gradient(90deg,#1f6feb,#388bfd);}
.mc.g::after{background:linear-gradient(90deg,#2ea043,#56d364);}
.mc.o::after{background:linear-gradient(90deg,#d29922,#e3b341);}
.mc.r::after{background:linear-gradient(90deg,#b91c1c,#ef4444);}
.mc.p::after{background:linear-gradient(90deg,#6f42c1,#a78bfa);}
.ml{color:#7a90b8;font-size:0.72rem;font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}
.mv{color:#e8f0ff;font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;line-height:1;}
.ms{color:#4d6080;font-size:0.75rem;margin-top:4px;}
.st2{color:#c8d0e7;font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:600;margin:24px 0 14px;padding-bottom:7px;border-bottom:1px solid #2a3a55;}
.ia-box{background:linear-gradient(135deg,#1a2540,#141e35);border:1px solid #2a4a7a;border-left:3px solid #388bfd;border-radius:10px;padding:18px 20px;margin-top:14px;color:#c8d0e7;line-height:1.7;font-size:0.9rem;}
.ia-lbl{font-size:0.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#388bfd;margin-bottom:8px;display:block;}
.aw{background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.3);border-radius:8px;padding:11px 15px;color:#e3b341;font-size:0.85rem;margin:7px 0;}
.ad{background:rgba(185,28,28,.1);border:1px solid rgba(185,28,28,.3);border-radius:8px;padding:11px 15px;color:#ef4444;font-size:0.85rem;margin:7px 0;}
.ok{background:rgba(46,160,67,.1);border:1px solid rgba(46,160,67,.3);border-radius:8px;padding:11px 15px;color:#56d364;font-size:0.85rem;margin:7px 0;}
.stTabs [data-baseweb="tab-list"]{background:#1a2035;border-radius:8px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{color:#7a90b8!important;border-radius:6px!important;font-weight:500!important;}
.stTabs [aria-selected="true"]{background:#1f6feb!important;color:white!important;}
.stNumberInput input,.stTextInput input,.stTextArea textarea{background:#1a2035!important;border:1px solid #2a3a55!important;color:#e8f0ff!important;border-radius:8px!important;}
label{color:#7a90b8!important;font-size:0.83rem!important;}
.stButton>button{background:linear-gradient(135deg,#1f6feb,#388bfd)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:9px 20px!important;}
</style>""", unsafe_allow_html=True)

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:20px 0 22px;'>
        <div style='font-size:2rem;'>💰</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:700;color:#e8f0ff;margin-top:6px;'>FlujIA</div>
        <div style='font-size:0.72rem;color:#4d6080;margin-top:3px;'>Predicción de Flujo de Fondos con IA</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    pag = st.radio("Nav", ["🏠  Inicio","📥  Cargar Datos","📈  Predicción ML","💧  Análisis de Liquidez","🤖  Asistente IA","📄  Generar Informe"],
                   label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#4d6080;text-align:center;'>Trabajo Final · IA a tu Disciplina<br>Augusto Primo · UCA Rosario</div>", unsafe_allow_html=True)

pag = pag.split("  ")[1]

if "D" not in st.session_state:
    st.session_state.D = {
        "empresa": "Distribuidora Comercial Rosario SRL",
        "ventas":    [8200000,8800000,9100000,9500000,10200000,10800000,11200000,11800000,12100000,12500000,12900000,13400000],
        "cobros":    [7500000,8100000,8400000,8800000,9400000,10000000,10300000,10900000,11200000,11600000,11900000,12400000],
        "pagos":     [5100000,5400000,5600000,5800000,6200000,6500000,6700000,7000000,7200000,7400000,7600000,7900000],
        "gastos":    [1800000,1900000,1950000,2000000,2100000,2200000,2250000,2300000,2400000,2450000,2500000,2600000],
        "salarios":  [1200000,1200000,1200000,1300000,1300000,1300000,1400000,1400000,1400000,1500000,1500000,1500000],
        "impuestos": [400000,420000,440000,460000,480000,500000,520000,540000,560000,580000,600000,620000],
        "chat": [],
    }
D = st.session_state.D

def calcular_flujo():
    flujo = []
    for i in range(12):
        egreso = D["pagos"][i] + D["gastos"][i] + D["salarios"][i] + D["impuestos"][i]
        flujo.append(D["cobros"][i] - egreso)
    return flujo

def predecir_proximos(serie, n=6):
    X = np.array(range(len(serie))).reshape(-1,1)
    y = np.array(serie)
    reg = LinearRegression().fit(X, y)
    futuros = np.array(range(len(serie), len(serie)+n)).reshape(-1,1)
    return reg.predict(futuros).tolist()

if pag == "Inicio":
    st.markdown("""<div class='app-header'>
        <div class='badge'>✦ Machine Learning aplicado a Finanzas</div>
        <h1>Predicción de Flujo de Fondos con IA</h1>
        <p>Análisis predictivo · Detección de déficit · Proyecciones automáticas · Asistente financiero</p>
    </div>""", unsafe_allow_html=True)
    flujo = calcular_flujo()
    total_cobros = sum(D["cobros"])
    total_egresos = sum(D["pagos"]) + sum(D["gastos"]) + sum(D["salarios"]) + sum(D["impuestos"])
    total_flujo = sum(flujo)
    meses_deficit = sum(1 for f in flujo if f < 0)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='mc b'><div class='ml'>Total Cobros</div><div class='mv'>{fp(total_cobros)}</div><div class='ms'>Últimos 12 meses</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='mc r'><div class='ml'>Total Egresos</div><div class='mv'>{fp(total_egresos)}</div><div class='ms'>Pagos + Gastos + Salarios + Imp.</div></div>", unsafe_allow_html=True)
    with c3:
        cls = "g" if total_flujo > 0 else "r"
        st.markdown(f"<div class='mc {cls}'><div class='ml'>Flujo Neto Total</div><div class='mv'>{fp(total_flujo)}</div><div class='ms'>Cobros menos egresos</div></div>", unsafe_allow_html=True)
    with c4:
        cls = "g" if meses_deficit == 0 else "r"
        st.markdown(f"<div class='mc {cls}'><div class='ml'>Meses con Déficit</div><div class='mv'>{meses_deficit}</div><div class='ms'>De 12 meses analizados</div></div>", unsafe_allow_html=True)
    st.markdown('<div class="st2">Flujo de Fondos Mensual</div>', unsafe_allow_html=True)
    colores = ["#56d364" if f >= 0 else "#ef4444" for f in flujo]
    fig = go.Figure(go.Bar(
        x=MESES, y=flujo,
        marker_color=colores,
        text=[fp(f) for f in flujo],
        textposition="outside",
        textfont=dict(color="#c8d0e7", size=10)
    ))
    fig.update_layout(**cp(), height=320, margin=dict(t=20,b=10,l=0,r=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="st2">Evolución de Cobros vs Egresos</div>', unsafe_allow_html=True)
    egresos_mes = [D["pagos"][i]+D["gastos"][i]+D["salarios"][i]+D["impuestos"][i] for i in range(12)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=MESES, y=D["cobros"], name="Cobros", line=dict(color="#56d364", width=2.5), mode="lines+markers"))
    fig2.add_trace(go.Scatter(x=MESES, y=egresos_mes, name="Egresos", line=dict(color="#ef4444", width=2.5), mode="lines+markers"))
    fig2.update_layout(**cp(), height=300, margin=dict(t=20,b=10,l=0,r=0), legend=dict(font=dict(color="#c8d0e7")))
    st.plotly_chart(fig2, use_container_width=True)

elif pag == "Cargar Datos":
    st.markdown("""<div class='app-header'>
        <div class='badge'>📥 Datos Históricos</div>
        <h1>Cargar Datos Financieros</h1>
        <p>Ingresá los datos mensuales de tu empresa para generar predicciones</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="st2">Datos de los últimos 12 meses</div>', unsafe_allow_html=True)
    with st.form("fd"):
        emp = st.text_input("Nombre de la empresa", value=D["empresa"])
        categorias = [("ventas","Ventas mensuales"),("cobros","Cobros mensuales"),
                      ("pagos","Pagos a proveedores"),("gastos","Gastos operativos"),
                      ("salarios","Salarios"),("impuestos","Impuestos")]
        nuevos = {}
        for key, label in categorias:
            st.markdown(f'<div class="st2">{label} ($)</div>', unsafe_allow_html=True)
            cols = st.columns(6)
            vals = []
            for i in range(12):
                with cols[i % 6]:
                    v = st.number_input(MESES[i], min_value=0, value=D[key][i], step=50000, format="%d", key=f"{key}_{i}")
                    vals.append(v)
            nuevos[key] = vals
        ok = st.form_submit_button("💾 Guardar datos")
    if ok:
        st.session_state.D.update({"empresa": emp, **nuevos})
        st.markdown('<div class="ok">✅ Datos guardados correctamente.</div>', unsafe_allow_html=True)
        D = st.session_state.D
    st.markdown('<div class="st2">Vista Previa</div>', unsafe_allow_html=True)
    flujo = calcular_flujo()
    df = pd.DataFrame({
        "Mes": MESES,
        "Ventas": [fp(v) for v in D["ventas"]],
        "Cobros": [fp(v) for v in D["cobros"]],
        "Egresos": [fp(D["pagos"][i]+D["gastos"][i]+D["salarios"][i]+D["impuestos"][i]) for i in range(12)],
        "Flujo Neto": [fp(f) for f in flujo],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

elif pag == "Predicción ML":
    st.markdown("""<div class='app-header'>
        <div class='badge'>📈 Machine Learning</div>
        <h1>Predicción de Flujo de Fondos</h1>
        <p>Modelo de regresión lineal entrenado con datos históricos · Proyección a 6 meses</p>
    </div>""", unsafe_allow_html=True)
    flujo = calcular_flujo()
    pred_cobros = predecir_proximos(D["cobros"])
    pred_pagos  = predecir_proximos(D["pagos"])
    pred_gastos = predecir_proximos(D["gastos"])
    pred_sal    = predecir_proximos(D["salarios"])
    pred_imp    = predecir_proximos(D["impuestos"])
    pred_flujo  = [pred_cobros[i]-(pred_pagos[i]+pred_gastos[i]+pred_sal[i]+pred_imp[i]) for i in range(6)]
    meses_fut   = ["Ene 26","Feb 26","Mar 26","Abr 26","May 26","Jun 26"]
    t1,t2,t3 = st.tabs(["📊 Proyecciones","🎯 Modelo ML","🤖 Análisis IA"])
    with t1:
        st.markdown('<div class="st2">Flujo de Fondos: Histórico vs Proyectado</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MESES, y=flujo, name="Histórico",
            line=dict(color="#388bfd", width=2.5), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=meses_fut, y=pred_flujo, name="Proyectado (ML)",
            line=dict(color="#a78bfa", width=2.5, dash="dot"), mode="lines+markers"))
        fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", annotation_text="Línea de déficit")
        fig.update_layout(**cp(), height=340, margin=dict(t=20,b=10,l=0,r=0), legend=dict(font=dict(color="#c8d0e7")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="st2">Detalle de Proyecciones</div>', unsafe_allow_html=True)
        df_pred = pd.DataFrame({
            "Mes": meses_fut,
            "Cobros Proyectados": [fp(v) for v in pred_cobros],
            "Egresos Proyectados": [fp(pred_pagos[i]+pred_gastos[i]+pred_sal[i]+pred_imp[i]) for i in range(6)],
            "Flujo Proyectado": [fp(v) for v in pred_flujo],
            "Estado": ["✅ Superávit" if v >= 0 else "❌ Déficit" for v in pred_flujo],
        })
        st.dataframe(df_pred, use_container_width=True, hide_index=True)
        for i, f in enumerate(pred_flujo):
            if f < 0:
                st.markdown(f'<div class="ad">❌ <b>{meses_fut[i]}</b>: se proyecta déficit de {fp(abs(f))} — planificar financiamiento.</div>', unsafe_allow_html=True)
            elif f < 500000:
                st.markdown(f'<div class="aw">⚠️ <b>{meses_fut[i]}</b>: flujo ajustado de {fp(f)} — monitorear de cerca.</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="st2">Información del Modelo de Machine Learning</div>', unsafe_allow_html=True)
        X = np.array(range(12)).reshape(-1,1)
        y = np.array(flujo)
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='mc b'><div class='ml'>Algoritmo</div><div class='mv' style='font-size:1rem;'>Regresión Lineal</div><div class='ms'>Scikit-learn</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='mc p'><div class='ml'>R² Score</div><div class='mv'>{r2:.3f}</div><div class='ms'>Bondad de ajuste</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='mc g'><div class='ml'>Datos de entrenamiento</div><div class='mv'>12 meses</div><div class='ms'>Serie histórica</div></div>", unsafe_allow_html=True)
        st.markdown('<div class="st2">Tendencia del Modelo</div>', unsafe_allow_html=True)
        x_line = list(range(18))
        y_line = [reg.intercept_ + reg.coef_[0]*x for x in x_line]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=MESES, y=flujo, name="Datos reales",
            mode="markers", marker=dict(color="#388bfd", size=10)))
        fig2.add_trace(go.Scatter(x=MESES+meses_fut, y=y_line[:18], name="Tendencia ML",
            line=dict(color="#a78bfa", width=2, dash="dot")))
        fig2.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        fig2.update_layout(**cp(), height=300, margin=dict(t=20,b=10,l=0,r=0), legend=dict(font=dict(color="#c8d0e7")))
        st.plotly_chart(fig2, use_container_width=True)
    with t3:
        if st.button("🤖 Analizar Predicciones con IA"):
            with st.spinner("Analizando proyecciones..."):
                deficit = [meses_fut[i] for i,f in enumerate(pred_flujo) if f < 0]
                resp = ia(f"Sos especialista en flujo de fondos para PyMEs argentinas. Analizá estas proyecciones de {D['empresa']} para los proximos 6 meses generadas con Machine Learning: {', '.join([f'{meses_fut[i]}: {fp(pred_flujo[i])}' for i in range(6)])}. Meses con deficit proyectado: {deficit if deficit else 'ninguno'}. Da: 1)Interpretacion de las proyecciones 2)Riesgos de liquidez 3)Recomendaciones concretas. Contexto: PyME argentina, inflacion alta.")
                st.markdown(f'<div class="ia-box"><span class="ia-lbl">🤖 Análisis IA</span>{resp.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

elif pag == "Análisis de Liquidez":
    st.markdown("""<div class='app-header'>
        <div class='badge'>💧 Liquidez</div>
        <h1>Análisis de Liquidez</h1>
        <p>Detección de problemas de liquidez · Alertas automáticas · Recomendaciones</p>
    </div>""", unsafe_allow_html=True)
    flujo = calcular_flujo()
    acumulado = []
    acum = 0
    for f in flujo:
        acum += f
        acumulado.append(acum)
    t1,t2 = st.tabs(["📊 Indicadores","📈 Flujo Acumulado"])
    with t1:
        st.markdown('<div class="st2">Indicadores de Liquidez por Mes</div>', unsafe_allow_html=True)
        rows = []
        for i in range(12):
            egreso = D["pagos"][i]+D["gastos"][i]+D["salarios"][i]+D["impuestos"][i]
            ratio = D["cobros"][i]/egreso if egreso else 0
            estado = "✅ Positivo" if flujo[i] >= 0 else "❌ Déficit"
            rows.append({
                "Mes": MESES[i],
                "Cobros": fp(D["cobros"][i]),
                "Egresos": fp(egreso),
                "Flujo Neto": fp(flujo[i]),
                "Ratio Cob/Egr": f"{ratio:.2f}",
                "Estado": estado,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="st2">Alertas</div>', unsafe_allow_html=True)
        alertas = 0
        for i in range(12):
            if flujo[i] < 0:
                st.markdown(f'<div class="ad">❌ <b>{MESES[i]}</b>: déficit de {fp(abs(flujo[i]))} — riesgo de incumplimiento de pagos.</div>', unsafe_allow_html=True)
                alertas += 1
            elif flujo[i] < 300000:
                st.markdown(f'<div class="aw">⚠️ <b>{MESES[i]}</b>: flujo muy ajustado de {fp(flujo[i])} — margen mínimo.</div>', unsafe_allow_html=True)
                alertas += 1
        if alertas == 0:
            st.markdown('<div class="ok">✅ No se detectaron problemas de liquidez en el período analizado.</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="st2">Flujo Acumulado del Período</div>', unsafe_allow_html=True)
        colores = ["#56d364" if a >= 0 else "#ef4444" for a in acumulado]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MESES, y=acumulado, name="Flujo acumulado",
            fill="tozeroy", line=dict(color="#388bfd", width=2.5),
            fillcolor="rgba(56,139,253,0.15)"))
        fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", annotation_text="Zona de riesgo")
        fig.update_layout(**cp(), height=340, margin=dict(t=20,b=10,l=0,r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="st2">Composición de Egresos</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Pagos proveedores","Gastos operativos","Salarios","Impuestos"],
            values=[sum(D["pagos"]),sum(D["gastos"]),sum(D["salarios"]),sum(D["impuestos"])],
            hole=0.55,
            marker=dict(colors=["#ef4444","#d29922","#a78bfa","#388bfd"],
                       line=dict(color="#0f1117",width=2)),
            textinfo="label+percent", textfont=dict(color="#c8d0e7",size=11)
        ))
        fig2.update_layout(**cp(), height=300, margin=dict(t=10,b=10,l=0,r=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

elif pag == "Asistente IA":
    st.markdown("""<div class='app-header'>
        <div class='badge'>🤖 Chat con IA</div>
        <h1>Asistente de Flujo de Fondos</h1>
        <p>Consultá cualquier duda sobre tu liquidez y proyecciones</p>
    </div>""", unsafe_allow_html=True)
    flujo = calcular_flujo()
    total_flujo = sum(flujo)
    deficit_meses = [MESES[i] for i,f in enumerate(flujo) if f < 0]
    ctx = f"Empresa: {D['empresa']}. Flujo neto total: {fp(total_flujo)}. Meses con deficit: {deficit_meses if deficit_meses else 'ninguno'}. Total cobros: {fp(sum(D['cobros']))}, Total egresos: {fp(sum(D['pagos'])+sum(D['gastos'])+sum(D['salarios'])+sum(D['impuestos']))}. PyME argentina, inflacion alta."
    st.markdown('<div class="st2">Preguntas Frecuentes</div>', unsafe_allow_html=True)
    sugs = ["Como mejorar mi flujo de fondos?","Que hacer si tengo deficit de liquidez?",
            "Como reducir mis egresos?","Que es el flujo de fondos acumulado?"]
    cols = st.columns(4)
    for i,s in enumerate(sugs):
        with cols[i]:
            if st.button(s, key=f"s{i}"):
                D["chat"].append({"r":"u","m":s})
                historial = "\n".join([f"{'Usuario' if m['r']=='u' else 'Asistente'}: {m['m']}" for m in D["chat"]])
                resp = ia(f"Sos asesor financiero PyMEs argentinas especialista en flujo de fondos. Contexto:\n{ctx}\n\nConversacion:\n{historial}\n\nResponde la ultima pregunta clara y concisa.")
                D["chat"].append({"r":"a","m":resp})
    if D["chat"]:
        st.markdown('<div class="st2">Conversación</div>', unsafe_allow_html=True)
        for m in D["chat"]:
            if m["r"]=="u":
                st.markdown(f'<div style="overflow:hidden"><div style="background:#1f2d44;border:1px solid #2a3a55;border-radius:10px 10px 2px 10px;padding:10px 14px;color:#e8f0ff;margin:6px 0 3px auto;max-width:80%;float:right;clear:both;font-size:0.88rem;">👤 {m["m"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="overflow:hidden"><div style="background:#1a2035;border:1px solid #2a4a7a;border-radius:10px 10px 10px 2px;padding:10px 14px;color:#c8d0e7;margin:3px auto 6px 0;max-width:85%;float:left;clear:both;font-size:0.88rem;line-height:1.6;">🤖 {m["m"].replace(chr(10),"<br>")}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    with st.form("cf", clear_on_submit=True):
        cx,cy = st.columns([5,1])
        with cx:
            umsg = st.text_input("Pregunta...", placeholder="Ej: Como mejorar mi liquidez?", label_visibility="collapsed")
        with cy:
            send = st.form_submit_button("Enviar")
    if send and umsg:
        D["chat"].append({"r":"u","m":umsg})
        historial = "\n".join([f"{'Usuario' if m['r']=='u' else 'Asistente'}: {m['m']}" for m in D["chat"]])
        resp = ia(f"Sos asesor financiero PyMEs argentinas. Contexto:\n{ctx}\n\nConversacion:\n{historial}\n\nResponde la ultima pregunta.")
        D["chat"].append({"r":"a","m":resp})
        st.rerun()
    if D["chat"] and st.button("Limpiar"):
        D["chat"] = []
        st.rerun()

elif pag == "Generar Informe":
    st.markdown("""<div class='app-header'>
        <div class='badge'>📄 Informe Automático</div>
        <h1>Informe de Flujo de Fondos</h1>
        <p>Diagnóstico completo generado por IA con predicciones y recomendaciones</p>
    </div>""", unsafe_allow_html=True)
    flujo = calcular_flujo()
    pred_cobros = predecir_proximos(D["cobros"])
    pred_pagos  = predecir_proximos(D["pagos"])
    pred_gastos = predecir_proximos(D["gastos"])
    pred_sal    = predecir_proximos(D["salarios"])
    pred_imp    = predecir_proximos(D["impuestos"])
    pred_flujo  = [pred_cobros[i]-(pred_pagos[i]+pred_gastos[i]+pred_sal[i]+pred_imp[i]) for i in range(6)]
    meses_fut   = ["Ene 26","Feb 26","Mar 26","Abr 26","May 26","Jun 26"]
    deficit_hist = [MESES[i] for i,f in enumerate(flujo) if f < 0]
    deficit_pred = [meses_fut[i] for i,f in enumerate(pred_flujo) if f < 0]
    if st.button("📄 Generar Informe Completo con IA", use_container_width=True):
        with st.spinner("Generando informe..."):
            inf = ia(f"Sos contador senior especialista en flujo de fondos para PyMEs argentinas. Genera un informe ejecutivo completo para {D['empresa']}. DATOS HISTORICOS (12 meses): Flujo neto mensual: {[fp(f) for f in flujo]}. Total cobros: {fp(sum(D['cobros']))}, Total egresos: {fp(sum(D['pagos'])+sum(D['gastos'])+sum(D['salarios'])+sum(D['impuestos']))}. Flujo neto total: {fp(sum(flujo))}. Meses con deficit historico: {deficit_hist if deficit_hist else 'ninguno'}. PROYECCIONES ML (6 meses): {[fp(f) for f in pred_flujo]}. Meses con deficit proyectado: {deficit_pred if deficit_pred else 'ninguno'}. Incluye: 1)RESUMEN EJECUTIVO 2)ANALISIS DEL FLUJO HISTORICO 3)PROYECCIONES CON MACHINE LEARNING 4)RIESGOS DE LIQUIDEZ 5)RECOMENDACIONES 6)CONCLUSION. PyME argentina, inflacion alta.")
            st.session_state["inf"] = inf
    if "inf" in st.session_state:
        inf = st.session_state["inf"]
        st.markdown(f"""<div style='background:linear-gradient(135deg,#1a2540,#141e35);border:1px solid #2a4a7a;border-radius:16px;padding:26px 30px;margin:18px 0;'>
            <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:700;color:#e8f0ff;margin-bottom:6px;'>Informe de Flujo de Fondos</div>
            <div style='color:#388bfd;font-size:0.82rem;margin-bottom:18px;'>{D["empresa"]} — Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
            <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px;'>
                <div style='background:rgba(56,139,253,.08);border:1px solid rgba(56,139,253,.2);border-radius:8px;padding:12px;'>
                    <div style='color:#7a90b8;font-size:0.7rem;text-transform:uppercase;'>Total Cobros</div>
                    <div style='color:#e8f0ff;font-size:1.1rem;font-weight:700;'>{fp(sum(D["cobros"]))}</div></div>
                <div style='background:rgba(46,160,67,.08);border:1px solid rgba(46,160,67,.2);border-radius:8px;padding:12px;'>
                    <div style='color:#7a90b8;font-size:0.7rem;text-transform:uppercase;'>Flujo Neto</div>
                    <div style='color:#56d364;font-size:1.1rem;font-weight:700;'>{fp(sum(flujo))}</div></div>
                <div style='background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);border-radius:8px;padding:12px;'>
                    <div style='color:#7a90b8;font-size:0.7rem;text-transform:uppercase;'>Meses Déficit</div>
                    <div style='color:#a78bfa;font-size:1.1rem;font-weight:700;'>{len(deficit_hist)}</div></div>
            </div>
            <div class='ia-box' style='margin:0;'><span class='ia-lbl'>📄 Informe generado por IA</span>{inf.replace(chr(10),"<br>")}</div>
        </div>""", unsafe_allow_html=True)
        st.download_button("⬇️ Descargar informe (.txt)",
            data=f"INFORME DE FLUJO DE FONDOS\n{D['empresa']}\nGenerado: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n{'='*60}\n\n{inf}",
            file_name="informe_flujo_fondos.txt", mime="text/plain")
