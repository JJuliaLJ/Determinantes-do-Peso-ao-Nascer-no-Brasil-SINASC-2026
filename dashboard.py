from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats

pio.templates["pt_br"] = go.layout.Template(layout=dict(separators=",."))
pio.templates.default = "plotly+pt_br"

DADOS = Path(__file__).parent / "dados"
CENTRO_SEMANA = 39
CENTRO_IDADE = 28
LIMIAR_BPN = 2500

ROTULOS = {
    "PESO": "Peso ao nascer (g)",
    "SEMAGESTAC": "Semanas de gestação",
    "IDADEMAE": "Idade da mãe (anos)",
    "CONSPRENAT": "Consultas de pré-natal",
    "MESPRENAT": "Mês de início do pré-natal",
    "ANOS_ESTUDO": "Anos de estudo da mãe",
    "QTDFILVIVO": "Filhos vivos anteriores",
    "APGAR5": "Apgar no 5º minuto",
    "UF": "Unidade federativa",
    "REGIAO": "Região",
    "ESC_CAT": "Escolaridade da mãe",
    "RACA_MAE": "Raça/cor da mãe",
    "FAIXA_MAE": "Faixa de idade da mãe",
    "SEXO_M": "Sexo do recém-nascido",
    "CESAREA": "Via de parto",
    "PRENATAL_ADEQ": "Adequação do pré-natal",
    "MULTIPLA": "Gestação múltipla",
    "ANOMALIA": "Anomalia congênita",
    "SEM_COMPANHEIRO": "Situação conjugal",
    "BAIXO_PESO": "Baixo peso ao nascer",
    "PREMATURO": "Prematuridade",
    "nascimentos": "Nascimentos",
    "peso_medio": "Peso médio (g)",
    "taxa_bpn": "Baixo peso (%)",
    "taxa_prematuro": "Prematuridade (%)",
    "taxa_cesarea": "Cesárea (%)",
    "prenatal_adequado": "Pré-natal adequado (%)",
    "cobertura_prenatal": "Pré-natal adequado (%)",
    "consultas_media": "Consultas de pré-natal (média)",
    "escolaridade_media": "Anos de estudo (média)",
    "residuo": "Resíduo do modelo agregado (pp)",
    "bpn_previsto": "Baixo peso previsto (%)",
}

BINARIAS = {
    "SEXO_M": {0.0: "Feminino", 1.0: "Masculino"},
    "CESAREA": {0.0: "Vaginal", 1.0: "Cesárea"},
    "PRENATAL_ADEQ": {0.0: "Inadequado", 1.0: "Adequado"},
    "MULTIPLA": {0.0: "Única", 1.0: "Múltipla"},
    "ANOMALIA": {0.0: "Ausente", 1.0: "Presente"},
    "SEM_COMPANHEIRO": {0.0: "Com companheiro", 1.0: "Sem companheiro"},
    "BAIXO_PESO": {0.0: "≥ 2.500 g", 1.0: "< 2.500 g"},
    "PREMATURO": {0.0: "≥ 37 semanas", 1.0: "< 37 semanas"},
}

CONTINUAS = [
    "PESO",
    "SEMAGESTAC",
    "IDADEMAE",
    "CONSPRENAT",
    "MESPRENAT",
    "ANOS_ESTUDO",
    "QTDFILVIVO",
    "APGAR5",
]

CATEGORICAS = [
    "REGIAO",
    "UF",
    "ESC_CAT",
    "RACA_MAE",
    "FAIXA_MAE",
    "SEXO_M",
    "CESAREA",
    "PRENATAL_ADEQ",
    "MULTIPLA",
    "ANOMALIA",
    "SEM_COMPANHEIRO",
]


def rotular(nome):
    return ROTULOS.get(nome, nome)


def numero(valor, decimais=0):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    return (
        f"{valor:,.{decimais}f}"
        .replace(",", "\x00")
        .replace(".", ",")
        .replace("\x00", ".")
    )


@st.cache_data(show_spinner=False)
def carregar(nome):
    return pd.read_parquet(DADOS / f"{nome}.parquet")


@st.cache_data(show_spinner=False)
def carregar_amostra():
    dados = pd.read_parquet(DADOS / "sinasc_amostra.parquet")
    dados["MES_NASC"] = dados["MES_NASC"].astype("string")
    for coluna, mapa in BINARIAS.items():
        dados[f"{coluna}_ROT"] = dados[coluna].map(mapa).astype("category")
    return dados


def coluna_exibicao(coluna):
    return f"{coluna}_ROT" if coluna in BINARIAS else coluna


def aplicar_filtros(dados, filtros):
    mascara = pd.Series(True, index=dados.index)
    for coluna, selecionados in filtros.items():
        if selecionados:
            mascara &= dados[coluna_exibicao(coluna)].isin(selecionados)
    return dados[mascara]


def intervalo_media(serie, confianca):
    amostra = serie.dropna()
    n = len(amostra)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    media = amostra.mean()
    erro = amostra.std(ddof=1) / np.sqrt(n)
    margem = stats.t.ppf(0.5 + confianca / 2, n - 1) * erro
    return media, media - margem, media + margem, n


def intervalo_wilson(casos, n, confianca):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = stats.norm.ppf(0.5 + confianca / 2)
    proporcao = casos / n
    centro = (proporcao + z**2 / (2 * n)) / (1 + z**2 / n)
    raio = (
        z
        / (1 + z**2 / n)
        * np.sqrt(proporcao * (1 - proporcao) / n + z**2 / (4 * n**2))
    )
    return proporcao * 100, (centro - raio) * 100, (centro + raio) * 100


def welch(grupo_a, grupo_b, confianca):
    a = grupo_a.dropna().to_numpy()
    b = grupo_b.dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    estatistica, valor_p = stats.ttest_ind(a, b, equal_var=False)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    erro = np.sqrt(var_a / len(a) + var_b / len(b))
    gl = erro**4 / (
        (var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1)
    )
    diferenca = a.mean() - b.mean()
    margem = stats.t.ppf(0.5 + confianca / 2, gl) * erro
    agrupado = np.sqrt(
        ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    )
    return {
        "n_a": len(a),
        "n_b": len(b),
        "media_a": a.mean(),
        "media_b": b.mean(),
        "diferenca": diferenca,
        "ic_inferior": diferenca - margem,
        "ic_superior": diferenca + margem,
        "estatistica": estatistica,
        "gl": gl,
        "valor_p": valor_p,
        "d_cohen": diferenca / agrupado if agrupado else np.nan,
    }


@st.cache_data(show_spinner=False)
def mapa_coeficientes():
    tabela = carregar("coeficientes_m3")
    return dict(zip(tabela["termo"], tabela["coeficiente"]))


def prever_peso(coeficientes, entrada):
    predito = coeficientes["Intercept"]
    termo_regiao = f"C(REGIAO, Treatment('SE'))[T.{entrada['REGIAO']}]"
    predito += coeficientes.get(termo_regiao, 0.0)
    termo_raca = f"C(RACA_MAE, Treatment('Branca'))[T.{entrada['RACA_MAE']}]"
    predito += coeficientes.get(termo_raca, 0.0)
    semana = entrada["SEMAGESTAC"] - CENTRO_SEMANA
    predito += coeficientes["SEMAGESTAC_C"] * semana
    predito += coeficientes["SEMAGESTAC_C2"] * semana**2
    predito += coeficientes["IDADEMAE_C"] * (entrada["IDADEMAE"] - CENTRO_IDADE)
    for termo in [
        "CONSPRENAT",
        "MESPRENAT",
        "ANOS_ESTUDO",
        "QTDFILVIVO",
        "SEXO_M",
        "CESAREA",
        "MULTIPLA",
        "ANOMALIA",
        "SEM_COMPANHEIRO",
    ]:
        predito += coeficientes[termo] * entrada[termo]
    return predito


def prever_vetorizado(coeficientes, dados):
    predito = pd.Series(coeficientes["Intercept"], index=dados.index, dtype="float64")
    for regiao in dados["REGIAO"].dropna().unique():
        termo = f"C(REGIAO, Treatment('SE'))[T.{regiao}]"
        if termo in coeficientes:
            predito += (dados["REGIAO"] == regiao) * coeficientes[termo]
    for raca in dados["RACA_MAE"].dropna().unique():
        termo = f"C(RACA_MAE, Treatment('Branca'))[T.{raca}]"
        if termo in coeficientes:
            predito += (dados["RACA_MAE"] == raca) * coeficientes[termo]
    semana = dados["SEMAGESTAC"] - CENTRO_SEMANA
    predito += coeficientes["SEMAGESTAC_C"] * semana
    predito += coeficientes["SEMAGESTAC_C2"] * semana**2
    predito += coeficientes["IDADEMAE_C"] * (dados["IDADEMAE"] - CENTRO_IDADE)
    for termo in [
        "CONSPRENAT",
        "MESPRENAT",
        "ANOS_ESTUDO",
        "QTDFILVIVO",
        "SEXO_M",
        "CESAREA",
        "MULTIPLA",
        "ANOMALIA",
        "SEM_COMPANHEIRO",
    ]:
        predito += coeficientes[termo] * dados[termo]
    return predito


def sigma_residual():
    metricas = carregar("metricas_modelo").set_index("metrica")["valor"]
    return float(metricas["RMSE_teste"])


st.set_page_config(
    page_title="SINASC 2026 — Peso ao nascer",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-size: 18px !important;
}

h1 {
    font-size: 42px !important;
}

h2 {
    font-size: 34px !important;
}

h3 {
    font-size: 28px !important;
}

p {
    font-size: 18px !important;
}

label {
    font-size: 18px !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] div {
    font-size: 16px !important;
}

button[data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 600 !important;
}

button[kind] {
    font-size: 18px !important;
}

div[data-baseweb="select"] {
    font-size: 18px !important;
}

input {
    font-size: 18px !important;
}

[role="radiogroup"] label {
    font-size: 18px !important;
}

[data-testid="stCheckbox"] label {
    font-size: 18px !important;
}

[data-testid="stSlider"] {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)


amostra = carregar_amostra()
coeficientes = mapa_coeficientes()

st.title("Peso ao nascer no Brasil — SINASC 2026")

st.sidebar.header("Filtros")
confianca = st.sidebar.select_slider(
    "Nível de confiança",
    options=[0.80, 0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda v: f"{v:.0%}",
)

filtros = {}
for coluna in ["REGIAO", "UF", "ESC_CAT", "RACA_MAE", "SEXO_M", "CESAREA", "PRENATAL_ADEQ"]:
    opcoes = sorted(amostra[coluna_exibicao(coluna)].dropna().unique().tolist())
    filtros[coluna] = st.sidebar.multiselect(rotular(coluna), opcoes, default=[])

faixa_semanas = st.sidebar.slider(
    rotular("SEMAGESTAC"),
    int(amostra["SEMAGESTAC"].min()),
    int(amostra["SEMAGESTAC"].max()),
    (int(amostra["SEMAGESTAC"].min()), int(amostra["SEMAGESTAC"].max())),
)
faixa_idade = st.sidebar.slider(
    rotular("IDADEMAE"),
    int(amostra["IDADEMAE"].min()),
    int(amostra["IDADEMAE"].max()),
    (int(amostra["IDADEMAE"].min()), int(amostra["IDADEMAE"].max())),
)

recorte = aplicar_filtros(amostra, filtros)
recorte = recorte[
    recorte["SEMAGESTAC"].between(*faixa_semanas)
    & recorte["IDADEMAE"].between(*faixa_idade)
]

st.sidebar.markdown(
    f"""
    <div>
        <p style='margin-bottom: 0px; color: white;'>Registros no recorte</p>
        <h1 style='text-align: center; margin-top: 0px; font-size: 20px;'>{numero(len(recorte))}</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Os filtros alteram as abas Exploração, Território, Temporal e Regressão. "
    "As abas Prevalências e Testes de hipóteses trazem resultados calculados "
    "sobre a população completa (1.388.675 registros)."
)

if recorte.empty:
    st.warning("Nenhum registro atende à combinação de filtros selecionada.")
    st.stop()

media, ic_inferior, ic_superior, n_peso = intervalo_media(recorte["PESO"], confianca)
casos_bpn = int(recorte["BAIXO_PESO"].sum())
proporcao_bpn, bpn_inferior, bpn_superior = intervalo_wilson(
    casos_bpn, int(recorte["BAIXO_PESO"].notna().sum()), confianca
)

colunas = st.columns(4)
colunas[0].metric(
    "Peso médio (g)",
    numero(media, 1),
    f"IC {confianca:.0%}: {numero(ic_inferior, 1)} – {numero(ic_superior, 1)}",
    delta_color="off",
)
colunas[1].metric(
    "Baixo peso (%)",
    numero(proporcao_bpn, 2),
    f"IC {confianca:.0%}: {numero(bpn_inferior, 2)} – {numero(bpn_superior, 2)}",
    delta_color="off",
)
colunas[2].metric(
    "Prematuridade (%)",
    numero(recorte["PREMATURO"].mean() * 100, 2),
)
colunas[3].metric(
    "Desvio-padrão do peso (g)",
    numero(recorte["PESO"].std(ddof=1), 1),
)

aba_exploracao, aba_territorio, aba_temporal, aba_prevalencia, aba_testes, aba_regressao, aba_simulador = st.tabs(
    [
        "Exploração",
        "Território",
        "Temporal",
        "Prevalências",
        "Testes de hipóteses",
        "Regressão",
        "Simulador",
    ]
)

with aba_exploracao:
    st.subheader("Distribuição de uma variável contínua")
    st.caption(
        "Objetivo: examinar forma, centro e dispersão da variável escolhida no recorte "
        "ativo, com corte opcional por grupo."
    )
    esquerda, direita = st.columns([1, 3])
    with esquerda:
        variavel = st.selectbox(
            "Variável", CONTINUAS, format_func=rotular, key="dist_var"
        )
        agrupador = st.selectbox(
            "Separar por",
            ["Nenhum"] + CATEGORICAS,
            format_func=lambda v: "Nenhum" if v == "Nenhum" else rotular(v),
            key="dist_grupo",
        )
        caixas = st.slider("Número de faixas", 20, 120, 60, key="dist_bins")
        marcar_limiar = st.checkbox(
            "Marcar 2.500 g", value=variavel == "PESO", key="dist_limiar"
        )
    with direita:
        if agrupador == "Nenhum":
            figura = px.histogram(
                recorte, x=variavel, nbins=caixas, labels={variavel: rotular(variavel)}
            )
        else:
            figura = px.histogram(
                recorte,
                x=variavel,
                color=coluna_exibicao(agrupador),
                nbins=caixas,
                barmode="overlay",
                opacity=0.6,
                labels={
                    variavel: rotular(variavel),
                    coluna_exibicao(agrupador): rotular(agrupador),
                },
            )
        if marcar_limiar and variavel == "PESO":
            figura.add_vline(x=LIMIAR_BPN, line_dash="dash", line_color="#C44E52")
        figura.update_layout(yaxis_title="Nascimentos", height=420)
        st.plotly_chart(figura, width="stretch")

    st.divider()
    st.subheader("Comparação entre grupos com intervalo de confiança")
    st.caption(
        "Objetivo: comparar médias entre categorias e verificar a sobreposição dos "
        "intervalos de confiança de cada grupo."
    )
    esquerda, direita = st.columns([1, 3])
    with esquerda:
        desfecho = st.selectbox(
            "Desfecho", CONTINUAS, format_func=rotular, key="comp_desfecho"
        )
        categoria = st.selectbox(
            "Categoria", CATEGORICAS, format_func=rotular, key="comp_cat"
        )
        formato = st.radio(
            "Representação", ["Média com IC", "Boxplot", "Violino"], key="comp_formato"
        )
    with direita:
        chave = coluna_exibicao(categoria)
        if formato == "Média com IC":
            linhas = []
            for valor, grupo in recorte.groupby(chave, observed=True):
                media_grupo, inferior, superior, n_grupo = intervalo_media(
                    grupo[desfecho], confianca
                )
                if n_grupo >= 2:
                    linhas.append(
                        {
                            "grupo": str(valor),
                            "media": media_grupo,
                            "inferior": media_grupo - inferior,
                            "superior": superior - media_grupo,
                            "n": n_grupo,
                        }
                    )
            resumo = pd.DataFrame(linhas).sort_values("media")
            figura = go.Figure(
                go.Scatter(
                    x=resumo["media"],
                    y=resumo["grupo"],
                    mode="markers",
                    marker=dict(size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=resumo["superior"],
                        arrayminus=resumo["inferior"],
                    ),
                    customdata=resumo["n"],
                    hovertemplate="%{y}<br>média: %{x:.1f}<br>n: %{customdata:,}<extra></extra>",
                )
            )
            figura.update_layout(
                xaxis_title=f"{rotular(desfecho)} — média e IC {confianca:.0%}",
                yaxis_title=rotular(categoria),
                height=460,
            )
        elif formato == "Boxplot":
            figura = px.box(
                recorte,
                x=chave,
                y=desfecho,
                labels={chave: rotular(categoria), desfecho: rotular(desfecho)},
            )
            figura.update_layout(height=460)
        else:
            figura = px.violin(
                recorte,
                x=chave,
                y=desfecho,
                box=True,
                labels={chave: rotular(categoria), desfecho: rotular(desfecho)},
            )
            figura.update_layout(height=460)
        st.plotly_chart(figura, width="stretch")

    st.divider()
    st.subheader("Relação entre duas variáveis contínuas")
    st.caption(
        "Objetivo: inspecionar a forma e a intensidade da associação entre duas "
        "variáveis, com a correlação calculada sobre o recorte ativo."
    )
    esquerda, direita = st.columns([1, 3])
    with esquerda:
        eixo_x = st.selectbox(
            "Eixo X", CONTINUAS, index=1, format_func=rotular, key="disp_x"
        )
        eixo_y = st.selectbox(
            "Eixo Y", CONTINUAS, index=0, format_func=rotular, key="disp_y"
        )
        colorir = st.selectbox(
            "Cor",
            ["Nenhum"] + CATEGORICAS,
            format_func=lambda v: "Nenhum" if v == "Nenhum" else rotular(v),
            key="disp_cor",
        )
        tendencia = st.checkbox("Ajuste linear", value=True, key="disp_tend")
        pontos = st.slider("Pontos amostrados", 2000, 40000, 12000, step=2000, key="disp_n")
    with direita:
        base = recorte[[eixo_x, eixo_y] + ([coluna_exibicao(colorir)] if colorir != "Nenhum" else [])].dropna()
        if len(base) > pontos:
            base = base.sample(pontos, random_state=7)
        figura = px.scatter(
            base,
            x=eixo_x,
            y=eixo_y,
            color=coluna_exibicao(colorir) if colorir != "Nenhum" else None,
            opacity=0.35,
            trendline="ols" if tendencia else None,
            trendline_color_override="#C44E52",
            labels={
                eixo_x: rotular(eixo_x),
                eixo_y: rotular(eixo_y),
                **({coluna_exibicao(colorir): rotular(colorir)} if colorir != "Nenhum" else {}),
            },
        )
        figura.update_layout(height=460)
        st.plotly_chart(figura, width="stretch")

        par = recorte[[eixo_x, eixo_y]].dropna()
        if len(par) > 2:
            pearson, p_pearson = stats.pearsonr(par[eixo_x], par[eixo_y])
            spearman, p_spearman = stats.spearmanr(par[eixo_x], par[eixo_y])
            z = np.arctanh(pearson)
            margem = stats.norm.ppf(0.5 + confianca / 2) / np.sqrt(len(par) - 3)
            st.dataframe(
                pd.DataFrame(
                    {
                        "medida": ["r de Pearson", "rô de Spearman", "r²"],
                        "valor": [pearson, spearman, pearson**2],
                        "valor_p": [p_pearson, p_spearman, np.nan],
                        f"IC {confianca:.0%} inferior": [np.tanh(z - margem), np.nan, np.nan],
                        f"IC {confianca:.0%} superior": [np.tanh(z + margem), np.nan, np.nan],
                        "n": [len(par), len(par), len(par)],
                    }
                ),
                hide_index=True,
                width="stretch",
            )

with aba_territorio:
    st.subheader("Indicadores por unidade federativa")
    st.caption(
        "Objetivo: ordenar as unidades federativas pelo indicador escolhido e localizar "
        "a posição de cada uma em relação às demais."
    )
    indicadores_uf = [
        "peso_medio",
        "taxa_bpn",
        "taxa_prematuro",
        "taxa_cesarea",
        "prenatal_adequado",
        "consultas_media",
        "escolaridade_media",
        "nascimentos",
    ]
    esquerda, direita = st.columns([1, 3])
    with esquerda:
        indicador = st.selectbox(
            "Indicador", indicadores_uf, format_func=rotular, key="uf_ind"
        )
        nivel = st.radio("Agregação", ["UF", "REGIAO"], format_func=rotular, key="uf_nivel")
    with direita:
        agregado = (
            recorte.groupby(nivel, observed=True)
            .agg(
                nascimentos=("PESO", "size"),
                peso_medio=("PESO", "mean"),
                taxa_bpn=("BAIXO_PESO", lambda s: s.mean() * 100),
                taxa_prematuro=("PREMATURO", lambda s: s.mean() * 100),
                taxa_cesarea=("CESAREA", lambda s: s.mean() * 100),
                prenatal_adequado=("PRENATAL_ADEQ", lambda s: s.mean() * 100),
                consultas_media=("CONSPRENAT", "mean"),
                escolaridade_media=("ANOS_ESTUDO", "mean"),
            )
            .reset_index()
            .sort_values(indicador)
        )
        figura = px.bar(
            agregado,
            x=indicador,
            y=nivel,
            orientation="h",
            labels={indicador: rotular(indicador), nivel: rotular(nivel)},
        )
        figura.add_vline(
            x=agregado[indicador].mean(), line_dash="dot", line_color="#4C72B0"
        )
        figura.update_layout(height=max(420, 22 * len(agregado)))
        st.plotly_chart(figura, width="stretch")
        st.dataframe(agregado, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Associação entre dois indicadores territoriais")
    st.caption(
        "Objetivo: cruzar dois indicadores no nível da unidade federativa, dimensionando "
        "os pontos pelo volume de nascimentos."
    )
    esquerda, direita = st.columns([1, 3])
    with esquerda:
        indicador_x = st.selectbox(
            "Indicador X", indicadores_uf, index=4, format_func=rotular, key="terr_x"
        )
        indicador_y = st.selectbox(
            "Indicador Y", indicadores_uf, index=1, format_func=rotular, key="terr_y"
        )
    with direita:
        figura = px.scatter(
            agregado,
            x=indicador_x,
            y=indicador_y,
            size="nascimentos",
            text=nivel,
            trendline="ols",
            trendline_color_override="#C44E52",
            labels={
                indicador_x: rotular(indicador_x),
                indicador_y: rotular(indicador_y),
            },
        )
        figura.update_traces(textposition="top center")
        figura.update_layout(height=520)
        st.plotly_chart(figura, width="stretch")

    st.divider()
    st.subheader("Resíduos do modelo agregado por unidade federativa")
    st.caption(
        "Objetivo: comparar a taxa de baixo peso observada com a prevista pelo modelo "
        "agregado, calculada sobre a população completa."
    )
    residuos = carregar("agregado_uf").reset_index().sort_values("residuo")
    figura = px.bar(
        residuos,
        x="residuo",
        y="UF",
        orientation="h",
        color="residuo",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"residuo": rotular("residuo"), "UF": rotular("UF")},
        hover_data=["taxa_bpn", "bpn_previsto", "nascimentos"],
    )
    figura.update_layout(height=640, coloraxis_showscale=False)
    st.plotly_chart(figura, width="stretch")

with aba_temporal:
    st.subheader("Evolução mensal dos indicadores")
    st.caption(
        "Objetivo: acompanhar o comportamento dos indicadores ao longo dos meses de "
        "2026 cobertos pelo dataset."
    )
    indicadores_mes = [
        "peso_medio",
        "taxa_bpn",
        "taxa_prematuro",
        "taxa_cesarea",
        "prenatal_adequado",
        "nascimentos",
    ]
    escolhidos = st.multiselect(
        "Indicadores",
        indicadores_mes,
        default=["peso_medio", "taxa_bpn"],
        format_func=rotular,
        key="mes_ind",
    )
    fonte = st.radio(
        "Fonte",
        ["Recorte filtrado (amostra)", "População completa"],
        horizontal=True,
        key="mes_fonte",
    )
    if fonte == "População completa":
        serie = carregar("serie_mensal").rename(columns={"mes": "MES_NASC"})
    else:
        serie = (
            recorte.groupby("MES_NASC", observed=True)
            .agg(
                nascimentos=("PESO", "size"),
                peso_medio=("PESO", "mean"),
                taxa_bpn=("BAIXO_PESO", lambda s: s.mean() * 100),
                taxa_prematuro=("PREMATURO", lambda s: s.mean() * 100),
                taxa_cesarea=("CESAREA", lambda s: s.mean() * 100),
                prenatal_adequado=("PRENATAL_ADEQ", lambda s: s.mean() * 100),
            )
            .reset_index()
            .sort_values("MES_NASC")
        )
    if not escolhidos:
        st.info("Selecione pelo menos um indicador.")
    else:
        figura = go.Figure()
        for posicao, indicador in enumerate(escolhidos):
            figura.add_trace(
                go.Scatter(
                    x=serie["MES_NASC"],
                    y=serie[indicador],
                    name=rotular(indicador),
                    mode="lines+markers",
                    yaxis="y" if posicao == 0 else "y2",
                )
            )
        figura.update_layout(
            xaxis_title="Mês de nascimento",
            yaxis=dict(title=rotular(escolhidos[0])),
            yaxis2=dict(
                title=rotular(escolhidos[1]) if len(escolhidos) > 1 else "",
                overlaying="y",
                side="right",
            ),
            height=480,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(figura, width="stretch")
        st.dataframe(serie, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Peso médio e baixo peso por semana gestacional")
    st.caption(
        "Objetivo: descrever o comportamento do peso e da proporção de baixo peso ao "
        "longo da duração da gestação, com IC por semana."
    )
    semanal = carregar("agregado_semana")
    coluna_semanal = st.radio(
        "Medida",
        ["media", "taxa_bpn"],
        horizontal=True,
        format_func=lambda v: "Peso médio (g)" if v == "media" else "Baixo peso (%)",
        key="sem_medida",
    )
    figura = go.Figure()
    if coluna_semanal == "media":
        figura.add_trace(
            go.Scatter(
                x=pd.concat([semanal["semana"], semanal["semana"][::-1]]),
                y=pd.concat([semanal["ic_sup"], semanal["ic_inf"][::-1]]),
                fill="toself",
                fillcolor="rgba(76,114,176,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="IC 95%",
                hoverinfo="skip",
            )
        )
    figura.add_trace(
        go.Scatter(
            x=semanal["semana"],
            y=semanal[coluna_semanal],
            mode="lines+markers",
            name="Peso médio (g)" if coluna_semanal == "media" else "Baixo peso (%)",
            line=dict(color="#4C72B0"),
            customdata=semanal["n"],
            hovertemplate="semana %{x}<br>%{y:.2f}<br>n: %{customdata:,}<extra></extra>",
        )
    )
    figura.add_vline(x=37, line_dash="dot", line_color="#937860")
    figura.update_layout(
        xaxis_title=rotular("SEMAGESTAC"),
        yaxis_title="Peso médio (g)" if coluna_semanal == "media" else "Baixo peso (%)",
        height=460,
    )
    st.plotly_chart(figura, width="stretch")

with aba_prevalencia:
    st.subheader("Prevalências com intervalo de confiança de Wilson")
    st.caption(
        "Objetivo: comparar a magnitude e a precisão das estimativas de proporção dos "
        "indicadores binários na população completa."
    )
    prevalencias = carregar("prevalencias")
    selecionadas = st.multiselect(
        "Indicadores",
        prevalencias["indicador"].tolist(),
        default=prevalencias["indicador"].tolist(),
        key="prev_ind",
    )
    tabela = prevalencias[prevalencias["indicador"].isin(selecionadas)].sort_values(
        "proporcao_%"
    )
    if tabela.empty:
        st.info("Selecione pelo menos um indicador.")
    else:
        figura = go.Figure(
            go.Scatter(
                x=tabela["proporcao_%"],
                y=tabela["indicador"],
                mode="markers",
                marker=dict(size=11),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=tabela["ic_superior_%"] - tabela["proporcao_%"],
                    arrayminus=tabela["proporcao_%"] - tabela["ic_inferior_%"],
                ),
                customdata=np.stack([tabela["casos"], tabela["n"]], axis=-1),
                hovertemplate="%{y}<br>%{x:.3f}%<br>casos: %{customdata[0]:,}<br>n: %{customdata[1]:,}<extra></extra>",
            )
        )
        figura.update_layout(
            xaxis_title="Proporção (%) com IC 95%", yaxis_title="", height=420
        )
        st.plotly_chart(figura, width="stretch")
        st.dataframe(tabela, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Prevalência por grupo no recorte filtrado")
    st.caption(
        "Objetivo: recalcular a proporção de um desfecho binário por categoria, com IC "
        "de Wilson no nível de confiança escolhido na barra lateral."
    )
    esquerda, direita = st.columns([1, 3])
    with esquerda:
        desfecho_binario = st.selectbox(
            "Desfecho",
            ["BAIXO_PESO", "PREMATURO", "CESAREA", "PRENATAL_ADEQ", "MULTIPLA", "ANOMALIA"],
            format_func=rotular,
            key="prev_desf",
        )
        categoria_binaria = st.selectbox(
            "Categoria",
            ["REGIAO", "UF", "ESC_CAT", "RACA_MAE", "FAIXA_MAE", "SEXO_M"],
            format_func=rotular,
            key="prev_cat",
        )
    with direita:
        chave = coluna_exibicao(categoria_binaria)
        linhas = []
        for valor, grupo in recorte.groupby(chave, observed=True):
            serie = grupo[desfecho_binario].dropna()
            if len(serie) >= 10:
                proporcao, inferior, superior = intervalo_wilson(
                    int(serie.sum()), len(serie), confianca
                )
                linhas.append(
                    {
                        "grupo": str(valor),
                        "proporcao": proporcao,
                        "inferior": proporcao - inferior,
                        "superior": superior - proporcao,
                        "n": len(serie),
                    }
                )
        if not linhas:
            st.info("Grupos com menos de 10 observações foram omitidos.")
        else:
            resumo = pd.DataFrame(linhas).sort_values("proporcao")
            figura = go.Figure(
                go.Scatter(
                    x=resumo["proporcao"],
                    y=resumo["grupo"],
                    mode="markers",
                    marker=dict(size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=resumo["superior"],
                        arrayminus=resumo["inferior"],
                    ),
                    customdata=resumo["n"],
                    hovertemplate="%{y}<br>%{x:.2f}%<br>n: %{customdata:,}<extra></extra>",
                )
            )
            figura.update_layout(
                xaxis_title=f"{rotular(desfecho_binario)} (%) — IC {confianca:.0%}",
                yaxis_title=rotular(categoria_binaria),
                height=max(400, 24 * len(resumo)),
            )
            st.plotly_chart(figura, width="stretch")

with aba_testes:
    st.subheader("Testes de hipóteses da análise")
    st.caption(
        "Objetivo: consultar estatística, graus de liberdade, valor-p, tamanho de efeito "
        "e intervalo de confiança de cada hipótese testada na população completa."
    )
    testes = carregar("testes_hipoteses")
    hipotese = st.selectbox("Hipótese", testes["hipotese"].tolist(), key="teste_hip")
    linha = testes[testes["hipotese"] == hipotese].iloc[0]
    colunas = st.columns(5)
    colunas[0].metric("Teste", linha["teste"])
    colunas[1].metric("Estatística", numero(linha["estatistica"], 3))
    colunas[2].metric("Graus de liberdade", numero(linha["graus_liberdade"]))
    colunas[3].metric(
        "Valor-p",
        "< 1e-300" if linha["valor_p"] == 0 else f"{linha['valor_p']:.3g}",
    )
    colunas[4].metric(linha["medida_efeito"], numero(linha["valor_efeito"], 4))

    comparaveis = testes.dropna(subset=["efeito", "ic_inferior", "ic_superior"])
    escala = st.radio(
        "Hipóteses no gráfico",
        ["Todas com IC disponível", "Apenas a selecionada"],
        horizontal=True,
        key="teste_escala",
    )
    if escala == "Apenas a selecionada":
        comparaveis = comparaveis[comparaveis["hipotese"] == hipotese]
    if comparaveis.empty:
        st.info("A hipótese selecionada não possui intervalo de confiança tabulado.")
    else:
        figura = go.Figure(
            go.Scatter(
                x=comparaveis["efeito"],
                y=comparaveis["hipotese"],
                mode="markers",
                marker=dict(size=11),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=comparaveis["ic_superior"] - comparaveis["efeito"],
                    arrayminus=comparaveis["efeito"] - comparaveis["ic_inferior"],
                ),
                hovertemplate="%{y}<br>efeito: %{x:.3f}<extra></extra>",
            )
        )
        figura.update_layout(
            xaxis_title="Estimativa do efeito com IC 95%", yaxis_title="", height=400
        )
        st.plotly_chart(figura, width="stretch")
    st.dataframe(testes, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Teste t de Welch sobre o recorte filtrado")
    st.caption(
        "Objetivo: executar a comparação de duas médias independentes sobre os dados "
        "filtrados, obtendo diferença pontual, IC e valor-p no nível escolhido."
    )
    esquerda, direita = st.columns([1, 2])
    with esquerda:
        desfecho_teste = st.selectbox(
            "Desfecho", CONTINUAS, format_func=rotular, key="welch_desf"
        )
        categoria_teste = st.selectbox(
            "Variável de agrupamento", CATEGORICAS, format_func=rotular, key="welch_cat"
        )
        chave = coluna_exibicao(categoria_teste)
        niveis = sorted(recorte[chave].dropna().unique().tolist())
        if len(niveis) < 2:
            st.info("A variável selecionada tem menos de dois grupos no recorte.")
            grupo_a = grupo_b = None
        else:
            grupo_a = st.selectbox("Grupo A", niveis, index=0, key="welch_a")
            grupo_b = st.selectbox("Grupo B", niveis, index=1, key="welch_b")
    with direita:
        if grupo_a is not None and grupo_a != grupo_b:
            resultado = welch(
                recorte.loc[recorte[chave] == grupo_a, desfecho_teste],
                recorte.loc[recorte[chave] == grupo_b, desfecho_teste],
                confianca,
            )
            if resultado is None:
                st.info("Grupos com observações insuficientes para o teste.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "medida": [
                                f"Média {grupo_a}",
                                f"Média {grupo_b}",
                                "Diferença (A − B)",
                                f"IC {confianca:.0%} inferior",
                                f"IC {confianca:.0%} superior",
                                "Estatística t",
                                "Graus de liberdade",
                                "Valor-p",
                                "d de Cohen",
                                f"n {grupo_a}",
                                f"n {grupo_b}",
                            ],
                            "valor": [
                                resultado["media_a"],
                                resultado["media_b"],
                                resultado["diferenca"],
                                resultado["ic_inferior"],
                                resultado["ic_superior"],
                                resultado["estatistica"],
                                resultado["gl"],
                                resultado["valor_p"],
                                resultado["d_cohen"],
                                resultado["n_a"],
                                resultado["n_b"],
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                figura = go.Figure(
                    go.Scatter(
                        x=[resultado["diferenca"]],
                        y=["A − B"],
                        mode="markers",
                        marker=dict(size=13),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=[resultado["ic_superior"] - resultado["diferenca"]],
                            arrayminus=[resultado["diferenca"] - resultado["ic_inferior"]],
                        ),
                    )
                )
                figura.add_vline(x=0, line_dash="dash", line_color="#C44E52")
                figura.update_layout(
                    xaxis_title=f"Diferença de médias — {rotular(desfecho_teste)}",
                    yaxis_title="",
                    height=240,
                )
                st.plotly_chart(figura, width="stretch")
        elif grupo_a is not None:
            st.info("Selecione dois grupos distintos.")

with aba_regressao:
    st.subheader("Comparação dos modelos estimados")
    st.caption(
        "Objetivo: confrontar os cinco modelos por ajuste, parcimônia e desempenho fora "
        "da amostra."
    )
    comparacao = carregar("comparacao_modelos")
    metrica_modelo = st.selectbox(
        "Métrica",
        ["R2_teste", "R2_treino", "R2_ajustado", "RMSE_teste", "MAE_teste", "AIC", "BIC"],
        key="reg_metrica",
    )
    figura = px.bar(
        comparacao,
        x="modelo",
        y=metrica_modelo,
        labels={"modelo": "Modelo", metrica_modelo: metrica_modelo},
        hover_data=["k_parametros", "gl_residuos"],
    )
    figura.update_layout(height=400)
    st.plotly_chart(figura, width="stretch")
    st.dataframe(comparacao, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Coeficientes do modelo M3 com erros-padrão HC3")
    st.caption(
        "Objetivo: ler a magnitude, o sinal e a precisão de cada coeficiente em gramas, "
        "com intervalo de confiança robusto à heterocedasticidade."
    )
    tabela_coeficientes = carregar("coeficientes_m3")
    sem_intercepto = tabela_coeficientes[tabela_coeficientes["termo"] != "Intercept"]
    termos = st.multiselect(
        "Termos",
        sem_intercepto["termo"].tolist(),
        default=sem_intercepto["termo"].tolist(),
        key="reg_termos",
    )
    recorte_coeficientes = sem_intercepto[sem_intercepto["termo"].isin(termos)].sort_values(
        "coeficiente"
    )
    if recorte_coeficientes.empty:
        st.info("Selecione pelo menos um termo.")
    else:
        figura = go.Figure(
            go.Scatter(
                x=recorte_coeficientes["coeficiente"],
                y=recorte_coeficientes["termo"],
                mode="markers",
                marker=dict(size=10),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=recorte_coeficientes["ic_superior"]
                    - recorte_coeficientes["coeficiente"],
                    arrayminus=recorte_coeficientes["coeficiente"]
                    - recorte_coeficientes["ic_inferior"],
                ),
                customdata=recorte_coeficientes["valor_p_hc3"],
                hovertemplate="%{y}<br>%{x:.2f} g<br>p: %{customdata:.3g}<extra></extra>",
            )
        )
        figura.add_vline(x=0, line_dash="dash", line_color="#C44E52")
        figura.update_layout(
            xaxis_title="Efeito sobre o peso ao nascer (g)",
            yaxis_title="",
            height=max(420, 26 * len(recorte_coeficientes)),
        )
        st.plotly_chart(figura, width="stretch")
        st.dataframe(tabela_coeficientes, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Diagnóstico dos resíduos no recorte filtrado")
    st.caption(
        "Objetivo: verificar os pressupostos de linearidade, homocedasticidade e "
        "normalidade aplicando o modelo M3 ao recorte ativo."
    )
    colunas_modelo = [
        "PESO",
        "SEMAGESTAC",
        "IDADEMAE",
        "CONSPRENAT",
        "MESPRENAT",
        "ANOS_ESTUDO",
        "QTDFILVIVO",
        "SEXO_M",
        "CESAREA",
        "MULTIPLA",
        "ANOMALIA",
        "SEM_COMPANHEIRO",
        "REGIAO",
        "RACA_MAE",
    ]
    base_modelo = recorte[colunas_modelo].dropna()
    if len(base_modelo) < 50:
        st.info("O recorte tem observações insuficientes para o diagnóstico.")
    else:
        if len(base_modelo) > 30000:
            base_modelo = base_modelo.sample(30000, random_state=7)
        ajustado = prever_vetorizado(coeficientes, base_modelo)
        residuo = base_modelo["PESO"] - ajustado
        colunas = st.columns(4)
        variacao_total = (
            (base_modelo["PESO"] - base_modelo["PESO"].mean()) ** 2
        ).sum()
        colunas[0].metric("n", numero(len(base_modelo)))
        colunas[1].metric("Erro médio (g)", numero(residuo.mean(), 2))
        colunas[2].metric("RMSE (g)", numero(np.sqrt((residuo**2).mean()), 1))
        colunas[3].metric(
            "R² no recorte", numero(1 - (residuo**2).sum() / variacao_total, 4)
        )
        grafico = st.radio(
            "Gráfico",
            ["Resíduo vs ajustado", "Observado vs previsto", "Distribuição do resíduo", "Quantis normais"],
            horizontal=True,
            key="reg_diag",
        )
        if grafico == "Resíduo vs ajustado":
            figura = px.scatter(
                x=ajustado,
                y=residuo,
                opacity=0.25,
                labels={"x": "Peso previsto (g)", "y": "Resíduo (g)"},
            )
            figura.add_hline(y=0, line_color="#C44E52")
        elif grafico == "Observado vs previsto":
            limites = [
                min(ajustado.min(), base_modelo["PESO"].min()),
                max(ajustado.max(), base_modelo["PESO"].max()),
            ]
            figura = px.scatter(
                x=ajustado,
                y=base_modelo["PESO"],
                opacity=0.25,
                labels={"x": "Peso previsto (g)", "y": "Peso observado (g)"},
            )
            figura.add_trace(
                go.Scatter(
                    x=limites,
                    y=limites,
                    mode="lines",
                    line=dict(color="#C44E52"),
                    name="Identidade",
                )
            )
        elif grafico == "Distribuição do resíduo":
            figura = px.histogram(
                x=residuo, nbins=80, labels={"x": "Resíduo (g)"}, histnorm="probability density"
            )
            grade = np.linspace(residuo.min(), residuo.max(), 200)
            figura.add_trace(
                go.Scatter(
                    x=grade,
                    y=stats.norm.pdf(grade, residuo.mean(), residuo.std(ddof=1)),
                    mode="lines",
                    line=dict(color="#C44E52"),
                    name="Normal de referência",
                )
            )
        else:
            padronizado = ((residuo - residuo.mean()) / residuo.std(ddof=1)).sort_values()
            teoricos = stats.norm.ppf(
                (np.arange(1, len(padronizado) + 1) - 0.5) / len(padronizado)
            )
            figura = px.scatter(
                x=teoricos,
                y=padronizado.to_numpy(),
                opacity=0.3,
                labels={"x": "Quantis teóricos", "y": "Resíduo padronizado"},
            )
            figura.add_trace(
                go.Scatter(
                    x=[teoricos.min(), teoricos.max()],
                    y=[teoricos.min(), teoricos.max()],
                    mode="lines",
                    line=dict(color="#C44E52"),
                    name="Referência",
                )
            )
        figura.update_layout(height=480, showlegend=False)
        st.plotly_chart(figura, width="stretch")

with aba_simulador:
    st.subheader("Previsão do peso ao nascer pelo modelo M3")
    st.caption(
        "Objetivo: gerar a previsão pontual, o intervalo de previsão e a probabilidade "
        "estimada de baixo peso para uma combinação de características definida pelo usuário."
    )
    sigma = sigma_residual()
    esquerda, direita = st.columns([1, 2])
    with esquerda:
        entrada = {}
        entrada["SEMAGESTAC"] = st.slider("Semanas de gestação", 22, 44, 39, key="sim_sem")
        entrada["IDADEMAE"] = st.slider("Idade da mãe", 10, 55, 28, key="sim_idade")
        entrada["CONSPRENAT"] = st.slider("Consultas de pré-natal", 0, 20, 8, key="sim_cons")
        entrada["MESPRENAT"] = st.slider("Mês de início do pré-natal", 1, 9, 3, key="sim_mes")
        entrada["ANOS_ESTUDO"] = st.slider("Anos de estudo da mãe", 0, 20, 11, key="sim_esc")
        entrada["QTDFILVIVO"] = st.slider("Filhos vivos anteriores", 0, 10, 1, key="sim_filhos")
        entrada["REGIAO"] = st.selectbox(
            "Região", sorted(amostra["REGIAO"].dropna().unique().tolist()), key="sim_regiao"
        )
        entrada["RACA_MAE"] = st.selectbox(
            "Raça/cor da mãe",
            sorted(amostra["RACA_MAE"].dropna().unique().tolist()),
            key="sim_raca",
        )
        entrada["SEXO_M"] = 1.0 if st.radio(
            "Sexo", ["Masculino", "Feminino"], key="sim_sexo"
        ) == "Masculino" else 0.0
        entrada["CESAREA"] = 1.0 if st.radio(
            "Via de parto", ["Cesárea", "Vaginal"], key="sim_parto"
        ) == "Cesárea" else 0.0
        entrada["MULTIPLA"] = 1.0 if st.checkbox("Gestação múltipla", key="sim_mult") else 0.0
        entrada["ANOMALIA"] = 1.0 if st.checkbox("Anomalia congênita", key="sim_anom") else 0.0
        entrada["SEM_COMPANHEIRO"] = 1.0 if st.checkbox(
            "Mãe sem companheiro", key="sim_comp"
        ) else 0.0
    with direita:
        predito = prever_peso(coeficientes, entrada)
        z = stats.norm.ppf(0.5 + confianca / 2)
        colunas = st.columns(3)
        colunas[0].metric("Peso previsto (g)", numero(predito))
        colunas[1].metric(
            f"Intervalo de previsão {confianca:.0%} (g)",
            f"{numero(predito - z * sigma)} – {numero(predito + z * sigma)}",
        )
        colunas[2].metric(
            "P(peso < 2.500 g)",
            f"{numero(stats.norm.cdf((LIMIAR_BPN - predito) / sigma) * 100, 1)}%",
        )

        eixo = st.selectbox(
            "Variar no eixo",
            ["SEMAGESTAC", "CONSPRENAT", "IDADEMAE", "ANOS_ESTUDO", "MESPRENAT", "QTDFILVIVO"],
            format_func=rotular,
            key="sim_eixo",
        )
        limites = {
            "SEMAGESTAC": range(22, 45),
            "CONSPRENAT": range(0, 21),
            "IDADEMAE": range(10, 56),
            "ANOS_ESTUDO": range(0, 21),
            "MESPRENAT": range(1, 10),
            "QTDFILVIVO": range(0, 11),
        }
        curva = []
        for valor in limites[eixo]:
            variante = dict(entrada)
            variante[eixo] = valor
            peso = prever_peso(coeficientes, variante)
            curva.append(
                {
                    eixo: valor,
                    "previsto": peso,
                    "inferior": peso - z * sigma,
                    "superior": peso + z * sigma,
                    "p_bpn": stats.norm.cdf((LIMIAR_BPN - peso) / sigma) * 100,
                }
            )
        curva = pd.DataFrame(curva)
        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=pd.concat([curva[eixo], curva[eixo][::-1]]),
                y=pd.concat([curva["superior"], curva["inferior"][::-1]]),
                fill="toself",
                fillcolor="rgba(76,114,176,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"Intervalo de previsão {confianca:.0%}",
                hoverinfo="skip",
            )
        )
        figura.add_trace(
            go.Scatter(
                x=curva[eixo],
                y=curva["previsto"],
                mode="lines+markers",
                line=dict(color="#4C72B0"),
                name="Peso previsto (g)",
            )
        )
        figura.add_hline(y=LIMIAR_BPN, line_dash="dash", line_color="#C44E52")
        figura.add_vline(x=entrada[eixo], line_dash="dot", line_color="#937860")
        figura.update_layout(
            xaxis_title=rotular(eixo),
            yaxis_title="Peso ao nascer (g)",
            height=400,
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(figura, width="stretch")

        figura = px.line(
            curva,
            x=eixo,
            y="p_bpn",
            labels={eixo: rotular(eixo), "p_bpn": "P(peso < 2.500 g) (%)"},
            markers=True,
        )
        figura.add_vline(x=entrada[eixo], line_dash="dot", line_color="#937860")
        figura.update_layout(height=320)
        st.plotly_chart(figura, width="stretch")

    st.divider()
    st.subheader("Comparação de cenários de pré-natal")
    st.caption(
        "Objetivo: contrastar a previsão entre combinações de consultas e mês de início "
        "do pré-natal, mantidas fixas as demais características definidas acima."
    )
    consultas_cenario = st.multiselect(
        "Consultas de pré-natal",
        list(range(0, 21)),
        default=[1, 3, 6, 9, 12],
        key="cen_cons",
    )
    meses_cenario = st.multiselect(
        "Mês de início", list(range(1, 10)), default=[1, 3, 6, 9], key="cen_mes"
    )
    if not consultas_cenario or not meses_cenario:
        st.info("Selecione ao menos um valor em cada eixo.")
    else:
        grade = []
        for consultas in sorted(consultas_cenario):
            for mes in sorted(meses_cenario):
                variante = dict(entrada)
                variante["CONSPRENAT"] = consultas
                variante["MESPRENAT"] = mes
                grade.append(
                    {
                        "consultas": consultas,
                        "mes_inicio": mes,
                        "previsto": prever_peso(coeficientes, variante),
                    }
                )
        grade = pd.DataFrame(grade).pivot(
            index="mes_inicio", columns="consultas", values="previsto"
        )
        figura = px.imshow(
            grade,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(
                x="Consultas de pré-natal",
                y="Mês de início do pré-natal",
                color="Peso previsto (g)",
            ),
        )
        figura.update_layout(height=380)
        st.plotly_chart(figura, width="stretch")
